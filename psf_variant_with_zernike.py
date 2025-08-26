```python
import math
import time
import unittest
import warnings
from dataclasses import dataclass
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import numpy.typing as npt

# Define module-level constants for clarity
ARCSEC_TO_RAD = math.pi / (180.0 * 3600.0)


def zernike_radial(n: int, m: int, rho: torch.Tensor) -> torch.Tensor:
    """
    Computes the radial Zernike polynomial R_n^m(rho) in a vectorized way.
    Implements the Noll (1976) indexing convention and formula.

    Args:
        n (int): The radial order.
        m (int): The azimuthal order.
        rho (torch.Tensor): A tensor of radial coordinates (0 to 1), with any shape.
                            The polynomial is computed element-wise.

    Returns:
        torch.Tensor: The Zernike radial polynomial evaluated at rho, same shape as rho.
    """
    if (n - m) % 2 != 0:
        raise ValueError(f"Invalid Zernike indices: (n-m) must be even. Got n={n}, m={m}.")
    m = abs(m)

    s_max = (n - m) // 2
    k = torch.arange(0, s_max + 1, device=rho.device)

    alternating_sign = (-1.0) ** k

    # Use log-gamma to compute log(n!) for numerical stability with large n.
    log_factorial_terms = (
        torch.lgamma(n - k + 1.0)
        - torch.lgamma(k + 1.0)
        - torch.lgamma((n + m) / 2.0 - k + 1.0)
        - torch.lgamma((n - m) / 2.0 - k + 1.0)
    )

    coeffs = alternating_sign * torch.exp(log_factorial_terms)
    # Use torch.pow to correctly handle 0**0 = 1 for the central pixel.
    powers = (n - 2 * k).float()
    rho_powers = torch.pow(rho.unsqueeze(-1), powers)
    radial_poly = torch.sum(coeffs * rho_powers, dim=-1)

    return radial_poly


def generate_psf_batch(
    zernike_indices: List[Tuple[int, int]],
    coefficient_batch: torch.Tensor,
    psf_npix: int,
    wavelength: float,
    pixel_scale: float,
    aperture_diameter: float,
    device: torch.device,
    pupil_oversample: int = 2,
) -> torch.Tensor:
    """
    Generates a batch of physically-scaled Point Spread Functions (PSFs).
    The Zernike indices and coefficients must follow the Noll (1976) convention.

    Args:
        zernike_indices (List[Tuple[int, int]]): A list of (n, m) Zernike index tuples.
        coefficient_batch (torch.Tensor): Tensor of Zernike coefficients of shape
            `(B, num_indices)`. Values are in meters of RMS wavefront error.
        psf_npix (int): The side length of the final output PSF array.
        wavelength (float): The wavelength of light in meters.
        pixel_scale (float): The target pixel scale in arcseconds/pixel.
        aperture_diameter (float): The diameter of the telescope aperture in meters.
        device (torch.device): The device (CPU or CUDA) for computation.
        pupil_oversample (int): The sampling factor for the pupil plane grid. Higher
            values better sample the pupil-plane phase, preventing aliasing in
            the final PSF, at the cost of increased memory and computation.

    Returns:
        torch.Tensor: A batch of normalized 2D PSFs of shape `(B, psf_npix, psf_npix)`.
    """
    if psf_npix % 2 == 0:
        raise ValueError("PSF size must be odd for an unambiguous center pixel.")
    if coefficient_batch.shape[1] != len(zernike_indices):
        raise ValueError(
            f"Dimension 1 of coefficient_batch ({coefficient_batch.shape[1]}) must "
            f"match the number of Zernike indices ({len(zernike_indices)})."
        )
    if pupil_oversample < 2:
        warnings.warn(
            f"pupil_oversample={pupil_oversample} is less than 2, which may lead "
            "to aliasing in the PSF. A value of 2 or higher is recommended."
        )

    pupil_npix = psf_npix * pupil_oversample

    y, x = torch.meshgrid(
        torch.linspace(-1, 1, pupil_npix, device=device),
        torch.linspace(-1, 1, pupil_npix, device=device),
        indexing="ij",
    )
    rho = torch.sqrt(x**2 + y**2)
    theta = torch.atan2(y, x)
    aperture_mask = (rho <= 1.0).float()

    # Vectorized Zernike basis generation for high performance
    zernike_basis = torch.stack(
        tuple(
            zernike_radial(n, m, rho)
            * (torch.cos(m * theta) if m >= 0 else torch.sin(abs(m) * theta))
            for n, m in zernike_indices
        )
    )
    # Combine basis with coefficients to get the Optical Path Difference (OPD) map
    opd_map = torch.einsum("bc,chw->bhw", coefficient_batch, zernike_basis)

    phase_error = (2 * torch.pi * opd_map) / wavelength
    pupil_function = aperture_mask * torch.exp(1j * phase_error)

    # The pupil function is padded before the FFT. This is equivalent to interpolating
    # the PSF in the focal plane, allowing us to precisely set the output pixel scale.
    native_scale_rad = wavelength / aperture_diameter
    target_scale_rad = pixel_scale * ARCSEC_TO_RAD
    zoom_factor = native_scale_rad / target_scale_rad
    fft_size = int(round(pupil_npix * zoom_factor))

    pad_total = fft_size - pupil_npix
    pad_before = pad_total // 2
    pad_after = pad_total - pad_before
    padded_pupil = F.pad(
        pupil_function, (pad_before, pad_after, pad_before, pad_after)
    )

    # Apply fftshift to center the zero-frequency component, which corresponds to the PSF's peak.
    fft_pupil = torch.fft.fftshift(torch.fft.fft2(padded_pupil), dim=(-2, -1))
    psf_full = torch.abs(fft_pupil) ** 2

    start = fft_size // 2 - psf_npix // 2
    end = start + psf_npix
    psf_batch = psf_full[:, start:end, start:end]

    psf_batch /= torch.sum(psf_batch, dim=(-2, -1), keepdim=True)
    return psf_batch


def spatially_variant_convolve_torch(
    image_tensor: torch.Tensor, psf_grid: torch.Tensor
) -> torch.Tensor:
    """
    Core PyTorch implementation of spatially variant convolution.

    This function implements a high-performance patch-based method using the
    'im2col' technique (realized via `F.unfold`). It pads the image to handle
    arbitrary sizes and boundary effects, then performs a single, massively
    parallel grouped convolution to apply a unique PSF to each image patch.

    Note:
        - This method creates hard boundaries between patches convolved with different
          PSFs. For smoothly varying PSFs, this is often a good approximation.
        - The `unfold` operation has a high memory footprint and may not be suitable
          for extremely large images on memory-constrained hardware.

    Args:
        image_tensor (torch.Tensor): The 2D input image tensor (H, W).
        psf_grid (torch.Tensor): A 4D tensor of shape `(G_h, G_w, P_h, P_w)`
            containing the PSFs.

    Returns:
        torch.Tensor: The convolved 2D image tensor with the original shape.
    """
    if image_tensor.ndim != 2:
        raise ValueError(f"Input image must be 2D, but got shape {image_tensor.shape}")

    h_orig, w_orig = image_tensor.shape
    grid_h, grid_w, psf_h, psf_w = psf_grid.shape
    num_patches = grid_h * grid_w

    if psf_h % 2 == 0 or psf_w % 2 == 0:
        raise ValueError("PSF dimensions must be odd for centered convolution.")

    h_padded_grid = math.ceil(h_orig / grid_h) * grid_h
    w_padded_grid = math.ceil(w_orig / grid_w) * grid_w
    patch_h = h_padded_grid // grid_h
    patch_w = w_padded_grid // grid_w

    if psf_h > patch_h or psf_w > patch_w:
        raise ValueError(
            f"PSF dimensions ({psf_h}x{psf_w}) cannot be larger than "
            f"the image patch dimensions ({patch_h}x{patch_w})."
        )

    image_padded_for_grid = F.pad(
        image_tensor, (0, w_padded_grid - w_orig, 0, h_padded_grid - h_orig)
    )

    pad_h_conv = (psf_h - 1) // 2
    pad_w_conv = (psf_w - 1) // 2
    image_padded_for_conv = F.pad(
        image_padded_for_grid, (pad_w_conv, pad_w_conv, pad_h_conv, pad_h_conv)
    )

    patch_h_with_padding = patch_h + 2 * pad_h_conv
    patch_w_with_padding = patch_w + 2 * pad_w_conv
    patches_as_columns = F.unfold(
        image_padded_for_conv.unsqueeze(0).unsqueeze(0),
        kernel_size=(patch_h_with_padding, patch_w_with_padding),
        stride=(patch_h, patch_w),
    )

    patches_as_batch = patches_as_columns.permute(0, 2, 1).reshape(
        1, num_patches, patch_h_with_padding, patch_w_with_padding
    )
    psfs = psf_grid.reshape(num_patches, 1, psf_h, psf_w)

    convolved_patches = F.conv2d(
        patches_as_batch, psfs, padding="valid", groups=num_patches
    )

    output_image_padded = (
        convolved_patches.view(grid_h, grid_w, patch_h, patch_w)
        .permute(0, 2, 1, 3)
        .reshape(h_padded_grid, w_padded_grid)
    )

    output_image = output_image_padded[:h_orig, :w_orig]
    return output_image


def spatially_variant_convolve(
    image: npt.NDArray[np.float32], psf_grid: torch.Tensor
) -> npt.NDArray[np.float32]:
    """
    NumPy convenience wrapper for the high-performance PyTorch convolution function.

    Convolves a 2D NumPy image with a spatially varying PSF grid. This wrapper
    handles the conversion to PyTorch tensors, moves data to the device inferred
    from the PSF grid, and converts the result back to a NumPy array.

    Args:
        image (npt.NDArray[np.float32]): The 2D input image as a NumPy array.
        psf_grid (torch.Tensor): A 4D tensor of shape `(G_h, G_w, P_h, P_w)`
            containing the PSFs.

    Returns:
        npt.NDArray[np.float32]: The convolved 2D image as a NumPy array.
    """
    if image.dtype != np.float32:
        warnings.warn(
            f"Input image dtype is {image.dtype}, casting to np.float32. "
            "For optimal performance, provide a float32 array."
        )
        image = image.astype(np.float32)

    device = psf_grid.device
    image_tensor = torch.from_numpy(image).to(device)
    result_tensor = spatially_variant_convolve_torch(image_tensor, psf_grid)
    return result_tensor.cpu().numpy()


def visualize_results(
    original_image: npt.NDArray[np.float32],
    convolved_image: npt.NDArray[np.float32],
    psf_grid: torch.Tensor,
    filename: str = "convolution_results.png",
) -> None:
    """Generates and saves plots to visualize the simulation results."""
    grid_h, grid_w = psf_grid.shape[:2]
    psf_samples = {
        f"PSF [0, 0]": psf_grid[0, 0],
        f"PSF [{grid_h - 1}, 0]": psf_grid[grid_h - 1, 0],
        f"PSF [{grid_h - 1}, {grid_w - 1}]": psf_grid[grid_h - 1, grid_w - 1],
    }

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Spatially Variant Convolution Simulation", fontsize=16)

    ax = axes[0, 0]
    im = ax.imshow(original_image, cmap="gray", origin="lower")
    ax.set_title("Original Image")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[0, 1]
    im = ax.imshow(convolved_image, cmap="gray", origin="lower")
    ax.set_title("Convolved Image")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    diff = np.abs(original_image - convolved_image)
    ax = axes[0, 2]
    im = ax.imshow(diff, cmap="hot", origin="lower")
    ax.set_title("Absolute Difference")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for i, (title, psf) in enumerate(psf_samples.items()):
        ax = axes[1, i]
        psf_log = np.log1p(psf.cpu().numpy())
        im = ax.imshow(psf_log, cmap="viridis", origin="lower")
        ax.set_title(title)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(filename)
    print(f"   Visualization saved to {filename}")
    plt.close(fig)


class TestSimulation(unittest.TestCase):
    def setUp(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.grid_size = 8
        self.psf_size = 17

    def test_identity_convolution(self) -> None:
        image = np.random.randn(256, 256).astype(np.float32)
        delta_psf = torch.zeros(self.psf_size, self.psf_size, device=self.device)
        delta_psf[self.psf_size // 2, self.psf_size // 2] = 1.0
        psf_grid = delta_psf.view(1, 1, self.psf_size, self.psf_size).repeat(
            self.grid_size, self.grid_size, 1, 1
        )
        convolved_image = spatially_variant_convolve(image, psf_grid)
        self.assertTrue(np.allclose(image, convolved_image, atol=1e-7))

    def test_core_flux_conservation(self) -> None:
        constant_image = np.ones((256, 256), dtype=np.float32)
        psf_grid = torch.rand(
            (self.grid_size, self.grid_size, self.psf_size, self.psf_size),
            device=self.device,
        )
        psf_grid /= psf_grid.sum(dim=(-2, -1), keepdim=True)
        convolved_image = spatially_variant_convolve(constant_image, psf_grid)

        inset = self.psf_size
        core_orig = constant_image[inset:-inset, inset:-inset]
        core_conv = convolved_image[inset:-inset, inset:-inset]
        self.assertTrue(np.allclose(core_conv, core_orig, atol=1e-5))

    def test_spatial_variance(self) -> None:
        image_size = 256
        psf_grid = torch.zeros(
            (self.grid_size, self.grid_size, self.psf_size, self.psf_size),
            device=self.device,
        )
        delta_psf = torch.zeros(self.psf_size, self.psf_size, device=self.device)
        delta_psf[self.psf_size // 2, self.psf_size // 2] = 1.0
        blur_psf = torch.ones(3, 3, device=self.device) / 9.0
        p_start, p_end = self.psf_size // 2 - 1, self.psf_size // 2 + 2

        psf_grid[: self.grid_size // 2, ...] = delta_psf
        psf_grid[self.grid_size // 2 :, :, p_start:p_end, p_start:p_end] = blur_psf

        patch_size = image_size // self.grid_size
        y, x = np.meshgrid(np.arange(image_size), np.arange(image_size))
        checkerboard = ((x // patch_size) % 2 == (y // patch_size) % 2).astype(np.float32)

        convolved = spatially_variant_convolve(checkerboard, psf_grid)

        mid_y = image_size // 2
        top_orig, top_conv = checkerboard[:mid_y, :], convolved[:mid_y, :]
        bottom_orig, bottom_conv = checkerboard[mid_y:, :], convolved[mid_y:, :]

        self.assertTrue(np.allclose(top_orig, top_conv, atol=1e-5))
        self.assertFalse(np.allclose(bottom_orig, bottom_conv, atol=1e-3))
        self.assertLess(np.std(bottom_conv), np.std(bottom_orig))

    def test_arbitrary_image_size(self) -> None:
        image_odd = np.random.randn(251, 253).astype(np.float32)
        psf_grid = torch.rand(
            (self.grid_size, self.grid_size, self.psf_size, self.psf_size),
            device=self.device,
        )
        convolved_image = spatially_variant_convolve(image_odd, psf_grid)
        self.assertEqual(image_odd.shape, convolved_image.shape)

    def test_large_psf_error(self) -> None:
        image = np.zeros((32, 32), dtype=np.float32)
        psf_grid = torch.ones(8, 8, 33, 33, device=self.device)
        with self.assertRaises(ValueError):
            spatially_variant_convolve(image, psf_grid)


def create_spatially_variant_zernike_coeffs(
    grid_shape: Tuple[int, int], max_astigmatism: float, max_coma: float, device: torch.device
) -> Tuple[List[Tuple[int, int]], torch.Tensor]:
    """Generates a smoothly varying batch of Zernike coefficients across a grid."""
    zernike_indices = [(2, -2), (2, 2), (3, -1), (3, 1)]
    grid_h, grid_w = grid_shape

    i, j = torch.meshgrid(
        torch.linspace(-1, 1, grid_h, device=device),
        torch.linspace(-1, 1, grid_w, device=device),
        indexing="ij",
    )
    i, j = i.flatten(), j.flatten()

    coefficients = torch.stack([
        i * max_astigmatism,
        j * max_astigmatism,
        (i * j) * max_coma,
        (i ** 2 - j ** 2) * max_coma
    ], dim=1)

    return zernike_indices, coefficients


@dataclass(frozen=True)
class SimConfig:
    image_shape: Tuple[int, int] = (256, 256)
    psf_size: int = 31
    grid_shape: Tuple[int, int] = (8, 8)
    pupil_oversample: int = 2
    wavelength: float = 550e-9
    pixel_scale: float = 0.05
    aperture_diameter: float = 2.4
    max_astigmatism_wfe: float = 150e-9
    max_coma_wfe: float = 100e-9

    def __post_init__(self):
        patch_h = math.ceil(self.image_shape[0] / self.grid_shape[0])
        patch_w = math.ceil(self.image_shape[1] / self.grid_shape[1])
        if self.psf_size > patch_h or self.psf_size > patch_w:
            raise ValueError(
                f"psf_size ({self.psf_size}) cannot be larger than the effective image patch size "
                f"({patch_h}x{patch_w})."
            )


def get_demo_image(shape: Tuple[int, int]) -> npt.NDArray[np.float32]:
    """Creates a checkerboard image for an unambiguous demonstration of blurring."""
    print("\n1. Creating a checkerboard test image for a clear demonstration...")
    patch_size = shape[0] // 8
    checker_size = patch_size // 4
    y, x = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]))
    image = ((x // checker_size) % 2 == (y // checker_size) % 2).astype(np.float32)
    return image


def run_benchmark(psf_grid: torch.Tensor, device: torch.device):
    """Runs a simple performance benchmark on a large image."""
    print(f"\n--- Running Performance Benchmark (1024x1024) on {device.type.upper()} ---")
    large_image = np.random.rand(1024, 1024).astype(np.float32)
    num_runs = 10

    if device.type == 'cuda':
        # Warm-up run to handle JIT compilation and memory allocation
        _ = spatially_variant_convolve(large_image, psf_grid)
        torch.cuda.synchronize()

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        for _ in range(num_runs):
            _ = spatially_variant_convolve(large_image, psf_grid)
        end_event.record()
        torch.cuda.synchronize()
        elapsed_time_ms = start_event.elapsed_time(end_event)
        avg_time = elapsed_time_ms / (1000.0 * num_runs)
        print(f"   Convolution on 1024x1024 image took: {avg_time:.4f} seconds (avg over {num_runs} runs)")
    else:  # CPU timing
        _ = spatially_variant_convolve(large_image, psf_grid)
        start_time = time.perf_counter()
        for _ in range(num_runs):
            _ = spatially_variant_convolve(large_image, psf_grid)
        end_time = time.perf_counter()
        avg_time = (end_time - start_time) / num_runs
        print(f"   Convolution on 1024x1024 image took: {avg_time:.4f} seconds (avg over {num_runs} runs)")


def main():
    """Main execution function for the simulation."""
    cfg = SimConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    image = get_demo_image(cfg.image_shape)
    print(f"   Input image shape: {image.shape}")

    print("\n2. Generating a spatially varying PSF grid...")
    start_time = time.perf_counter()
    zernike_indices, coefficients = create_spatially_variant_zernike_coeffs(
        cfg.grid_shape, cfg.max_astigmatism_wfe, cfg.max_coma_wfe, device
    )
    psf_grid = generate_psf_batch(
        zernike_indices=zernike_indices, coefficient_batch=coefficients, psf_npix=cfg.psf_size,
        wavelength=cfg.wavelength, pixel_scale=cfg.pixel_scale,
        aperture_diameter=cfg.aperture_diameter, device=device,
        pupil_oversample=cfg.pupil_oversample
    ).reshape(*cfg.grid_shape, cfg.psf_size, cfg.psf_size)
    end_time = time.perf_counter()
    print(f"   PSF grid generation took: {end_time - start_time:.4f} seconds")

    print("\n3. Performing spatially variant convolution...")
    start_time = time.perf_counter()
    convolved_image = spatially_variant_convolve(image, psf_grid)
    end_time = time.perf_counter()
    print(f"   Convolution on {image.shape[0]}x{image.shape[1]} image took: {end_time - start_time:.4f} seconds")

    print("\n4. Output image statistics:")
    inset = cfg.psf_size
    core_flux_ratio = np.sum(convolved_image[inset:-inset, inset:-inset]) / np.sum(image[inset:-inset, inset:-inset]) * 100
    print(f"   Core flux preservation: {core_flux_ratio:.2f}%")
    flux_ratio = convolved_image.sum() / image.sum() * 100
    print(f"   Total output/input flux ratio: {flux_ratio:.2f}% (Note: <100% is due to flux convolved off-image)")

    patch_h = image.shape[0] // cfg.grid_shape[0]
    patch_w = image.shape[1] // cfg.grid_shape[1]
    corners = {
        "Top-left": (slice(0, patch_h), slice(0, patch_w)),
        "Top-right": (slice(0, patch_h), slice(-patch_w, None)),
        "Bottom-left": (slice(-patch_h, None), slice(0, patch_w)),
        "Bottom-right": (slice(-patch_h, None), slice(-patch_w, None)),
    }
    print("   Standard deviation change in corner patches (blurring reduces std dev):")
    for name, s in corners.items():
        std_orig = np.std(image[s])
        std_conv = np.std(convolved_image[s])
        print(f"     {name:<15}: {std_orig:.4f} -> {std_conv:.4f}")

    print("\n5. Visualizing results...")
    visualize_results(image, convolved_image, psf_grid)

    run_benchmark(psf_grid, device)


if __name__ == "__main__":
    main()
    print("\nTo run the comprehensive unit test suite, execute the following command:")
    print(f"python -m unittest {__file__}")
```
