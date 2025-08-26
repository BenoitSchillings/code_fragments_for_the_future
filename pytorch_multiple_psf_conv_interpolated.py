python -m unittest spatially_varying_convolution.py
"""
import time
import timeit
import unittest
from typing import Annotated, Literal, TypeAlias, TypeVar

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray

# Type aliases for clarity and to document expected shapes
ImageArray: TypeAlias = NDArray[np.float32]
# Shape: (grid_h, grid_w, kernel_h, kernel_w)
PSFGridArray: TypeAlias = NDArray[np.float32]
T_Array = TypeVar("T_Array", np.ndarray, torch.Tensor)


def spatially_varying_convolution_tensor(
    image_tensor: torch.Tensor,
    psf_grid_tensor: torch.Tensor,
    mode: Literal['cross-correlation', 'convolution'] = 'cross-correlation'
) -> torch.Tensor:
    """
    Performs spatially varying convolution on PyTorch tensors.

    The operation is performed independently on each patch, meaning there is no
    pixel sharing across patch boundaries. The padding is functionally similar
    to `mode='same'` for odd-sized kernels in libraries like SciPy.

    Args:
        image_tensor: A 2D PyTorch tensor representing the input image.
                      Its dimensions must be divisible by the grid dimensions.
        psf_grid_tensor: A 4D PyTorch tensor with shape (grid_h, grid_w, kH, kW).
                         The grid dimensions are inferred from this shape.
        mode: The operation to perform. 'cross-correlation' (default) is faster.
              'convolution' performs a true mathematical convolution.

    Returns:
        The processed 2D image as a PyTorch tensor on the same device as the input.
    """
    if image_tensor.ndim != 2:
        raise ValueError(f"Input tensor must be 2D, but got shape {image_tensor.shape}")
    if psf_grid_tensor.ndim != 4:
        raise ValueError(f"PSF grid tensor must be 4D, but got shape {psf_grid_tensor.shape}")

    H, W = image_tensor.shape
    grid_h, grid_w, kH, kW = psf_grid_tensor.shape
    patch_H, patch_W = H // grid_h, W // grid_w
    num_groups = grid_h * grid_w

    if kH > patch_H or kW > patch_W:
        raise ValueError(f"PSF kernel dimensions ({kH}, {kW}) cannot be larger than patch dimensions ({patch_H}, {patch_W}).")
    if kH % 2 == 0 or kW % 2 == 0:
        raise ValueError(f"PSF kernel dimensions must be odd, but got ({kH}, {kW}).")
    if H % grid_h != 0 or W % grid_w != 0:
        raise ValueError(f"Image dimensions ({H}, {W}) must be divisible by grid dimensions ({grid_h}, {grid_w}).")

    image_tensor_4d = image_tensor.view(1, 1, H, W)

    # 1. Reshape image into a batch of patches ("space-to-depth")
    # (1, 1, H, W) -> (1, 1, grid_h, patch_H, grid_w, patch_W)
    image_with_grid_dims = image_tensor_4d.view(1, 1, grid_h, patch_H, grid_w, patch_W)
    # -> (1, grid_h, grid_w, 1, patch_H, patch_W) to group spatial grid dimensions
    image_permuted = image_with_grid_dims.permute(0, 2, 4, 1, 3, 5)
    # -> (1, num_groups, patch_H, patch_W) to map patches to the channel dimension
    image_patches = image_permuted.contiguous().view(1, num_groups, patch_H, patch_W)

    # 2. Prepare kernels for grouped operation
    # (grid_h, grid_w, kH, kW) -> (num_groups, 1, kH, kW)
    prepared_kernels = psf_grid_tensor.view(num_groups, 1, kH, kW)
    if mode == 'convolution':
        prepared_kernels = torch.flip(prepared_kernels, dims=[-2, -1])

    # 3. Perform grouped 2D operation
    padding = (kH // 2, kW // 2)
    processed_patches = F.conv2d(image_patches, prepared_kernels, padding=padding, groups=num_groups)

    # 4. Reshape processed patches back into a single image ("depth-to-space")
    # (1, num_groups, patch_H, patch_W) -> (1, 1, H, W)
    patches_with_grid_dims = processed_patches.view(1, grid_h, grid_w, 1, patch_H, patch_W)
    patches_permuted_back = patches_with_grid_dims.permute(0, 3, 1, 4, 2, 5)
    output_tensor_4d = patches_permuted_back.contiguous().view(1, 1, H, W)

    return output_tensor_4d.view(H, W)


def spatially_varying_convolution(
    image: ImageArray,
    psf_grid: PSFGridArray,
    mode: Literal['cross-correlation', 'convolution'] = 'cross-correlation',
    device: str | torch.device | None = None
) -> ImageArray:
    """
    Convolves or cross-correlates a 2D NumPy array with a spatially varying PSF.

    This is a NumPy wrapper around the core tensor-based convolution function.
    It handles data conversion and device management.

    Args:
        image: A 2D NumPy array representing the input image. Inputs will be
               cast to float32. Its dimensions must be divisible by the grid
               dimensions.
        psf_grid: A 4D NumPy array with shape (grid_h, grid_w, kH, kW).
                  The grid dimensions are inferred from this shape.
        mode: The operation to perform. 'cross-correlation' (default) is faster.
              'convolution' performs a true mathematical convolution.
        device: The PyTorch device for computation (e.g., 'cuda', 'cpu').
                If None, it will be auto-detected.

    Returns:
        The processed 2D image as a NumPy array of dtype float32.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    final_device = torch.device(device)

    # Use torch.as_tensor to avoid unnecessary copies of NumPy data
    image_tensor = torch.as_tensor(image, device=final_device).to(dtype=torch.float32)
    psf_grid_tensor = torch.as_tensor(psf_grid, device=final_device).to(dtype=torch.float32)

    # Delegate all logic and validation to the core tensor function
    output_tensor = spatially_varying_convolution_tensor(image_tensor, psf_grid_tensor, mode)

    return output_tensor.cpu().numpy()


class TestSpatiallyVaryingConvolution(unittest.TestCase):
    """Unit tests for the spatially varying convolution functions."""

    def test_numpy_wrapper_validation(self):
        """Tests that the NumPy wrapper raises errors for invalid inputs."""
        image_np = np.zeros((64, 64), dtype=np.float32)
        psf_grid_np = np.zeros((8, 8, 5, 5), dtype=np.float32)
        
        # Test validation that happens before tensor conversion
        with self.assertRaisesRegex(ValueError, "must be a 2D array"):
            spatially_varying_convolution(image_np.flatten(), psf_grid_np)
        with self.assertRaisesRegex(ValueError, "must be a 4D array"):
            spatially_varying_convolution(image_np, psf_grid_np[0])

    def test_tensor_core_validation(self):
        """Tests that the core tensor function raises errors for invalid inputs."""
        image_tensor = torch.zeros(64, 64)
        psf_grid_tensor = torch.zeros(8, 8, 5, 5)

        with self.assertRaisesRegex(ValueError, "must be odd"):
            spatially_varying_convolution_tensor(image_tensor, torch.zeros(8, 8, 4, 4))
        with self.assertRaisesRegex(ValueError, "must be divisible"):
            spatially_varying_convolution_tensor(torch.zeros(63, 64), psf_grid_tensor)
        with self.assertRaisesRegex(ValueError, "cannot be larger than"):
            spatially_varying_convolution_tensor(image_tensor, torch.zeros(8, 8, 9, 9))

    def test_tensor_io(self):
        """Tests the tensor-in, tensor-out pathway, including device persistence."""
        image_tensor = torch.randn(64, 64)
        psf_grid_tensor = torch.randn(8, 8, 5, 5)

        output_cpu = spatially_varying_convolution_tensor(image_tensor.to("cpu"), psf_grid_tensor.to("cpu"))
        self.assertIsInstance(output_cpu, torch.Tensor)
        self.assertEqual(output_cpu.device.type, 'cpu')
        
        if torch.cuda.is_available():
            output_cuda = spatially_varying_convolution_tensor(image_tensor.to("cuda"), psf_grid_tensor.to("cuda"))
            self.assertIsInstance(output_cuda, torch.Tensor)
            self.assertEqual(output_cuda.device.type, 'cuda')

    def test_correctness_against_naive(self):
        """Verify PyTorch output matches a naive SciPy loop for both modes."""
        try:
            from scipy.signal import convolve2d, correlate2d
        except ImportError:
            self.skipTest("SciPy not installed, skipping comparison test.")

        image = np.random.rand(64, 64).astype(np.float32)
        psf_grid = np.random.rand(8, 8, 5, 5).astype(np.float32)

        expected_corr = _naive_spatially_varying_op(image, psf_grid, convolve=False)
        actual_corr = spatially_varying_convolution(image, psf_grid, mode='cross-correlation')
        np.testing.assert_allclose(actual_corr, expected_corr, atol=1e-5)

        expected_conv = _naive_spatially_varying_op(image, psf_grid, convolve=True)
        actual_conv = spatially_varying_convolution(image, psf_grid, mode='convolution')
        np.testing.assert_allclose(actual_conv, expected_conv, atol=1e-5)


def _create_demo_data(img_size: int, grid_dim: int, psf_size: int) -> tuple[ImageArray, PSFGridArray, dict]:
    """Generates a test image and a varied PSF grid."""
    image = np.zeros((img_size, img_size), dtype=np.float32)
    image[img_size//4:3*img_size//4, img_size//2] = 1.0
    image[img_size//2, img_size//4:3*img_size//4] = 1.0
    box_start, box_end = img_size//2 - img_size//8, img_size//2 + img_size//8
    image[box_start:box_end, box_start:box_end] = 0.5

    patch_coords_map = {
        "blur": (4, 3), "sobel": (2, 4), "motion": (4, 4), "identity": (3, 3)
    }

    psf_grid = np.zeros((grid_dim, grid_dim, psf_size, psf_size), dtype=np.float32)
    psf_center = psf_size // 2
    
    identity_psf = np.zeros((psf_size, psf_size), dtype=np.float32)
    identity_psf[psf_center, psf_center] = 1.0
    
    blur_psf = np.ones((psf_size, psf_size), dtype=np.float32) / (psf_size ** 2)
    
    sobel_v = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    vedge_psf = np.zeros((psf_size, psf_size), dtype=np.float32)
    vedge_psf[psf_center-1:psf_center+2, psf_center-1:psf_center+2] = sobel_v
    
    hmotion_psf = np.zeros((psf_size, psf_size), dtype=np.float32)
    hmotion_psf[psf_center, :] = 1.0 / psf_size
    
    psf_map = {"blur": blur_psf, "sobel": vedge_psf, "motion": hmotion_psf, "identity": identity_psf}

    psf_grid[:, :] = identity_psf
    for name, coords in patch_coords_map.items():
        psf_grid[coords] = psf_map[name]
    return image, psf_grid, patch_coords_map

def _naive_spatially_varying_op(image: ImageArray, psf_grid: PSFGridArray, convolve: bool) -> ImageArray:
    """A naive, slow implementation using a loop over SciPy for baseline comparison."""
    from scipy.signal import convolve2d, correlate2d
    H, W = image.shape
    grid_h, grid_w, _, _ = psf_grid.shape
    patch_H, patch_W = H // grid_h, W // grid_w
    output_image = np.zeros_like(image)
    op = convolve2d if convolve else correlate2d
    for r in range(grid_h):
        for c in range(grid_w):
            sy, ey = r * patch_H, (r + 1) * patch_H
            sx, ex = c * patch_W, (c + 1) * patch_W
            image_patch = image[sy:ey, sx:ex]
            psf = psf_grid[r, c]
            processed_patch = op(image_patch, psf, mode='same', boundary='fill', fillvalue=0)
            output_image[sy:ey, sx:ex] = processed_patch
    return output_image


def main():
    """Runs a demonstration and benchmark of the convolution function."""
    IMAGE_SIZE, PSF_SIZE, GRID_DIM = 1024, 11, 8
    print("--- Demonstrative Example & Benchmark ---")

    example_image, example_psf_grid, patch_coords_map = _create_demo_data(IMAGE_SIZE, GRID_DIM, PSF_SIZE)
    patch_size = IMAGE_SIZE // GRID_DIM

    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nPerforming operations on device: '{device_name}'...")

    try:
        # Warm-up run (especially important for GPU)
        _ = spatially_varying_convolution(example_image, example_psf_grid)
        num_runs = 10
        corr_time = timeit.timeit(
            lambda: spatially_varying_convolution(example_image, example_psf_grid, mode='cross-correlation'),
            number=num_runs
        ) / num_runs
        print(f"\n  PyTorch (Cross-Correlation) Time: {corr_time:.6f} seconds (avg over {num_runs} runs)")

        naive_time = timeit.timeit(
            lambda: _naive_spatially_varying_op(example_image, example_psf_grid, convolve=False),
            number=1
        )
        print(f"  Naive (SciPy correlate2d) Time: {naive_time:.6f} seconds")
        if corr_time > 0: print(f"  Speedup: {naive_time / corr_time:.2f}x")
        
        output_image = spatially_varying_convolution(example_image, example_psf_grid)
        np.testing.assert_allclose(output_image, _naive_spatially_varying_op(example_image, example_psf_grid, convolve=False), atol=1e-5)
        print("  Results from both methods are numerically consistent.")

    except ImportError:
        print("\nSciPy not found. Skipping benchmark comparison. Install with: pip install scipy")
        output_image = spatially_varying_convolution(example_image, example_psf_grid)

    print("\n--- Per-Patch Verification (Mean and Std Dev) ---")
    for name, coords in sorted(patch_coords_map.items(), key=lambda item: item[1]):
        py, px = coords
        input_patch = example_image[py*patch_size:(py+1)*patch_size, px*patch_size:(px+1)*patch_size]
        output_patch = output_image[py*patch_size:(py+1)*patch_size, px*patch_size:(px+1)*patch_size]
        print(f"  Patch {coords} ('{name}'):")
        print(f"    Input  -> Mean: {input_patch.mean():.4f}, Std: {input_patch.std():.4f}")
        print(f"    Output -> Mean: {output_patch.mean():.4f}, Std: {output_patch.std():.4f}")

    try:
        import matplotlib.pyplot as plt
        vmin, vmax = np.percentile(output_image, [1, 99])
        fig, axes = plt.subplots(1, 2, figsize=(12, 6), tight_layout=True)
        axes[0].imshow(example_image, cmap='gray'); axes[0].set_title('Input Image'); axes[0].axis('off')
        axes[1].imshow(output_image, cmap='gray', vmin=vmin, vmax=vmax); axes[1].set_title('Processed Image'); axes[1].axis('off')
        plt.savefig('convolution_result.png')
        print("\nSaved visualization to 'convolution_result.png'")
    except ImportError:
        print("\nMatplotlib not found. Skipping visualization. Install with: pip install matplotlib")

if __name__ == "__main__":
    main()
