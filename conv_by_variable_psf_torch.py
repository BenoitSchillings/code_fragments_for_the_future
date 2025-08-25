```python
import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal import convolve2d
import time
import sys
import unittest
from typing import Union, Optional, Tuple

def _create_test_data(
    image_dims: Tuple[int, int],
    grid_dims: Tuple[int, int],
    psf_dims: Tuple[int, int]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates a random image and a grid of normalized PSFs.

    Args:
        image_dims (Tuple[int, int]): Dimensions (H, W) of the output image.
        grid_dims (Tuple[int, int]): Grid dimensions (grid_h, grid_w) for the PSFs.
        psf_dims (Tuple[int, int]): Dimensions (psf_h, psf_w) of each individual PSF.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing:
            - image (np.ndarray): A 2D random float32 image.
            - psfs (np.ndarray): A 4D grid of random float32 PSFs, where each
                                PSF is normalized to sum to 1.
    """
    image = np.random.rand(*image_dims).astype(np.float32)
    psfs = np.random.rand(*grid_dims, *psf_dims).astype(np.float32)

    # Normalize each PSF to have a sum of 1 to conserve energy/brightness
    psf_sums = psfs.sum(axis=(2, 3), keepdims=True)
    # Add a small epsilon for numerical stability, preventing division by zero
    psfs /= (psf_sums + 1e-8)
    return image, psfs

def ground_truth_tiled_convolution(image: np.ndarray, psfs: np.ndarray) -> np.ndarray:
    """
    A simple, slow implementation of tiled convolution for verification.
    This function uses a loop and scipy.signal.convolve2d on non-overlapping patches.

    Args:
        image (np.ndarray): The 2D input image array.
        psfs (np.ndarray): A 4D array of PSFs with shape (grid_h, grid_w, psf_h, psf_w).

    Returns:
        np.ndarray: The convolved 2D image.
    """
    grid_h, grid_w = psfs.shape[:2]
    img_h, img_w = image.shape
    patch_h = img_h // grid_h
    patch_w = img_w // grid_w

    output_image = np.zeros_like(image, dtype=np.float32)
    for i in range(grid_h):
        for j in range(grid_w):
            r_start, c_start = i * patch_h, j * patch_w
            r_end, c_end = r_start + patch_h, c_start + patch_w

            patch = image[r_start:r_end, c_start:c_end]
            psf = psfs[i, j]

            convolved_patch = convolve2d(patch, psf, mode='same', boundary='fill', fillvalue=0)
            output_image[r_start:r_end, c_start:c_end] = convolved_patch
    return output_image


def spatially_varying_convolution_pytorch(
    image: Union[np.ndarray, torch.Tensor],
    psfs: Union[np.ndarray, torch.Tensor],
    device: Optional[str] = None
) -> torch.Tensor:
    """
    Performs a high-performance tiled convolution with a spatially varying PSF using PyTorch.

    Note:
        This function implements a *tiled* or *piecewise* convolution, NOT a true,
        mathematically continuous spatially varying convolution. It partitions the image
        into non-overlapping patches and convolves each with a different kernel.
        This approach is very fast but will produce noticeable grid artifacts at the
        patch boundaries because each convolution is independent and does not account
        for data in adjacent patches. A more physically accurate (and much more complex)
        method would require an overlap-add or overlap-save strategy.

    Args:
        image (Union[np.ndarray, torch.Tensor]): The 2D input image array. Its dimensions (H, W)
                                                 must be perfectly divisible by the PSF grid dimensions.
        psfs (Union[np.ndarray, torch.Tensor]): A 4D array of PSFs with shape (grid_h, grid_w, psf_h, psf_w).
                                                The PSF dimensions (psf_h, psf_w) must be odd. This ensures a
                                                clear center pixel, required for the padding calculation
                                                to achieve a 'same'-sized output.
        device (Optional[str], optional): The device name to perform computation on (e.g., 'cuda', 'cpu').
                                          If None, auto-detects GPU. Defaults to None.

    Returns:
        torch.Tensor: The 2D convolved image as a PyTorch tensor on the specified compute device.
    """
    compute_device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Use torch.as_tensor for efficient conversion, avoiding data copy if possible.
    image_tensor = torch.as_tensor(image, dtype=torch.float32, device=compute_device)
    psfs_tensor = torch.as_tensor(psfs, dtype=torch.float32, device=compute_device)

    if image_tensor.ndim != 2:
        raise ValueError("Input image must be a 2D array.")
    if psfs_tensor.ndim != 4:
        raise ValueError("PSFs array must be a 4D array.")

    grid_h, grid_w, psf_h, psf_w = psfs_tensor.shape
    img_h, img_w = image_tensor.shape

    if psf_h % 2 == 0 or psf_w % 2 == 0:
        raise ValueError("PSF dimensions must be odd for 'same' padding to work correctly.")

    if img_h % grid_h != 0 or img_w % grid_w != 0:
        raise ValueError(f"Image dimensions ({img_h}, {img_w}) must be divisible "
                         f"by grid dimensions ({grid_h}, {grid_w}).")

    patch_h = img_h // grid_h
    patch_w = img_w // grid_w
    num_patches = grid_h * grid_w

    # Reshape image and PSFs for batch processing
    image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)
    psfs_tensor = psfs_tensor.view(num_patches, 1, psf_h, psf_w)

    # Unfold the image into a batch of patches and reshape for grouped convolution.
    patches = image_tensor.unfold(2, patch_h, patch_h).unfold(3, patch_w, patch_w)
    patches = patches.contiguous().view(1, num_patches, patch_h, patch_w)

    padding_h = psf_h // 2
    padding_w = psf_w // 2

    convolved_patches = F.conv2d(
        patches,
        # PyTorch's `conv2d` performs cross-correlation. To match the mathematical
        # definition of convolution (as used by scipy), the kernel must be flipped.
        weight=torch.flip(psfs_tensor, dims=[-2, -1]),
        padding=(padding_h, padding_w),
        groups=num_patches
    )

    # Fold the convolved patches back into a single image.
    # output shape after view: (grid_h, grid_w, patch_h, patch_w)
    output = convolved_patches.view(grid_h, grid_w, patch_h, patch_w)
    # We need to reorder the dimensions so that the patch rows and columns are
    # contiguous in memory for the final reshape.
    # Permute to (grid_h, patch_h, grid_w, patch_w). This groups all patch rows
    # from all grid tiles together, preparing for the final reshape into (H, W).
    output = output.permute(0, 2, 1, 3).reshape(img_h, img_w)

    return output


class TestConvolution(unittest.TestCase):
    def _run_test_case(self, image_dims, grid_dims, psf_dims, atol=1e-5):
        """Helper to run a full test case for a given configuration."""
        image, psfs = _create_test_data(image_dims, grid_dims, psf_dims)

        expected = ground_truth_tiled_convolution(image, psfs)
        actual_tensor = spatially_varying_convolution_pytorch(image, psfs)
        actual = actual_tensor.cpu().numpy()

        self.assertTrue(
            np.allclose(expected, actual, atol=atol),
            f"Output mismatch for image {image_dims}, grid {grid_dims}, psf {psf_dims}"
        )

    def test_correctness_square_8x8_grid(self):
        """Verifies correctness for the prompt's 8x8 grid requirement."""
        self._run_test_case(image_dims=(128, 128), grid_dims=(8, 8), psf_dims=(7, 7))

    def test_correctness_rectangular_image(self):
        """Verifies correctness with a rectangular image."""
        self._run_test_case(image_dims=(128, 256), grid_dims=(4, 8), psf_dims=(5, 5))

    def test_correctness_non_square_grid(self):
        """Verifies correctness with a non-square grid."""
        self._run_test_case(image_dims=(256, 128), grid_dims=(8, 4), psf_dims=(9, 9))

    def test_correctness_non_square_psf(self):
        """Verifies correctness with non-square PSFs."""
        self._run_test_case(image_dims=(128, 128), grid_dims=(4, 4), psf_dims=(5, 9))
    
    def test_correctness_single_tile(self):
        """Verifies correctness for a 1x1 grid (equivalent to a standard convolution)."""
        self._run_test_case(image_dims=(64, 64), grid_dims=(1, 1), psf_dims=(7, 7))

    def test_correctness_minimal_psf(self):
        """Verifies correctness with a minimal 1x1 PSF."""
        self._run_test_case(image_dims=(128, 128), grid_dims=(8, 8), psf_dims=(1, 1))

    def test_error_with_even_psfs(self):
        """Verifies that even-sized PSFs correctly raise a ValueError."""
        image, psfs = _create_test_data(
            image_dims=(128, 128), grid_dims=(8, 8), psf_dims=(6, 6)
        )
        with self.assertRaisesRegex(ValueError, "PSF dimensions must be odd"):
            spatially_varying_convolution_pytorch(image, psfs)


def run_performance_demo():
    """
    Runs a large-scale example to benchmark performance and verify correctness.
    """
    print("\n--- Running Performance Example and Verification ---")

    img_size = (1024, 1024)
    psf_size = (17, 17)
    grid_size = (8, 8)

    print(f"Image size: {img_size}")
    print(f"PSF grid size: {grid_size}")
    print(f"Individual PSF size: {psf_size}")

    image, psfs = _create_test_data(img_size, grid_size, psf_size)

    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device_name}")

    try:
        # Warm-up run for accurate timing
        _ = spatially_varying_convolution_pytorch(image, psfs, device=device_name)

        start_time = time.time()
        convolved_tensor = spatially_varying_convolution_pytorch(image, psfs, device=device_name)
        if convolved_tensor.is_cuda:
            torch.cuda.synchronize()
        end_time = time.time()

        execution_time = end_time - start_time
        convolved_image = convolved_tensor.cpu().numpy()

        print(f"\nExecution time: {execution_time:.6f} seconds")

        print("\nVerifying large-scale result against ground truth...")
        print("(This may take a moment)")
        gt_start_time = time.time()
        expected_large_output = ground_truth_tiled_convolution(image, psfs)
        gt_end_time = time.time()
        print(f"Ground truth computation time: {gt_end_time - gt_start_time:.4f} seconds")

        if np.allclose(expected_large_output, convolved_image, atol=1e-4):
            print("Verification PASSED: High-performance output matches ground truth.")
        else:
            error = np.abs(expected_large_output - convolved_image).max()
            print(f"Verification FAILED. Maximum absolute error: {error}")

    except (ValueError, RuntimeError) as e:
        print(f"\nPerformance example FAILED with an error: {e}", file=sys.stderr)


if __name__ == "__main__":
    print("--- Running Unit Tests ---")
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestConvolution)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    if result.wasSuccessful():
        run_performance_demo()
    else:
        print("\nUnit tests failed. Skipping performance demonstration.", file=sys.stderr)
        sys.exit(1)
