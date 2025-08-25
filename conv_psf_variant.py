```python
import numpy as np
from scipy.signal import fftconvolve
import unittest
from numpy.typing import NDArray, ArrayLike
import concurrent.futures
from typing import TypeAlias, Generator, Optional


# --- Type Aliases for Clarity ---
FloatArray: TypeAlias = NDArray[np.float32]


def _yield_convolution_tasks(image: FloatArray, psf_grid: FloatArray, pad_mode: str) -> Generator[tuple[tuple[FloatArray, FloatArray], tuple[slice, slice]], None, None]:
    """
    Private generator to prepare tasks and output locations for convolution.
    Handles non-divisible image dimensions gracefully.
    """
    img_h, img_w = image.shape
    grid_rows, grid_cols, psf_h, psf_w = psf_grid.shape

    sub_h = img_h // grid_rows
    sub_w = img_w // grid_cols

    pad_h = psf_h // 2
    pad_w = psf_w // 2

    padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode=pad_mode)

    for i in range(grid_rows):
        for j in range(grid_cols):
            # Handle remainder for last row/column in non-divisible cases
            start_row = i * sub_h
            end_row = (i + 1) * sub_h if i < grid_rows - 1 else img_h
            start_col = j * sub_w
            end_col = (j + 1) * sub_w if j < grid_cols - 1 else img_w

            # Slice the padded image. The slice dimensions are the tile size
            # plus the padding on both sides.
            padded_tile = padded_image[start_row : end_row + 2 * pad_h, start_col : end_col + 2 * pad_w]

            task = (padded_tile, psf_grid[i, j])
            output_slice = (slice(start_row, end_row), slice(start_col, end_col))
            yield task, output_slice


def _convolve_padded_tile(task_data: tuple[FloatArray, FloatArray]) -> FloatArray:
    """
    Helper function to convolve a single padded image tile with its PSF.
    Designed for use with a parallel executor.
    """
    padded_tile, psf = task_data
    return fftconvolve(padded_tile, psf, mode='valid')


def spatially_varying_convolve(
    image: ArrayLike,
    psf_grid: ArrayLike,
    pad_mode: str = 'reflect',
    parallel: bool = True,
    max_workers: int | None = None
) -> FloatArray:
    """
    Approximates a spatially varying convolution using a parallelized approach.

    This function implements a block-wise or piecewise-constant approximation.
    It partitions the input array into a grid of sub-sections (matching the grid
    dimensions of `psf_grid`) and convolves each sub-section with a single,
    corresponding PSF. An overlap-save strategy is used by padding each
    sub-section before convolution to ensure seamless stitching. This process
    can be run in parallel using a thread pool, which is efficient as `fftconvolve`
    releases Python's Global Interpreter Lock (GIL).

    Note on Memory: This implementation pre-pads the entire image, which is fast
    for moderately sized inputs but can consume significant memory for very large
    images. A more memory-efficient alternative would pad each tile individually.

    Args:
        image (ArrayLike): The 2D input array (image). It will be cast to float32.
        psf_grid (ArrayLike): A 4D array of shape (grid_rows, grid_cols,
                               psf_h, psf_w) containing the unique
                               PSFs. All PSFs within the grid must have the same,
                               odd dimensions (psf_h, psf_w) and be normalized
                               to sum to 1. It will be cast to float32.
        pad_mode (str, optional): The padding mode used by `np.pad`. Defaults
                                  to 'reflect', which provides a reasonable
                                  flux-conserving boundary condition but is not
                                  perfect. 'constant' will lose flux at edges.
        parallel (bool, optional): Whether to execute the convolutions in parallel.
                                   Defaults to True.
        max_workers (int | None, optional): The maximum number of threads to use
                                            for parallel execution. Defaults to
                                            None, which lets the executor decide.

    Returns:
        FloatArray: The 2D array resulting from the convolution, with the
                    same shape as the input image.
    """
    # --- Input Validation and Type Enforcement ---
    image = np.asarray(image, dtype=np.float32)
    psf_grid = np.asarray(psf_grid, dtype=np.float32)

    if image.ndim != 2:
        raise ValueError(f"Input 'image' must be a 2D array, but got {image.ndim} dimensions.")
    if psf_grid.ndim != 4:
        raise ValueError(f"Input 'psf_grid' must be a 4D array, but got {psf_grid.ndim} dimensions.")

    grid_rows, grid_cols, psf_h, psf_w = psf_grid.shape
    if psf_h % 2 == 0 or psf_w % 2 == 0:
        raise ValueError("PSF dimensions must be odd.")

    psf_sums = np.sum(psf_grid, axis=(2, 3))
    if not np.allclose(psf_sums, np.ones((grid_rows, grid_cols))):
        raise ValueError("All PSFs in the grid must be normalized to sum to 1.")

    # --- Task Preparation ---
    output_image = np.empty_like(image)
    tasks_with_slices = list(_yield_convolution_tasks(image, psf_grid, pad_mode))

    if not tasks_with_slices:
        return output_image

    tasks, output_slices = zip(*tasks_with_slices)

    # --- Execution ---
    if parallel:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = executor.map(_convolve_padded_tile, tasks)
    else:
        results = map(_convolve_padded_tile, tasks)

    for out_slice, convolved_tile in zip(output_slices, results):
        output_image[out_slice] = convolved_tile

    return output_image


class TestSpatiallyVaryingConvolve(unittest.TestCase):
    def setUp(self):
        """Set up deterministic test data for verification."""
        self.rng = np.random.default_rng(seed=42)
        # Define constants for test configuration
        self.GRID_DIMS = (4, 4)
        self.PSF_SIZE = 15
        self.SUB_SECTION_DIMS = (32, 32)
        self.psf_grid = create_gaussian_psf_grid(self.GRID_DIMS, self.PSF_SIZE)
        self.image_dims = (self.GRID_DIMS[0] * self.SUB_SECTION_DIMS[0], self.GRID_DIMS[1] * self.SUB_SECTION_DIMS[1])

    def _run_centered_deltas_test(self, parallel: bool):
        """Helper to test convolution with deltas at the center of each sub-section."""
        # Create ONE image with a delta in the center of EACH tile
        image = np.zeros(self.image_dims, dtype=np.float32)
        sub_h, sub_w = self.SUB_SECTION_DIMS
        grid_h, grid_w = self.GRID_DIMS

        for i in range(grid_h):
            for j in range(grid_w):
                center_y = i * sub_h + sub_h // 2
                center_x = j * sub_w + sub_w // 2
                image[center_y, center_x] = 1.0

        # Convolve the entire image at once
        convolved = spatially_varying_convolve(image, self.psf_grid, parallel=parallel)

        # Now, verify each PSF in the single output image
        psf_half = self.PSF_SIZE // 2
        for i in range(grid_h):
            for j in range(grid_w):
                center_y = i * sub_h + sub_h // 2
                center_x = j * sub_w + sub_w // 2

                psf_slice_y = slice(center_y - psf_half, center_y + psf_half + 1)
                psf_slice_x = slice(center_x - psf_half, center_x + psf_half + 1)

                output_psf = convolved[psf_slice_y, psf_slice_x]
                expected_psf = self.psf_grid[i, j]
                np.testing.assert_allclose(
                    output_psf, expected_psf, rtol=1e-5, atol=1e-5,
                    err_msg=f"PSF mismatch at grid index ({i}, {j}) with parallel={parallel}"
                )

    def test_centered_deltas_serial(self):
        self._run_centered_deltas_test(parallel=False)

    def test_centered_deltas_parallel(self):
        self._run_centered_deltas_test(parallel=True)

    def _run_seam_test(self, parallel: bool):
        """Helper to verify seamless stitching with a delta on a boundary."""
        image = np.zeros(self.image_dims, dtype=np.float32)
        seam_y = self.SUB_SECTION_DIMS[0]
        seam_x = self.SUB_SECTION_DIMS[1]
        image[seam_y, seam_x] = 1.0  # Delta exactly on a 4-corner seam

        psf = np.zeros((self.PSF_SIZE, self.PSF_SIZE), dtype=np.float32)
        psf[self.PSF_SIZE // 2, self.PSF_SIZE // 2] = 1.0  # PSF is a delta
        constant_psf_grid = np.tile(psf, (*self.GRID_DIMS, 1, 1))

        convolved = spatially_varying_convolve(image, constant_psf_grid, parallel=parallel)
        np.testing.assert_allclose(image, convolved, atol=1e-6)

    def test_seam_continuity(self):
        self._run_seam_test(parallel=False)
        self._run_seam_test(parallel=True)

    def test_non_divisible_dimensions(self):
        """Test graceful handling of image dimensions not divisible by the grid."""
        image = np.zeros((130, 130), dtype=np.float32)
        image[65, 65] = 1.0
        convolved = spatially_varying_convolve(image, self.psf_grid, parallel=False)
        self.assertEqual(image.shape, convolved.shape)
        np.testing.assert_allclose(np.sum(image), np.sum(convolved), rtol=1e-5)

    def test_parallel_vs_serial_equivalence(self):
        """Ensures the parallel and serial execution paths produce identical results."""
        image = self.rng.random(self.image_dims, dtype=np.float32)
        result_parallel = spatially_varying_convolve(image, self.psf_grid, parallel=True)
        result_serial = spatially_varying_convolve(image, self.psf_grid, parallel=False)
        np.testing.assert_allclose(result_parallel, result_serial, rtol=1e-5, atol=1e-5)

    def test_raises_for_even_psf(self):
        psf_grid_even = np.ones((4, 4, 10, 10), dtype=np.float32)
        image = np.zeros((128, 128))
        with self.assertRaisesRegex(ValueError, "PSF dimensions must be odd"):
            spatially_varying_convolve(image, psf_grid_even)

    def test_raises_for_non_normalized_psf(self):
        psf_grid_unnormalized = np.ones((4, 4, 15, 15), dtype=np.float32)
        image = np.zeros((128, 128))
        with self.assertRaisesRegex(ValueError, "All PSFs in the grid must be normalized"):
            spatially_varying_convolve(image, psf_grid_unnormalized)


def create_star_image(image_size: tuple[int, int], num_stars: int = 250, seed: int = 123) -> FloatArray:
    """Generates a sample image of a star field."""
    rng = np.random.default_rng(seed=seed)
    image = np.zeros(image_size, dtype=np.float32)
    y_coords = rng.integers(0, image_size[0], num_stars)
    x_coords = rng.integers(0, image_size[1], num_stars)
    image[y_coords, x_coords] = rng.uniform(0.5, 1.0, num_stars)
    return image


def create_gaussian_psf_grid(grid_dims: tuple[int, int] = (8, 8), psf_size: int = 31) -> FloatArray:
    """Generates a grid of spatially varying Gaussian PSFs."""
    psf_center = psf_size // 2
    y, x = np.mgrid[-psf_center:psf_center + 1, -psf_center:psf_center + 1]
    grid_y, grid_x = np.mgrid[0:grid_dims[0], 0:grid_dims[1]]
    center_y, center_x = (grid_dims[0] - 1) / 2.0, (grid_dims[1] - 1) / 2.0
    max_distance = np.sqrt(center_y**2 + center_x**2)
    if max_distance == 0:
        max_distance = 1.0
    distance_from_center = np.sqrt((grid_y - center_y)**2 + (grid_x - center_x)**2)
    sigma = 1.0 + 3.5 * (distance_from_center / max_distance)
    sigma_grid = sigma.reshape(*grid_dims, 1, 1)
    psf_grid = np.exp(-((x**2 + y**2) / (2.0 * sigma_grid**2))).astype(np.float32)
    psf_grid /= np.sum(psf_grid, axis=(2, 3), keepdims=True)
    return psf_grid


def run_performance_demo():
    """Runs the performance demonstration and visualization."""
    import time
    import matplotlib.pyplot as plt

    # --- Demo 1: Large Square Image ---
    print("\n--- DEMO 1: Large Square Image (8x8 Grid) ---")
    image_size_1 = (4096, 4096)
    grid_dims_1 = (8, 8)
    psf_size_1 = 31
    pad_mode_demo = 'reflect'

    sample_image_1 = create_star_image(image_size=image_size_1)
    sample_psf_grid_1 = create_gaussian_psf_grid(grid_dims=grid_dims_1, psf_size=psf_size_1)

    print(f"Input image shape: {sample_image_1.shape}")
    print(f"PSF grid shape: {sample_psf_grid_1.shape}")
    print(f"Using padding mode: '{pad_mode_demo}'")

    print("\nTiming single-threaded (serial) implementation...")
    start_time_serial = time.time()
    _ = spatially_varying_convolve(sample_image_1, sample_psf_grid_1, parallel=False)
    time_serial = time.time() - start_time_serial
    print(f"Serial implementation completed in: {time_serial:.4f} seconds")

    print("\nTiming high-performance parallel implementation...")
    start_time_parallel = time.time()
    parallel_convolved_1 = spatially_varying_convolve(sample_image_1, sample_psf_grid_1, parallel=True)
    time_parallel = time.time() - start_time_parallel
    print(f"Parallel implementation completed in: {time_parallel:.4f} seconds")

    if time_parallel > 0 and time_serial > time_parallel:
        speedup = time_serial / time_parallel
        print(f"\nPerformance Result: Parallel implementation is {speedup:.2f}x faster.")
    else:
        print("\nPerformance note: Parallel implementation not faster.")

    print("\nChecking flux conservation...")
    input_flux = np.sum(sample_image_1)
    output_flux = np.sum(parallel_convolved_1)
    print(f"Input flux: {input_flux:.4f}, Output flux: {output_flux:.4f}")
    np.testing.assert_allclose(input_flux, output_flux, rtol=5e-4)
    print("Flux conservation test PASSED (within tolerance for reflect padding).")

    # --- Demo 2: Non-Square, Non-Divisible Image ---
    print("\n--- DEMO 2: Non-Square, Non-Divisible Image (4x3 Grid) ---")
    image_size_2 = (1025, 770)
    grid_dims_2 = (4, 3)
    psf_size_2 = 25
    
    sample_image_2 = create_star_image(image_size=image_size_2)
    sample_psf_grid_2 = create_gaussian_psf_grid(grid_dims=grid_dims_2, psf_size=psf_size_2)
    
    print(f"Input image shape: {sample_image_2.shape}")
    print(f"PSF grid shape: {sample_psf_grid_2.shape}")
    
    start_time = time.time()
    _ = spatially_varying_convolve(sample_image_2, sample_psf_grid_2)
    print(f"Non-divisible case completed successfully in: {time.time() - start_time:.4f} seconds.")

    # --- Visualization ---
    print(f"\n--- Visualizing Results (from Demo 1) ---")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(sample_image_1[:256, :256], cmap='gray')
    axes[0].set_title("Original Image (Zoomed In)")
    axes[0].axis('off')
    axes[1].imshow(sample_psf_grid_1[grid_dims_1[0] // 2, grid_dims_1[1] // 2], cmap='hot')
    axes[1].set_title("Example PSF (from grid center)")
    axes[1].axis('off')
    axes[2].imshow(parallel_convolved_1[:256, :256], cmap='gray')
    axes[2].set_title("Convolved Image (Zoomed In)")
    axes[2].axis('off')
    plt.tight_layout()
    output_filename = 'convolution_results.png'
    plt.savefig(output_filename)
    print(f"Saved visualization to {output_filename}")


def run_tests() -> bool:
    """Runs the unit test suite."""
    suite = unittest.TestLoader().loadTestsFromTestCase(TestSpatiallyVaryingConvolve)
    runner = unittest.TextTestRunner()
    result = runner.run(suite)
    if result.wasSuccessful():
        print("\nAll unit tests passed.")
        return True
    print("\nUnit tests failed.")
    return False


if __name__ == "__main__":
    print("="*60)
    print("STEP 1: RUNNING UNIT TESTS")
    print("="*60)
    tests_passed = run_tests()

    if tests_passed:
        print("\n" + "="*60)
        print("STEP 2: RUNNING PERFORMANCE & VISUALIZATION DEMO")
        print("="*60)
        run_performance_demo()
    else:
        print("\nSkipping demo due to unit test failures.")
```
