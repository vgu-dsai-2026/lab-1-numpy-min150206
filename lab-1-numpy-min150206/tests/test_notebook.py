from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

# -----------------------------------------------------------------------------
# CONSTANTS
# -----------------------------------------------------------------------------

NOTEBOOK_PATH = Path(__file__).resolve().parents[1] / "notebook.py"
SPEC = importlib.util.spec_from_file_location("student_notebook", NOTEBOOK_PATH)
NOTEBOOK_MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(NOTEBOOK_MODULE)

load_image_np = NOTEBOOK_MODULE.load_image_np
center_crop = NOTEBOOK_MODULE.center_crop
flip_horizontal = NOTEBOOK_MODULE.flip_horizontal
normalize_01 = NOTEBOOK_MODULE.normalize_01
rgb_to_gray = NOTEBOOK_MODULE.rgb_to_gray
channel_summary = NOTEBOOK_MODULE.channel_summary
convolve2d_matmul = NOTEBOOK_MODULE.convolve2d_matmul
flatten_image = NOTEBOOK_MODULE.flatten_image
extract_features = NOTEBOOK_MODULE.extract_features
build_feature_matrix = NOTEBOOK_MODULE.build_feature_matrix

TEST_DATA_DIR = Path(__file__).resolve().parents[1] / "data"

GAUSSIAN_KERNEL = np.array(
    [
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1],
    ],
    dtype=np.float32,
) / 16.0

SOBEL_Y_KERNEL = np.array(
    [
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1],
    ],
    dtype=np.float32,
)

# -----------------------------------------------------------------------------
# Question 1: load_image_np
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("image_name", "expected_shape", "expected_min", "expected_mean", "expected_max"),
    [
        pytest.param(
            "cat/cat_0005.jpg",
            (64, 64, 3),
            0,
            115,
            206,
            id="cat-image-stats",
        ),
        pytest.param(
            "dog/dog_0005.jpg",
            (64, 64, 3),
            0,
            89,
            255,
            id="dog-image-stats",
        ),
    ],
)
def test_load_image_np_real_images(
    image_name: str,
    expected_shape: tuple,
    expected_min: int,
    expected_mean: int,
    expected_max: int,
) -> None:
    image_path = TEST_DATA_DIR / image_name

    # Ensure file exists (important for CI)
    assert image_path.exists(), f"Missing test image: {image_path}"

    result = load_image_np(image_path)

    # Type & structure checks
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.uint8
    assert result.shape == expected_shape

    # Value checks
    assert result.min() == expected_min
    assert int(result.mean()) == expected_mean
    assert result.max() == expected_max


# -----------------------------------------------------------------------------
# Question 2: center_crop
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("image", "crop_size", "expected"),
    [
        pytest.param(
            np.arange(6 * 8 * 3, dtype=np.uint8).reshape(6, 8, 3),
            4,
            np.arange(6 * 8 * 3, dtype=np.uint8).reshape(6, 8, 3)[1:5, 2:6],
            id="even-dimensions",
        ),
        pytest.param(
            np.arange(7 * 9 * 3, dtype=np.uint8).reshape(7, 9, 3),
            3,
            np.arange(7 * 9 * 3, dtype=np.uint8).reshape(7, 9, 3)[2:5, 3:6],
            id="odd-dimensions",
        ),
    ],
)
def test_center_crop(image: np.ndarray, crop_size: int, expected: np.ndarray) -> None:
    result = center_crop(image, crop_size=crop_size)

    assert result.shape == (crop_size, crop_size, 3)
    np.testing.assert_array_equal(result, expected)


# -----------------------------------------------------------------------------
# Question 3: flip_horizontal
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("image", "expected"),
    [
        pytest.param(
            np.array(
                [
                    [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                    [[10, 11, 12], [13, 14, 15], [16, 17, 18]],
                ],
                dtype=np.uint8,
            ),
            np.array(
                [
                    [[7, 8, 9], [4, 5, 6], [1, 2, 3]],
                    [[16, 17, 18], [13, 14, 15], [10, 11, 12]],
                ],
                dtype=np.uint8,
            ),
            id="three-column-image",
        ),
        pytest.param(
            np.array(
                [
                    [[9, 8, 7], [6, 5, 4]],
                    [[3, 2, 1], [0, 1, 2]],
                    [[4, 5, 6], [7, 8, 9]],
                ],
                dtype=np.uint8,
            ),
            np.array(
                [
                    [[6, 5, 4], [9, 8, 7]],
                    [[0, 1, 2], [3, 2, 1]],
                    [[7, 8, 9], [4, 5, 6]],
                ],
                dtype=np.uint8,
            ),
            id="two-column-image",
        ),
    ],
)
def test_flip_horizontal(image: np.ndarray, expected: np.ndarray) -> None:
    result = flip_horizontal(image)

    np.testing.assert_array_equal(result, expected)


# -----------------------------------------------------------------------------
# Question 4: normalize_01
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("image", "expected"),
    [
        pytest.param(
            np.array([[[0, 128, 255]]], dtype=np.uint8),
            np.array([[[0.0, 128.0 / 255.0, 1.0]]], dtype=np.float32),
            id="basic-scaling",
        ),
        pytest.param(
            np.array(
                [
                    [[10, 20, 30], [40, 50, 60]],
                    [[70, 80, 90], [100, 110, 120]],
                ],
                dtype=np.uint8,
            ),
            np.array(
                [
                    [[10, 20, 30], [40, 50, 60]],
                    [[70, 80, 90], [100, 110, 120]],
                ],
                dtype=np.float32,
            )
            / 255.0,
            id="preserves-shape-and-range",
        ),
    ],
)
def test_normalize_01(image: np.ndarray, expected: np.ndarray) -> None:
    result = normalize_01(image)

    assert result.dtype == np.float32
    np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-6)
    assert result.min() >= 0.0
    assert result.max() <= 1.0


# -----------------------------------------------------------------------------
# Question 5: rgb_to_gray
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("image_float", "expected"),
    [
        pytest.param(
            np.array(
                [
                    [[0.25, 0.25, 0.25], [0.75, 0.75, 0.75]],
                    [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
                ],
                dtype=np.float32,
            ),
            np.array(
                [[0.25, 0.75], [0.0, 1.0]],
                dtype=np.float32,
            ),
            id="preserves-neutral-gray",
        ),
        pytest.param(
            np.array(
                [
                    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                    [[0.0, 0.0, 1.0], [1.0, 1.0, 1.0]],
                ],
                dtype=np.float32,
            ),
            None,
            id="green-brighter-than-red-and-blue",
        ),
    ],
)
def test_rgb_to_gray(image_float: np.ndarray, expected: np.ndarray | None) -> None:
    result = rgb_to_gray(image_float)

    assert result.shape == image_float.shape[:2]
    assert np.issubdtype(result.dtype, np.floating)

    if expected is not None:
        np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-6)
    else:
        assert result[0, 1] > result[0, 0] > result[1, 0]
        assert result[1, 1] == pytest.approx(1.0, rel=1e-6, abs=1e-6)


# -----------------------------------------------------------------------------
# Question 6: channel_summary
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("image_float", "expected_means", "expected_brightest"),
    [
        pytest.param(
            np.array(
                [
                    [[0.0, 0.5, 1.0], [1.0, 0.5, 0.5]],
                    [[0.5, 0.5, 0.5], [1.0, 0.0, 1.0]],
                ],
                dtype=np.float32,
            ),
            np.array([0.625, 0.375, 0.75], dtype=np.float32),
            2,
            id="computes-channel-means",
        ),
        pytest.param(
            np.array(
                [
                    [[0.4, 0.1, 0.4], [0.4, 0.1, 0.4]],
                    [[0.4, 0.1, 0.4], [0.4, 0.1, 0.4]],
                ],
                dtype=np.float32,
            ),
            np.array([0.4, 0.1, 0.4], dtype=np.float32),
            0,
            id="ties-pick-first-maximum",
        ),
    ],
)
def test_channel_summary(
    image_float: np.ndarray, expected_means: np.ndarray, expected_brightest: int
) -> None:
    means, brightest = channel_summary(image_float)

    np.testing.assert_allclose(means, expected_means, rtol=1e-6, atol=1e-6)
    assert brightest == expected_brightest


# -----------------------------------------------------------------------------
# Question 7: convolve2d_matmul
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Question 7: convolve2d_matmul
# -----------------------------------------------------------------------------

TEST_IMAGE_GRAY = np.array(
    [
        [0, 1, 2, 3, 4],
        [5, 6, 7, 8, 9],
        [10, 11, 12, 13, 14],
        [15, 16, 17, 18, 19],
        [20, 21, 22, 23, 24],
    ],
    dtype=np.float32,
)


@pytest.mark.parametrize(
    ("kernel", "expected"),
    [
        pytest.param(
            GAUSSIAN_KERNEL,
            np.array(
                [
                    [6.0, 7.0, 8.0],
                    [11.0, 12.0, 13.0],
                    [16.0, 17.0, 18.0],
                ],
                dtype=np.float32,
            ),
            id="gaussian-kernel-ramp-image",
        ),
        pytest.param(
            SOBEL_Y_KERNEL,
            np.array(
                [
                    [-40.0, -40.0, -40.0],
                    [-40.0, -40.0, -40.0],
                    [-40.0, -40.0, -40.0],
                ],
                dtype=np.float32,
            ),
            id="sobel-y-kernel-ramp-image",
        ),
    ],
)
def test_convolve2d_matmul(kernel: np.ndarray, expected: np.ndarray) -> None:
    result = convolve2d_matmul(TEST_IMAGE_GRAY, kernel)
    np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-6)

# -----------------------------------------------------------------------------
# Question 8: flatten_image
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("image", "expected"),
    [
        pytest.param(
            np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32),
            np.array([1, 2, 3, 4, 5, 6], dtype=np.float32),
            id="2d-array",
        ),
        pytest.param(
            np.array(
                [
                    [[1, 2], [3, 4]],
                    [[5, 6], [7, 8]],
                ],
                dtype=np.float32,
            ),
            np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float32),
            id="3d-array",
        ),
    ],
)
def test_flatten_image(image: np.ndarray, expected: np.ndarray) -> None:
    result = flatten_image(image)

    assert result.ndim == 1
    np.testing.assert_array_equal(result, expected)


# -----------------------------------------------------------------------------
# Question 9: extract_features
# -----------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("kernel", "channel_means", "brightest_channel", "gray", "filtered", "image_float"),
    [
        pytest.param(
            GAUSSIAN_KERNEL,
            np.array([0.1, 0.2, 0.3], dtype=np.float32),
            2,
            np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32),
            np.full((2, 2), 7.0, dtype=np.float32),
            np.zeros((2, 2, 3), dtype=np.float32),
            id="gaussian-kernel-features",
        ),
        pytest.param(
            SOBEL_Y_KERNEL,
            np.array([0.7, 0.1, 0.2], dtype=np.float32),
            0,
            np.array([[1.0, 1.0], [3.0, 5.0]], dtype=np.float32),
            np.array([[1.0, -1.0], [-1.0, 1.0]], dtype=np.float32),
            np.array(
                [
                    [[0.0, 0.5, 0.25], [1.0, 0.5, 0.25]],
                    [[2.0, 0.5, 0.25], [3.0, 0.5, 0.25]],
                ],
                dtype=np.float32,
            ),
            id="sobel-kernel-features",
        ),
    ],
)
def test_extract_features_with_kernels(
    monkeypatch: pytest.MonkeyPatch,
    kernel: np.ndarray,
    channel_means: np.ndarray,
    brightest_channel: int,
    gray: np.ndarray,
    filtered: np.ndarray,
    image_float: np.ndarray,
) -> None:
    image = np.zeros((64, 64, 3), dtype=np.uint8)

    # Mock pipeline
    monkeypatch.setattr(
        NOTEBOOK_MODULE,
        "center_crop",
        lambda _image, crop_size=48: image_float.astype(np.uint8),
    )
    monkeypatch.setattr(
        NOTEBOOK_MODULE,
        "normalize_01",
        lambda _image: image_float,
    )
    monkeypatch.setattr(
        NOTEBOOK_MODULE,
        "rgb_to_gray",
        lambda _image_float: gray,
    )
    monkeypatch.setattr(
        NOTEBOOK_MODULE,
        "channel_summary",
        lambda _image_float: (channel_means, brightest_channel),
    )

    # 👇 Important: ensure kernel is passed correctly
    def fake_convolve(_gray, passed_kernel):
        assert np.array_equal(passed_kernel, kernel)
        return filtered

    monkeypatch.setattr(
        NOTEBOOK_MODULE,
        "convolve2d_matmul",
        fake_convolve,
    )

    result = extract_features(image, kernel)

    # Basic checks
    assert result.dtype == np.float32
    assert result.shape == (10,)

    # Expected computation
    channel_stds = image_float.std(axis=(0, 1)).astype(np.float32)
    row_std_mean = np.apply_along_axis(np.std, 1, gray).mean().astype(np.float32)

    expected_vector = np.concatenate(
        [
            channel_means,
            channel_stds,
            np.array(
                [
                    float(brightest_channel),
                    float(filtered.mean()),
                    float(filtered.std()),
                    float(row_std_mean),
                ],
                dtype=np.float32,
            ),
        ]
    ).astype(np.float32)

    np.testing.assert_allclose(result, expected_vector, rtol=1e-6, atol=1e-6)

# -----------------------------------------------------------------------------
# Question 10: build_feature_matrix
# -----------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("image_names", "kernel", "expected_labels"),
    [
        pytest.param(
            ["cat/cat_0005.jpg", "dog/dog_0005.jpg"],
            GAUSSIAN_KERNEL,
            np.array([0, 1], dtype=np.int64),
            id="cat-dog-gaussian",
        ),
        pytest.param(
            ["dog/dog_0005.jpg", "cat/cat_0005.jpg"],
            SOBEL_Y_KERNEL,
            np.array([1, 0], dtype=np.int64),
            id="dog-cat-sobel",
        ),
    ],
)
def test_build_feature_matrix(
    monkeypatch: pytest.MonkeyPatch,
    image_names: list[str],
    kernel: np.ndarray,
    expected_labels: np.ndarray,
) -> None:
    paths = [TEST_DATA_DIR / name for name in image_names]

    for path in paths:
        assert path.exists(), f"Missing test image: {path}"

    def fake_extract_features(image: np.ndarray, passed_kernel: np.ndarray) -> np.ndarray:
        assert np.array_equal(passed_kernel, kernel)
        value = float(image[0, 0, 0])
        return np.array([value, value + 0.5], dtype=np.float32)

    monkeypatch.setattr(NOTEBOOK_MODULE, "extract_features", fake_extract_features)

    X, y = build_feature_matrix(paths, kernel)

    assert X.shape == (len(paths), 2)
    assert y.shape == (len(paths),)
    np.testing.assert_array_equal(y, expected_labels)

    # Since the test images are different, the first pixel values should differ too.
    assert X[0, 0] != X[1, 0]
