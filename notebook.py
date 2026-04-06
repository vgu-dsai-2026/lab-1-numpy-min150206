from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from lab_utils.visualization import plot_feature_vector, show_image_gallery
DATA_ROOT = Path('data')
LABELS = ('cat', 'dog')
LABEL_TO_INDEX = {'cat': 0, 'dog': 1}
IMAGE_EXTENSIONS = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp')
SEED = 1234

def label_from_path(path: Path) -> str:
    label = path.parent.name
    if label not in LABEL_TO_INDEX:
        raise ValueError(f'Unexpected label folder: {path}')
    return label

def load_preview_image(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert('RGB'))

def list_image_paths(label: str) -> list[Path]:
    label_dir = DATA_ROOT / label
    paths = []
    for pattern in IMAGE_EXTENSIONS:
        paths.extend(label_dir.glob(pattern))
    return sorted(paths)

def shuffled_paths(paths: list[Path], seed_offset: int=0) -> list[Path]:
    rng = np.random.default_rng(SEED + seed_offset)
    indices = rng.permutation(len(paths))
    return [paths[int(idx)] for idx in indices]

def sample_paths(paths: list[Path], count: int, seed_offset: int) -> list[Path]:
    ordered = shuffled_paths(paths, seed_offset=seed_offset)
    return ordered[:min(count, len(ordered))]

def sample_per_class(paths: list[Path], n_per_class: int, seed_offset: int=0) -> list[Path]:
    sampled = []
    for label_index, label in enumerate(LABELS):
        label_paths = [path for path in paths if label_from_path(path) == label]
        sampled.extend(sample_paths(label_paths, n_per_class, seed_offset + 50 * label_index))
    return sampled

def split_train_test(paths: list[Path], train_ratio: float=0.7, seed_offset: int=0):
    shuffled = shuffled_paths(paths, seed_offset)
    split_idx = int(len(shuffled) * train_ratio)
    return (shuffled[:split_idx], shuffled[split_idx:])
expected = [DATA_ROOT / 'cat', DATA_ROOT / 'dog']
cat_paths = list_image_paths('cat')
dog_paths = list_image_paths('dog')
cat_dog_paths = cat_paths + dog_paths
cat_train, cat_test = split_train_test(cat_paths, 0.7, seed_offset=0)
dog_train, dog_test = split_train_test(dog_paths, 0.7, seed_offset=100)
train_paths = cat_train + dog_train
test_paths = cat_test + dog_test
preview_paths = sample_per_class(cat_dog_paths, n_per_class=3, seed_offset=10)
preview_images = [load_preview_image(path) for path in preview_paths]
preview_titles = [f'{label_from_path(path)}: {path.name}' for path in preview_paths]

def load_image_np(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        return np.asarray(img.convert('RGB'))
sample_path = cat_paths[0]
sample_image = load_image_np(sample_path)

def center_crop(image: np.ndarray, crop_size: int=48) -> np.ndarray:
    h, w = image.shape[:2]
    row_start = (h - crop_size) // 2
    col_start = (w - crop_size) // 2
    return image[row_start:row_start + crop_size, col_start:col_start + crop_size]
cropped_image = center_crop(sample_image, crop_size=48)

def flip_horizontal(image: np.ndarray) -> np.ndarray:
    return image[:, ::-1]
flipped_image = flip_horizontal(cropped_image)

def normalize_01(image: np.ndarray) -> np.ndarray:
    return image.astype(np.float32) / 255.0

def show_histograms(uint8_img, float_img):
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.hist(uint8_img.ravel(), bins=50)
    plt.title('Before (uint8: 0–255)')
    plt.subplot(1, 2, 2)
    plt.hist(float_img.ravel(), bins=50)
    plt.title('After (float: 0–1)')
    plt.tight_layout()
    plt.show()
sample_float = normalize_01(cropped_image)

def rgb_to_gray(image_float: np.ndarray) -> np.ndarray:
    gray = 0.299 * image_float[..., 0] + 0.587 * image_float[..., 1] + 0.114 * image_float[..., 2]
    return gray.astype(np.float32)
sample_gray = rgb_to_gray(sample_float)
fig, axes = plt.subplots(1, 2, figsize=(8, 4))
CHANNEL_NAMES = np.array(['red', 'green', 'blue'])

def channel_summary(image_float: np.ndarray) -> tuple[np.ndarray, int]:
    means = image_float.mean(axis=(0, 1))
    brightest = int(np.argmax(means))
    return (means, brightest)
sample_channel_means, sample_brightest = channel_summary(sample_float)
fig, ax = plt.subplots(figsize=(5, 3))
EDGE_KERNEL = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)

def convolve2d_matmul(image_gray: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    kh, kw = kernel.shape
    h, w = image_gray.shape
    out_h, out_w = (h - kh + 1, w - kw + 1)
    k_flat = kernel.flatten()
    out = np.zeros((out_h, out_w), dtype=np.float32)
    for r in range(out_h):
        for c in range(out_w):
            patch = image_gray[r:r + kh, c:c + kw].flatten()
            out[r, c] = patch @ k_flat
    return out
sample_filtered = convolve2d_matmul(sample_gray, EDGE_KERNEL)
fig, axes = plt.subplots(1, 2, figsize=(8, 4))

def flatten_image(image: np.ndarray) -> np.ndarray:
    return image.flatten()
sample_flat = flatten_image(sample_gray)
fig, ax = plt.subplots(figsize=(10, 3))
FEATURE_NAMES = ['mean_r', 'mean_g', 'mean_b', 'std_r', 'std_g', 'std_b', 'brightest_channel', 'edge_mean', 'edge_std', 'row_std_mean']

def extract_features(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    cropped = center_crop(image, crop_size=48)
    image_float = normalize_01(cropped)
    gray = rgb_to_gray(image_float)
    channel_means, brightest_channel = channel_summary(image_float)
    channel_stds = image_float.std(axis=(0, 1)).astype(np.float32)
    filtered = convolve2d_matmul(gray, kernel)
    row_std_profile = np.apply_along_axis(np.std, 1, gray)
    row_std_mean = np.array([np.apply_along_axis(np.std, 1, gray).mean()], dtype=np.float32)
    features = np.concatenate([channel_means, channel_stds, np.array([brightest_channel], dtype=np.float32), np.array([filtered.mean()], dtype=np.float32), np.array([filtered.std()], dtype=np.float32), row_std_mean])
    return features.astype(np.float32)
sample_features = extract_features(sample_image, EDGE_KERNEL)
fig, ax = plot_feature_vector(sample_features, FEATURE_NAMES, title='Sample NumPy feature vector')

def build_feature_matrix(paths: list[Path], kernel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    fl = []
    ll = []
    for path in paths:
        image = load_image_np(path)
        feat = extract_features(image, kernel)
        fl.append(feat)
        ll.append(LABEL_TO_INDEX[label_from_path(path)])
    X = np.stack(fl)
    y = np.array(ll, dtype=np.int64)
    return (X, y)
X_train, y_train = build_feature_matrix(train_paths, EDGE_KERNEL)
X_test, y_test = build_feature_matrix(test_paths, EDGE_KERNEL)
train_feature_mean = X_train.mean(axis=0)
fig, ax = plt.subplots(figsize=(10, 4))
image = ax.imshow(X_train, aspect='auto', cmap='viridis')
fig, ax = plot_feature_vector(train_feature_mean, FEATURE_NAMES, title='Average training feature vector')
