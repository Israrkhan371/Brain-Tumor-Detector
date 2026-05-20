import os
import cv2
import h5py
import numpy as np

from skimage.feature import (
    graycomatrix,
    graycoprops,
    hog,
    local_binary_pattern
)

# =========================================================
# CLEAN PATH
# =========================================================

def clean_path(path):

    path = path.strip()

    path = path.replace("{", "")
    path = path.replace("}", "")

    path = path.replace("file:///", "")

    return path

# =========================================================
# FEATURE EXTRACTION
# =========================================================

def extract_features(image):

    features = []

    # Resize
    image = cv2.resize(image, (128, 128))

    # Convert type
    image = image.astype(np.uint8)

    # Blur
    image = cv2.GaussianBlur(image, (5, 5), 0)

    # =====================================================
    # GLCM FEATURES
    # =====================================================

    glcm = graycomatrix(
        image,
        distances=[1, 2, 3],
        angles=[0, np.pi / 4, np.pi / 2],
        symmetric=True,
        normed=True
    )

    properties = [
        'contrast',
        'correlation',
        'energy',
        'homogeneity',
        'ASM',
        'dissimilarity'
    ]

    for prop in properties:

        values = graycoprops(glcm, prop)

        features.extend(values.flatten())

    # =====================================================
    # HOG FEATURES
    # =====================================================

    hog_features = hog(
        image,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        visualize=False
    )

    features.extend(hog_features[:500])

    # =====================================================
    # LBP FEATURES
    # =====================================================

    radius = 1

    n_points = 8 * radius

    lbp = local_binary_pattern(
        image,
        n_points,
        radius,
        method='uniform'
    )

    hist, _ = np.histogram(
        lbp.ravel(),
        bins=np.arange(0, n_points + 3),
        range=(0, n_points + 2)
    )

    hist = hist.astype(float)

    hist /= (hist.sum() + 1e-6)

    features.extend(hist)

    # =====================================================
    # STATISTICAL FEATURES
    # =====================================================

    features.extend([

        np.mean(image),
        np.std(image),
        np.var(image),
        np.max(image),
        np.min(image),
        np.median(image)

    ])

    return np.array(features)

# =========================================================
# LOAD IMAGE
# =========================================================

def load_image(path):

    path = clean_path(path)

    ext = os.path.splitext(path)[1].lower()

    # =====================================================
    # LOAD .MAT FILE
    # =====================================================

    if ext == ".mat":

        mat = h5py.File(path, 'r')

        cjdata = mat['cjdata']

        image = np.array(cjdata['image'])

        label = int(np.array(cjdata['label'])[0][0]) - 1

    # =====================================================
    # LOAD NORMAL IMAGE
    # =====================================================

    else:

        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise ValueError("Cannot load image.")

        label = None

    return image, label