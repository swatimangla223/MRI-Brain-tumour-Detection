import os
import warnings

import cv2
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from skimage.feature import hog

warnings.filterwarnings("ignore")

DATA_DIR = "dataset"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")
IMG_SIZE = (128, 128)
HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (2, 2)


def load_data_from_dir(data_dir):
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Directory not found: {data_dir}")

    images = []
    labels = []
    class_names = sorted(
        [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    )

    if not class_names:
        raise ValueError(f"No class folders found in {data_dir}")

    for label in class_names:
        class_path = os.path.join(data_dir, label)
        for img_file in sorted(os.listdir(class_path)):
            if not img_file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
                continue

            img_path = os.path.join(class_path, img_file)
            img = cv2.imread(img_path)
            if img is None:
                continue

            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            resized_img = cv2.resize(gray_img, IMG_SIZE)
            images.append(resized_img)
            labels.append(label)

    if not images:
        raise ValueError(f"No valid images found in {data_dir}")

    return np.array(images), np.array(labels), class_names


print("Loading training data...")
X_train_raw, y_train_raw, class_names = load_data_from_dir(TRAIN_DIR)
print(f"Found {len(X_train_raw)} training images in classes: {class_names}")

if os.path.exists(TEST_DIR):
    print("Loading test data...")
    X_test_raw, y_test_raw, _ = load_data_from_dir(TEST_DIR)
    print(f"Found {len(X_test_raw)} test images")
    X_all = np.concatenate([X_train_raw, X_test_raw])
    y_all = np.concatenate([y_train_raw, y_test_raw])
else:
    X_all = X_train_raw
    y_all = y_train_raw

le = LabelEncoder()
y_encoded = le.fit_transform(y_all)


def extract_hog_features(images):
    features = []
    for img in images:
        hog_feat = hog(
            img,
            orientations=HOG_ORIENTATIONS,
            pixels_per_cell=HOG_PIXELS_PER_CELL,
            cells_per_block=HOG_CELLS_PER_BLOCK,
            transform_sqrt=True,
            block_norm="L2-Hys",
        )
        features.append(hog_feat)
    return np.array(features)


print("Extracting HOG features...")
X_features = extract_hog_features(X_all)
print(f"Feature vector size: {X_features.shape[1]}")

X_train, X_test, y_train, y_test = train_test_split(
    X_features,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded,
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Training Decision Tree Classifier...")
clf = DecisionTreeClassifier(max_depth=20, random_state=42, min_samples_split=10)
clf.fit(X_train_scaled, y_train)

y_pred = clf.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {acc:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))

joblib.dump(clf, "decision_tree_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(le, "label_encoder.pkl")
print("Model, scaler, and label encoder saved successfully.")
