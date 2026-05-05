import os
import numpy as np
import cv2
from pathlib import Path
import tensorflow as tf

IMG_SIZE = 256
BATCH_SIZE = 4
AUTOTUNE = tf.data.AUTOTUNE

def load_image(image_path):
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0
    return img

def load_images_from_folder(folder_path, max_images=1000):
    folder = Path(folder_path)
    extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_paths = []
    for ext in extensions:
        image_paths.extend(folder.glob(f'*{ext}'))
        image_paths.extend(folder.glob(f'*{ext.upper()}'))
    image_paths = sorted(image_paths)[:max_images]
    images = []
    for path in image_paths:
        img = load_image(path)
        if img is not None:
            images.append(img)
    print(f"Loaded {len(images)} images from {folder_path}")
    return np.array(images, dtype=np.float32)

def create_dataset(cover_images, secret_images, batch_size=BATCH_SIZE, shuffle=True):
    min_len = min(len(cover_images), len(secret_images))
    cover_images = cover_images[:min_len]
    secret_images = secret_images[:min_len]
    dataset = tf.data.Dataset.from_tensor_slices((cover_images, secret_images))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=min_len)
    dataset = dataset.batch(batch_size).prefetch(AUTOTUNE)
    return dataset

def split_dataset(images, train_ratio=0.8, val_ratio=0.1):
    total = len(images)
    train_end = int(total * train_ratio)
    val_end = int(total * (train_ratio + val_ratio))
    train = images[:train_end]
    val = images[train_end:val_end]
    test = images[val_end:]
    print(f"Split -> Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    return train, val, test

if __name__ == "__main__":
    print("Dataset module ready.")
    print(f"Image size: {IMG_SIZE}x{IMG_SIZE}")
    print(f"Batch size: {BATCH_SIZE}")