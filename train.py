import os
import sys
import numpy as np
import tensorflow as tf
from src.pipeline import HideNetPipeline
from src.dataset import load_images_from_folder, create_dataset, split_dataset

EPOCHS = 50
BATCH_SIZE = 4
IMG_SIZE = 256
SAVE_PATH = '/content/drive/MyDrive/HideNet/models'
COVER_PATH = '/content/drive/MyDrive/HideNet/data/cover'
SECRET_PATH = '/content/drive/MyDrive/HideNet/data/secret'

def train():
    print("Loading datasets...")
    cover_images = load_images_from_folder(COVER_PATH, max_images=800)
    secret_images = load_images_from_folder(SECRET_PATH, max_images=800)

    if len(cover_images) == 0 or len(secret_images) == 0:
        print("No images found! Please add images to data/cover and data/secret folders.")
        return

    cover_train, cover_val, _ = split_dataset(cover_images)
    secret_train, secret_val, _ = split_dataset(secret_images)

    train_dataset = create_dataset(cover_train, secret_train, batch_size=BATCH_SIZE)
    val_dataset = create_dataset(cover_val, secret_val, batch_size=BATCH_SIZE, shuffle=False)

    print("Building pipeline...")
    pipeline = HideNetPipeline(img_size=IMG_SIZE)

    best_loss = float('inf')
    history = []

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        epoch_losses = {
            'total_loss': [], 'hide_loss': [],
            'reveal_loss': [], 'adversarial_loss': [], 'detector_loss': []
        }

        for batch_idx, (cover_batch, secret_batch) in enumerate(train_dataset):
            losses = pipeline.train_step(cover_batch, secret_batch)
            for k, v in losses.items():
                epoch_losses[k].append(float(v))

            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx} - Total: {losses['total_loss']:.4f} "
                    f"Hide: {losses['hide_loss']:.4f} "
                    f"Reveal: {losses['reveal_loss']:.4f}")

        avg_losses = {k: np.mean(v) for k, v in epoch_losses.items()}
        history.append(avg_losses)
        print(f"Epoch {epoch+1} avg - Total: {avg_losses['total_loss']:.4f} "
            f"Hide: {avg_losses['hide_loss']:.4f} "
            f"Reveal: {avg_losses['reveal_loss']:.4f} "
            f"Detector: {avg_losses['detector_loss']:.4f}")

        if avg_losses['total_loss'] < best_loss:
            best_loss = avg_losses['total_loss']
            pipeline.save_models(SAVE_PATH)
            print(f"  -> Best model saved! Loss: {best_loss:.4f}")

    print("\nTraining complete!")
    np.save('/content/drive/MyDrive/HideNet/outputs/training_history.npy', history)
    print("History saved!")

if __name__ == "__main__":
    train()