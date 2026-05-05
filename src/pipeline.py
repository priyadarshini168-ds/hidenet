import tensorflow as tf
from src.hidenet import build_hidenet
from src.revealnet import build_revealnet
from src.detector import build_detector
from src.losses import total_hidenet_loss, detector_loss

class HideNetPipeline:
    def __init__(self, img_size=256):
        self.img_size = img_size
        self.hidenet = build_hidenet((img_size, img_size, 3))
        self.revealnet = build_revealnet((img_size, img_size, 3))
        self.detector = build_detector((img_size, img_size, 3))

        self.hide_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002)
        self.reveal_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002)
        self.detector_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

    @tf.function
    def train_step(self, cover, secret):
        with tf.GradientTape(persistent=True) as tape:
            stego = self.hidenet([cover, secret], training=True)
            revealed = self.revealnet(stego, training=True)
            det_on_cover = self.detector(cover, training=True)
            det_on_stego = self.detector(stego, training=True)

            total_loss, h_loss, r_loss, a_loss = total_hidenet_loss(
                cover, stego, secret, revealed, det_on_stego)
            d_loss = detector_loss(det_on_cover, det_on_stego)

        hide_vars = self.hidenet.trainable_variables
        reveal_vars = self.revealnet.trainable_variables
        det_vars = self.detector.trainable_variables

        hide_grads = tape.gradient(total_loss, hide_vars)
        reveal_grads = tape.gradient(total_loss, reveal_vars)
        det_grads = tape.gradient(d_loss, det_vars)

        self.hide_optimizer.apply_gradients(zip(hide_grads, hide_vars))
        self.reveal_optimizer.apply_gradients(zip(reveal_grads, reveal_vars))
        self.detector_optimizer.apply_gradients(zip(det_grads, det_vars))

        del tape

        return {
            'total_loss': total_loss,
            'hide_loss': h_loss,
            'reveal_loss': r_loss,
            'adversarial_loss': a_loss,
            'detector_loss': d_loss
        }

    def save_models(self, path):
        self.hidenet.save(f'{path}/hidenet.h5')
        self.revealnet.save(f'{path}/revealnet.h5')
        self.detector.save(f'{path}/detector.h5')
        print(f"Models saved to {path}")

    def load_models(self, path):
        self.hidenet = tf.keras.models.load_model(f'{path}/hidenet.h5')
        self.revealnet = tf.keras.models.load_model(f'{path}/revealnet.h5')
        self.detector = tf.keras.models.load_model(f'{path}/detector.h5')
        print(f"Models loaded from {path}")

if __name__ == "__main__":
    import numpy as np
    print("Building pipeline...")
    pipeline = HideNetPipeline()
    cover = tf.constant(np.random.rand(2, 256, 256, 3).astype(np.float32))
    secret = tf.constant(np.random.rand(2, 256, 256, 3).astype(np.float32))
    print("Running one train step...")
    losses = pipeline.train_step(cover, secret)
    for k, v in losses.items():
        print(f"  {k}: {v:.4f}")
    print("Pipeline working!")
