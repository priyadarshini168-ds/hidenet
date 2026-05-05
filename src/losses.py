import tensorflow as tf

def ssim_loss(y_true, y_pred):
    return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))

def mse_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

def hide_loss(cover, stego, alpha=0.7, beta=0.3):
    return alpha * ssim_loss(cover, stego) + beta * mse_loss(cover, stego)

def reveal_loss(secret, revealed):
    return mse_loss(secret, revealed)

def adversarial_loss(detector_on_stego):
    # HideNet wants detector to output 0 (think stego is clean)
    return tf.reduce_mean(tf.square(detector_on_stego))

def detector_loss(detector_on_cover, detector_on_stego):
    # Detector wants: cover=0, stego=1
    real_loss = tf.reduce_mean(tf.square(detector_on_cover))
    fake_loss = tf.reduce_mean(tf.square(detector_on_stego - 1))
    return (real_loss + fake_loss) / 2

def total_hidenet_loss(cover, stego, secret, revealed, detector_on_stego,
    w_hide=1.0, w_reveal=1.0, w_adv=0.001):
    h_loss = hide_loss(cover, stego)
    r_loss = reveal_loss(secret, revealed)
    a_loss = adversarial_loss(detector_on_stego)
    total = w_hide * h_loss + w_reveal * r_loss + w_adv * a_loss
    return total, h_loss, r_loss, a_loss

if __name__ == "__main__":
    import numpy as np
    dummy = tf.constant(np.random.rand(2, 256, 256, 3).astype(np.float32))
    dummy2 = tf.constant(np.random.rand(2, 256, 256, 3).astype(np.float32))
    dummy_det = tf.constant(np.random.rand(2, 1).astype(np.float32))
    h = hide_loss(dummy, dummy2)
    r = reveal_loss(dummy, dummy2)
    a = adversarial_loss(dummy_det)
    print(f"Hide loss: {h:.4f}")
    print(f"Reveal loss: {r:.4f}")
    print(f"Adversarial loss: {a:.4f}")
    print("Losses module ready!")