import os
import glob
import csv
import tensorflow as tf

class GenerationEvaluator:
    def __init__(self,
                 vae_loss_file="vae_losses.txt",
                 gan_loss_file="gan_losses.txt",
                 vae_img_dir="generated/vae",
                 gan_img_dir="generated/gan",
                 log_dir="evaluation_logs"):
        self.vae_loss_file = vae_loss_file
        self.gan_loss_file = gan_loss_file
        self.vae_img_dir = vae_img_dir
        self.gan_img_dir = gan_img_dir
        self.log_dir = log_dir

        # Create writer
        self.writer = tf.summary.create_file_writer(self.log_dir)

    def load_vae_losses(self):
        epochs, losses = [], []
        with open(self.vae_loss_file, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                epochs.append(int(row["epoch"]))
                losses.append(float(row["avg_loss"]))
        return epochs, losses

    def load_gan_losses(self):
        epochs, d_losses, g_losses = [], [], []
        with open(self.gan_loss_file, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                epochs.append(int(row["epoch"]))
                d_losses.append(float(row["avg_d_loss"]))
                g_losses.append(float(row["avg_g_loss"]))
        return epochs, d_losses, g_losses

    def log_losses(self):
        epochs, vae_losses = self.load_vae_losses()
        _, d_losses, g_losses = self.load_gan_losses()
        with self.writer.as_default():
            # VAE loss
            for e, loss in zip(epochs, vae_losses):
                tf.summary.scalar("VAE/avg_loss", loss, step=e)
            # GAN losses
            for e, d, g in zip(epochs, d_losses, g_losses):
                tf.summary.scalar("GAN/discriminator_loss", d, step=e)
                tf.summary.scalar("GAN/generator_loss", g, step=e)
        print(f"Logged losses to TensorBoard under '{self.log_dir}'")

    def load_images(self, directory, max_images=8):
        # Gather PNGs
        paths = sorted(glob.glob(os.path.join(directory, "*.png")))
        paths = paths[:max_images]
        imgs = []
        for p in paths:
            img_raw = tf.io.read_file(p)
            img = tf.image.decode_png(img_raw, channels=3)
            img = tf.image.convert_image_dtype(img, tf.float32)  # [0,1]
            imgs.append(img)
        if imgs:
            return tf.stack(imgs, axis=0)
        else:
            return None

    def log_images(self):
        vae_imgs = self.load_images(self.vae_img_dir)
        gan_imgs = self.load_images(self.gan_img_dir)
        with self.writer.as_default():
            if vae_imgs is not None:
                tf.summary.image("VAE/generated_samples", vae_imgs, max_outputs=vae_imgs.shape[0], step=0)
                tf.summary.histogram("VAE/pixel_values", tf.reshape(vae_imgs, [-1]), step=0)
            if gan_imgs is not None:
                tf.summary.image("GAN/generated_samples", gan_imgs, max_outputs=gan_imgs.shape[0], step=0)
                tf.summary.histogram("GAN/pixel_values", tf.reshape(gan_imgs, [-1]), step=0)
        print(f"Logged sample images to TensorBoard under '{self.log_dir}'")

    def run_all(self):
        self.log_losses()
        self.log_images()


if __name__ == "__main__":
    evaluator = GenerationEvaluator(
        vae_loss_file="loss/vae_losses.txt",
        gan_loss_file="loss/gan_losses.txt",
        vae_img_dir="generated/vae",
        gan_img_dir="generated/gan",
        log_dir="evaluation_logs"
    )
    evaluator.run_all()
    print("Done. Launch `tensorboard --logdir evaluation_logs` to inspect.")
