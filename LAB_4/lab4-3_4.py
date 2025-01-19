import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
import matplotlib.pyplot as plt

# Завантаження та попередня обробка датасету Fashion MNIST
(train_images, _), (test_images, _) = tf.keras.datasets.fashion_mnist.load_data()
train_images = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0
train_images = np.expand_dims(train_images, axis=-1)
test_images = np.expand_dims(test_images, axis=-1)

# Параметри моделі
latent_dim = 16  # Розмір латентного простору
input_dim = (28, 28, 1)

# Створення енкодера
input_layer = layers.Input(shape=input_dim)
x = layers.Conv2D(32, kernel_size=3, activation='relu', strides=2, padding='same')(input_layer)
x = layers.Conv2D(64, kernel_size=3, activation='relu', strides=2, padding='same')(x)
x = layers.Flatten()(x)
x = layers.Dense(128, activation='relu')(x)
mean = layers.Dense(latent_dim, name='latent_mean')(x)
log_variance = layers.Dense(latent_dim, name='latent_log_var')(x)

# Функція для вибірки
def reparameterization(args):
    mean, log_var = args
    epsilon = tf.random.normal(shape=tf.shape(mean))
    return mean + tf.exp(0.5 * log_var) * epsilon

latent_vector = layers.Lambda(reparameterization, output_shape=(latent_dim,), name='latent_vector')([mean, log_variance])

# Створення енкодера
encoder = Model(input_layer, [mean, log_variance, latent_vector], name='Encoder')

# Створення декодера
latent_input = layers.Input(shape=(latent_dim,))
y = layers.Dense(7 * 7 * 64, activation='relu')(latent_input)
y = layers.Reshape((7, 7, 64))(y)
y = layers.Conv2DTranspose(64, kernel_size=3, activation='relu', strides=2, padding='same')(y)
y = layers.Conv2DTranspose(32, kernel_size=3, activation='relu', strides=2, padding='same')(y)
output_layer = layers.Conv2DTranspose(1, kernel_size=3, activation='sigmoid', padding='same')(y)

decoder = Model(latent_input, output_layer, name='Decoder')

# VAE з кастомним обчисленням втрат
class VariationalAutoencoder(Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VariationalAutoencoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)
        reconstruction_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(inputs, reconstructed), axis=(1, 2))
        reconstruction_loss *= 28 * 28
        kl_divergence = -0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
        total_loss = tf.reduce_mean(reconstruction_loss + kl_divergence)
        self.add_loss(total_loss)
        return reconstructed

vae = VariationalAutoencoder(encoder, decoder)
vae.compile(optimizer=tf.keras.optimizers.Adam())
vae.fit(train_images, train_images, epochs=20, batch_size=128, validation_data=(test_images, test_images))

# Візуалізація результатів
def visualize_generated_images(decoder, num_images=10):
    random_latents = np.random.normal(size=(num_images, latent_dim))
    generated = decoder.predict(random_latents)
    plt.figure(figsize=(15, 3))
    for i in range(num_images):
        plt.subplot(1, num_images, i + 1)
        plt.imshow(generated[i, :, :, 0], cmap='gray')
        plt.axis('off')
    plt.show()

visualize_generated_images(decoder)