from __future__ import print_function

from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

DATA_DIR = str(Path.home() / 'tensorflow-datasets')
SEED = 479
BATCH_SIZE = 32
EPOCHS = 4
TRAIN_SAMPLES = 12000
VALIDATION_SAMPLES = 2000
TEST_SAMPLES = 2000
TSNE_SAMPLES = 600
SPARSITY_COEFF = 0.0005
NOISE_COEFF = 0.3   

tf.keras.utils.set_random_seed(SEED)


def prepare(sample):
    image = tf.cast(sample['image'], tf.float32) / 255.0
    return image, sample['label']


def load_data():
    print('Loading MNIST...', flush=True)
    ds = tfds.load('mnist', data_dir=DATA_DIR)
    train = ds['train'].take(TRAIN_SAMPLES).map(prepare)
    validation = ds['train'].skip(TRAIN_SAMPLES).take(VALIDATION_SAMPLES).map(prepare)
    test = ds['test'].take(TEST_SAMPLES).map(prepare)
    train = train.shuffle(1024, seed=SEED, reshuffle_each_iteration=True)
    return (train.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE),
            validation.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE),
            test.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE))


def create_autoencoder():
    """Notebook's convolutional encoder/decoder idea, using simple layers
    explicitly permitted by the homework. Latent shape: (7,7,8).
    """
    encoder_network = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(32, 3, padding='same', activation=tf.nn.swish),
        tf.keras.layers.Conv2D(32, 3, strides=2, padding='same', activation=tf.nn.swish),
        tf.keras.layers.Conv2D(16, 3, strides=2, padding='same', activation=tf.nn.swish),
        tf.keras.layers.Conv2D(8, 3, padding='same', activation=tf.nn.swish),
    ], name='encoder')

    # Transposed convolutions are one of the notebook's upscaling methods.
    decoder_network = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(7, 7, 8)),
        tf.keras.layers.Conv2DTranspose(16, 3, strides=2, padding='same', activation=tf.nn.swish),
        tf.keras.layers.Conv2DTranspose(32, 3, strides=2, padding='same', activation=tf.nn.swish),
        tf.keras.layers.Conv2D(1, 3, padding='same', activation='sigmoid'),
    ], name='decoder')
    return encoder_network, decoder_network


def reconstruction_loss(x, x_hat):
    return tf.reduce_mean(tf.square(x_hat - x))


def sparse_autoencoder_loss(x, code, x_hat):
    sparsity_loss = tf.reduce_mean(tf.reduce_sum(tf.abs(code), axis=[1, 2, 3]))
    return reconstruction_loss(x, x_hat) + SPARSITY_COEFF * sparsity_loss


def noisy_images(x):
    return tf.clip_by_value(x + NOISE_COEFF * tf.random.normal(tf.shape(x)), 0.0, 1.0)


def evaluate_mse(dataset, encoder_network, decoder_network):
    total_squared_error = 0.0
    total_pixels = 0
    for x, _ in dataset:
        code = encoder_network(x, training=False)
        output = decoder_network(code, training=False)
        total_squared_error += float(tf.reduce_sum(tf.square(output - x)).numpy())
        total_pixels += int(tf.size(x).numpy())
    return total_squared_error / total_pixels


def train_version(name, train_ds, validation_ds):
    tf.keras.utils.set_random_seed(SEED)
    encoder_network, decoder_network = create_autoencoder()
    optimizer = tf.keras.optimizers.Adam()
    trainable = encoder_network.trainable_variables + decoder_network.trainable_variables
    history = []

    print('\nTraining ' + name + '...', flush=True)
    for epoch in range(EPOCHS):
        running = []
        for batch in train_ds:
            x = batch[0]
            with tf.GradientTape() as tape:
                input_x = noisy_images(x) if name == 'Denoising AE' else x
                code = encoder_network(input_x, training=True)
                output = decoder_network(code, training=True)
                if name == 'Sparse AE':
                    loss = sparse_autoencoder_loss(x, code, output)
                else:
                    loss = reconstruction_loss(x, output)
            variables = encoder_network.trainable_variables + decoder_network.trainable_variables
            gradients = tape.gradient(loss, variables)
            optimizer.apply_gradients(zip(gradients, variables))
            running.append(float(loss.numpy()))

        val_mse = evaluate_mse(validation_ds, encoder_network, decoder_network)
        train_objective = float(np.mean(running))
        history.append({'model': name, 'epoch': epoch + 1,
                        'training_objective': train_objective,
                        'validation_mse': val_mse})
        print('  Epoch {}/{}: training objective={:.5f}, validation MSE={:.5f}'.format(
            epoch + 1, EPOCHS, train_objective, val_mse), flush=True)
    return encoder_network, decoder_network, history


def example_images(test_ds, models):
    x, _ = next(iter(test_ds))
    x = x[:5]
    tf.random.set_seed(SEED)
    noise = noisy_images(x)
    fig, axes = plt.subplots(len(x), 5, figsize=(11, 11))
    titles = ['Original', 'Noisy input', 'Baseline AE', 'Sparse AE', 'Denoising AE']
    outputs = {}
    for name, (encoder, decoder) in models.items():
        model_input = noise if name == 'Denoising AE' else x
        outputs[name] = decoder(encoder(model_input, training=False), training=False).numpy()
    for i in range(len(x)):
        imgs = [x[i].numpy(), noise[i].numpy(), outputs['Baseline AE'][i],
                outputs['Sparse AE'][i], outputs['Denoising AE'][i]]
        for j, img in enumerate(imgs):
            axes[i, j].imshow(img[:, :, 0], cmap='gray', vmin=0, vmax=1)
            axes[i, j].axis('off')
            if i == 0:
                axes[i, j].set_title(titles[j])
    fig.suptitle('MNIST original images and autoencoder reconstructions', y=0.99)
    fig.tight_layout()
    fig.savefig('N07hw_reconstructions.png', dpi=180, bbox_inches='tight')
    plt.close(fig)


def latent_tsne(test_ds, models):
    all_images = []
    all_labels = []
    for x, y in test_ds:
        all_images.append(x.numpy())
        all_labels.append(y.numpy())
    images = np.concatenate(all_images)[:TSNE_SAMPLES]
    labels = np.concatenate(all_labels)[:TSNE_SAMPLES]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, (name, (encoder, _)) in zip(axes, models.items()):
        codes = encoder(tf.constant(images), training=False).numpy()
        flat_codes = codes.reshape(len(codes), -1)
        print('Running t-SNE for ' + name + '...', flush=True)
        embedding = TSNE(n_components=2, random_state=SEED,
                         perplexity=30, init='pca', learning_rate='auto').fit_transform(flat_codes)
        plot = ax.scatter(embedding[:, 0], embedding[:, 1], c=labels,
                          cmap='tab10', s=9, alpha=0.8, vmin=-0.5, vmax=9.5)
        ax.set_title(name)
        ax.set_xlabel('t-SNE dimension 1')
        ax.set_ylabel('t-SNE dimension 2')
    fig.colorbar(plot, ax=axes, label='True MNIST digit', ticks=range(10), shrink=0.8)
    fig.suptitle('Latent representations visualized with t-SNE')
    fig.savefig('N07hw_tsne.png', dpi=180, bbox_inches='tight')
    plt.close(fig)


def main():
    train_ds, validation_ds, test_ds = load_data()
    models = {}
    histories = []
    summary = []

    for name in ['Baseline AE', 'Sparse AE', 'Denoising AE']:
        encoder, decoder, history = train_version(name, train_ds, validation_ds)
        models[name] = (encoder, decoder)
        histories.extend(history)
        val_mse = evaluate_mse(validation_ds, encoder, decoder)
        test_mse = evaluate_mse(test_ds, encoder, decoder)
        summary.append({'model': name, 'validation_mse': val_mse,
                        'test_mse': test_mse})

    history_df = pd.DataFrame(histories)
    summary_df = pd.DataFrame(summary)
    history_df.to_csv('N07hw_history.csv', index=False)
    summary_df.to_csv('N07hw_results.csv', index=False)
    print('\nValidation and test reconstruction MSE (lower is better):')
    print(summary_df.to_string(index=False, float_format=lambda x: f'{x:.6f}'))

    plt.figure(figsize=(7, 4))
    for name in models:
        subset = history_df[history_df['model'] == name]
        plt.plot(subset['epoch'], subset['validation_mse'], marker='o', label=name)
    plt.xlabel('Epoch')
    plt.ylabel('Validation reconstruction MSE')
    plt.title('MNIST Autoencoder Validation Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig('N07hw_validation_loss.png', dpi=180)
    plt.close()

    print('Creating reconstruction images...', flush=True)
    example_images(test_ds, models)
    latent_tsne(test_ds, models)
    print('\nSaved N07hw_results.csv, N07hw_history.csv,', flush=True)
    print('N07hw_validation_loss.png, N07hw_reconstructions.png, N07hw_tsne.png', flush=True)


if __name__ == '__main__':
    main()
