#CSCE 479/879 Hackathon 5 homework

from __future__ import print_function

import csv
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

TRAIN_NOISE = 0.1
TEST_NOISE_LEVELS = [0.0, 0.2, 0.4, 0.6, 0.8]
EPOCHS = 10
BATCH_SIZE = 64
SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)


def add_salt_pepper_noise(images, prob=0.1):
    """Randomly replace pixels with black or white, independently per channel."""
    random_values = tf.random.uniform(tf.shape(images))
    noisy_images = tf.where(random_values < prob / 2.0, 0.0, images)
    noisy_images = tf.where(random_values > 1.0 - prob / 2.0, 1.0, noisy_images)
    return noisy_images


class ResBlock(tf.keras.layers.Layer):
    def __init__(self, filters):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters, 3, padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(filters, 3, padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization()

    def call(self, x, training=False):
        residual = self.conv1(x)
        residual = self.bn1(residual, training=training)
        residual = tf.nn.relu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual, training=training)
        return tf.nn.relu(x + residual)


class Bottleneck(tf.keras.layers.Layer):
    def __init__(self, filters):
        super().__init__()
        bottleneck_filters = filters // 4
        self.conv1 = tf.keras.layers.Conv2D(bottleneck_filters, 1, padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(bottleneck_filters, 3, padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.conv3 = tf.keras.layers.Conv2D(filters, 1, padding='same')
        self.bn3 = tf.keras.layers.BatchNormalization()

    def call(self, x, training=False):
        residual = self.conv1(x)
        residual = self.bn1(residual, training=training)
        residual = tf.nn.relu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual, training=training)
        residual = tf.nn.relu(residual)
        residual = self.conv3(residual)
        residual = self.bn3(residual, training=training)
        return tf.nn.relu(x + residual)


def create_model(block_type):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(32, 32, 3)))
    model.add(tf.keras.layers.Conv2D(32, 3, padding='same', activation=tf.nn.relu))
    if block_type == 'Residual':
        model.add(ResBlock(32))
    else:
        model.add(Bottleneck(32))
    model.add(tf.keras.layers.MaxPool2D(pool_size=2))
    model.add(tf.keras.layers.Conv2D(64, 3, padding='same', activation=tf.nn.relu))
    model.add(tf.keras.layers.GlobalAveragePooling2D())
    model.add(tf.keras.layers.Dense(10))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    return model


def main():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train = x_train.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0
    y_train = y_train.reshape(-1)
    y_test = y_test.reshape(-1)

    x_validation, y_validation = x_train[-5000:], y_train[-5000:]
    x_train, y_train = x_train[:-5000], y_train[:-5000]

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_ds = train_ds.shuffle(10000, seed=SEED).batch(BATCH_SIZE)
    train_ds = train_ds.map(
        lambda images, labels: (add_salt_pepper_noise(images, TRAIN_NOISE), labels),
        num_parallel_calls=tf.data.AUTOTUNE
    ).prefetch(tf.data.AUTOTUNE)

    validation_images = add_salt_pepper_noise(
        tf.convert_to_tensor(x_validation), TRAIN_NOISE
    )
    validation_ds = tf.data.Dataset.from_tensor_slices(
        (validation_images, y_validation)
    ).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    results = {}
    for block_type in ['Residual', 'Bottleneck']:
        print('\nTraining', block_type, 'model')
        model = create_model(block_type)
        model.summary()
        history = model.fit(
            train_ds,
            validation_data=validation_ds,
            epochs=EPOCHS,
            callbacks=[tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=2, restore_best_weights=True
            )]
        )
        print('Epochs trained:', len(history.history['loss']))

        accuracies = []
        for noise_level in TEST_NOISE_LEVELS:
            tf.random.set_seed(SEED)
            noisy_test = add_salt_pepper_noise(
                tf.convert_to_tensor(x_test), noise_level
            )
            test_ds = tf.data.Dataset.from_tensor_slices(
                (noisy_test, y_test)
            ).batch(BATCH_SIZE)
            loss, accuracy = model.evaluate(test_ds, verbose=0)
            accuracies.append(float(accuracy))
            print(f'{block_type}: noise={noise_level:.1f}, '
                  f'test accuracy={accuracy:.4f}, loss={loss:.4f}')
        results[block_type] = accuracies

    print('\nNoise Level    Residual Accuracy    Bottleneck Accuracy')
    for i, noise_level in enumerate(TEST_NOISE_LEVELS):
        print(f'{noise_level:10.1f}    {results["Residual"][i]:17.4f}    '
              f'{results["Bottleneck"][i]:19.4f}')

    with open('N05hw_results.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Noise Level', 'Residual Accuracy', 'Bottleneck Accuracy'])
        for i, noise_level in enumerate(TEST_NOISE_LEVELS):
            writer.writerow([noise_level, results['Residual'][i],
                             results['Bottleneck'][i]])

    plt.plot(TEST_NOISE_LEVELS, results['Residual'], marker='o', label='Residual')
    plt.plot(TEST_NOISE_LEVELS, results['Bottleneck'], marker='o', label='Bottleneck')
    plt.xlabel('Salt-and-Pepper Noise Level')
    plt.ylabel('Test Accuracy')
    plt.title('CIFAR-10 Classification Accuracy vs. Noise Level')
    plt.xticks(TEST_NOISE_LEVELS)
    plt.legend()
    plt.tight_layout()
    plt.savefig('N05hw_accuracy.png', dpi=200)
    plt.show()


if __name__ == '__main__':
    main()
