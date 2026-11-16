# Hackathon 4 Homework


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = (
    tf.keras.datasets.mnist.load_data()
)

x_train = x_train.astype(np.float32) / 255.0
x_test = x_test.astype(np.float32) / 255.0
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

x_validation = x_train[54000:]
y_validation = y_train[54000:]

x_train = x_train[:54000]
y_train = y_train[:54000]


print("Training:", x_train.shape)
print("Validation:", x_validation.shape)
print("Testing:", x_test.shape)


train_fit = (
    tf.data.Dataset
    .from_tensor_slices((x_train, y_train))
    .shuffle(1024)
    .batch(32)
)

validation_fit = (
    tf.data.Dataset
    .from_tensor_slices((x_validation, y_validation))
    .batch(32)
)

test_fit = (
    tf.data.Dataset
    .from_tensor_slices((x_test, y_test))
    .batch(32)
)


mnist_model = tf.keras.Sequential()

mnist_model.add(
    tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=3,
        padding='same',
        activation=tf.nn.relu,
        name='hidden_1'
    )
)

mnist_model.add(
    tf.keras.layers.Conv2D(
        filters=64,
        kernel_size=3,
        padding='same',
        activation=tf.nn.relu,
        name='hidden_2'
    )
)

mnist_model.add(
    tf.keras.layers.MaxPool2D(
        padding='same'
    )
)

mnist_model.add(
    tf.keras.layers.Flatten()
)

mnist_model.add(
    tf.keras.layers.Dense(
        10,
        name='final'
    )
)


mnist_model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True
    ),
    metrics=['accuracy']
)


early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=2,
    restore_best_weights=True
)


history = mnist_model.fit(
    train_fit,
    validation_data=validation_fit,
    epochs=20,
    callbacks=[early_stopping]
)


test_loss, test_accuracy = mnist_model.evaluate(test_fit)

print()
print("Test Accuracy:", test_accuracy)
print("Test Loss:", test_loss)
print("Epochs:", len(history.history['loss']))


plt.plot(
    history.history['accuracy'],
    label='Training Accuracy'
)

plt.plot(
    history.history['val_accuracy'],
    label='Validation Accuracy'
)

plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('MNIST CNN Accuracy')
plt.legend()
plt.show()


true_labels = []
predicted_labels = []

for images, labels in test_fit:

    logits = mnist_model(images)

    predictions = tf.argmax(
        logits,
        axis=1
    )

    true_labels.extend(labels.numpy())
    predicted_labels.extend(predictions.numpy())


confusion = tf.math.confusion_matrix(
    true_labels,
    predicted_labels,
    num_classes=10
)

confusion = tf.cast(
    confusion,
    tf.float32
)


normalized_confusion = (
    confusion /
    tf.reduce_sum(
        confusion,
        axis=1,
        keepdims=True
    )
)

print()
print("Normalized Confusion Matrix:")
print(normalized_confusion)


plt.figure(figsize=(8, 8))

plt.imshow(
    normalized_confusion,
    cmap='Blues'
)

plt.colorbar()

plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Normalized MNIST Confusion Matrix')

plt.xticks(range(10))
plt.yticks(range(10))

for i in range(10):
    for j in range(10):

        plt.text(
            j,
            i,
            '{:.2f}'.format(
                normalized_confusion[i, j].numpy()
            ),
            ha='center',
            va='center'
        )

plt.show()


def interpolate_images(
    baseline,
    image,
    alphas
):

    alphas_x = alphas[:, tf.newaxis, tf.newaxis, tf.newaxis]

    baseline_x = tf.expand_dims(
        baseline,
        axis=0
    )

    input_x = tf.expand_dims(
        image,
        axis=0
    )

    delta = input_x - baseline_x

    images = (
        baseline_x +
        alphas_x * delta
    )

    return images


def compute_gradients(
    images,
    target_class_idx
):

    with tf.GradientTape() as tape:

        tape.watch(images)

        logits = mnist_model(images)

        target_logits = logits[
            :,
            target_class_idx
        ]

    return tape.gradient(
        target_logits,
        images
    )


def integral_approximation(
    gradients
):

    grads = (
        gradients[:-1] +
        gradients[1:]
    ) / 2.0

    integrated_gradients = tf.reduce_mean(
        grads,
        axis=0
    )

    return integrated_gradients


def integrated_gradients(
    baseline,
    image,
    target_class_idx,
    m_steps=50
):

    alphas = tf.linspace(
        start=0.0,
        stop=1.0,
        num=m_steps + 1
    )

    interpolated_images = interpolate_images(
        baseline=baseline,
        image=image,
        alphas=alphas
    )

    gradients = compute_gradients(
        images=interpolated_images,
        target_class_idx=target_class_idx
    )

    avg_gradients = integral_approximation(
        gradients=gradients
    )

    integrated_gradients = (
        image - baseline
    ) * avg_gradients

    return integrated_gradients


correct_samples = []
incorrect_samples = []

for i in range(len(true_labels)):

    if predicted_labels[i] == true_labels[i]:

        correct_samples.append(i)

    else:

        incorrect_samples.append(i)


print()
print(
    "First 5 Correct Samples:",
    correct_samples[:5]
)

print(
    "First 5 Incorrect Samples:",
    incorrect_samples[:5]
)


baseline = tf.zeros(
    shape=(28, 28, 1),
    dtype=tf.float32
)


def show_mnist_ig(sample_num):

    image = tf.convert_to_tensor(
        x_test[sample_num],
        dtype=tf.float32
    )

    true_label = y_test[sample_num]

    logits = mnist_model(
        tf.expand_dims(
            image,
            axis=0
        )
    )

    predicted_label = tf.argmax(
        logits,
        axis=1
    ).numpy()[0]

    attributions = integrated_gradients(
        baseline=baseline,
        image=image,
        target_class_idx=predicted_label,
        m_steps=50
    )

    attribution_mask = tf.reduce_sum(
        tf.math.abs(attributions),
        axis=-1
    )


    plt.figure(figsize=(8, 4))


    plt.subplot(1, 2, 1)

    plt.imshow(
        tf.squeeze(image),
        cmap='gray'
    )

    plt.title(
        'True: ' +
        str(true_label) +
        '  Predicted: ' +
        str(predicted_label)
    )

    plt.axis('off')


    plt.subplot(1, 2, 2)

    plt.imshow(
        attribution_mask,
        cmap='inferno'
    )

    plt.title(
        'Integrated Gradients'
    )

    plt.axis('off')

    plt.show()


print("\nCorrectly Classified Examples")

show_mnist_ig(
    correct_samples[0]
)

show_mnist_ig(
    correct_samples[1]
)


print("\nIncorrectly Classified Examples")

show_mnist_ig(
    incorrect_samples[0]
)

show_mnist_ig(
    incorrect_samples[1]
)