# Hackathon 6 part 1
from __future__ import print_function

import random
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import pandas as pd

DATA_DIR = r'C:\\tensorflow-datasets'
MAX_SEQ_LEN = 128
MAX_TOKENS = 5000
BATCH_SIZE = 32
EPOCHS = 3
TRAIN_SAMPLES = 12000
TEST_SAMPLES = 300
SEED = 479

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

print('hello')
def swap_adjacent(text):
    words = text.split()
    if len(words) > 1:
        i = random.randrange(len(words) - 1)
        words[i], words[i + 1] = words[i + 1], words[i]
    return ' '.join(words)


def shuffle_words(text):
    words = text.split()
    random.shuffle(words)
    return ' '.join(words)


def evaluate(texts, true_labels, model):
    predictions = []
    probabilities = []

    for i in range(0, len(texts), BATCH_SIZE):
        batch = tf.constant(texts[i:i + BATCH_SIZE])
        logits = model(batch, training=False)
        probs = tf.nn.softmax(logits).numpy()
        predictions.extend(np.argmax(probs, axis=-1).tolist())
        probabilities.extend(probs.tolist())

    predictions = np.array(predictions)
    probabilities = np.array(probabilities)
    accuracy = np.mean(predictions == true_labels)
    return predictions, probabilities, accuracy


def main():
    print('Loading IMDB dataset...')
    ds = tfds.load('imdb_reviews', data_dir=DATA_DIR)
    train_ds = ds['train'].shuffle(25000, seed=SEED,
                                   reshuffle_each_iteration=False).take(TRAIN_SAMPLES)
    test_ds = ds['test'].take(TEST_SAMPLES)

    vectorize_layer = tf.keras.layers.TextVectorization(
        max_tokens=MAX_TOKENS,
        output_sequence_length=MAX_SEQ_LEN
    )
    train_text = train_ds.map(lambda x: x['text'])
    vectorize_layer.adapt(train_text)

    VOCAB_SIZE = len(vectorize_layer.get_vocabulary())
    EMBEDDING_SIZE = int(np.sqrt(VOCAB_SIZE))
    embedding_layer = tf.keras.layers.Embedding(VOCAB_SIZE, EMBEDDING_SIZE)
    cells = [tf.keras.layers.LSTMCell(64), tf.keras.layers.LSTMCell(32)]
    rnn = tf.keras.layers.RNN(cells)
    output_layer = tf.keras.layers.Dense(2)
    model = tf.keras.Sequential([
        vectorize_layer, embedding_layer, rnn, output_layer
    ])

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    train_batches = train_ds.map(
        lambda x: (x['text'], x['label'])
    ).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    print('Training the notebook-style LSTM...')
    model.fit(train_batches, epochs=EPOCHS)

    texts = []
    labels = []
    for item in tfds.as_numpy(test_ds):
        texts.append(item['text'].decode('utf-8'))
        labels.append(int(item['label']))
    labels = np.array(labels)

    variants = {'Original': texts}
    random.seed(SEED)
    variants['Adjacent swap'] = [swap_adjacent(text) for text in texts]
    random.seed(SEED)
    variants['Full shuffle'] = [shuffle_words(text) for text in texts]

    details = pd.DataFrame({'original_text': texts, 'true_label': labels})
    summary = []
    original_predictions = None
    original_probabilities = None

    for name, samples in variants.items():
        print('Evaluating:', name)
        predictions, probabilities, accuracy = evaluate(samples, labels, model)
        key = name.lower().replace(' ', '_')
        details[key + '_text'] = samples
        details[key + '_prediction'] = predictions
        details[key + '_negative_probability'] = probabilities[:, 0]
        details[key + '_positive_probability'] = probabilities[:, 1]

        if original_predictions is None:
            original_predictions = predictions
            original_probabilities = probabilities

        summary.append({
            'condition': name,
            'accuracy': accuracy,
            'mean_predicted_class_probability': np.mean(
                np.max(probabilities, axis=1)),
            'mean_change_in_predicted_class_probability': np.mean(
                np.max(probabilities, axis=1) -
                np.max(original_probabilities, axis=1)),
            'mean_absolute_change_in_positive_probability': np.mean(
                np.abs(probabilities[:, 1] - original_probabilities[:, 1])),
            'number_of_changed_predictions': np.sum(
                predictions != original_predictions)
        })

    results = pd.DataFrame(summary)
    print('\nPart 1 results:')
    print(results.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    results.to_csv('N06P1_summary.csv', index=False)
    details.to_csv('N06P1_predictions.csv', index=False)

    plt.figure(figsize=(7, 4))
    plt.bar(results['condition'], results['accuracy'])
    plt.ylim(0, 1)
    plt.ylabel('Accuracy')
    plt.title('IMDB Sentiment Accuracy vs. Word Order')
    plt.tight_layout()
    plt.savefig('N06P1_accuracy.png', dpi=180)
    plt.show()
    print('\nSaved N06P1_summary.csv, N06P1_predictions.csv, N06P1_accuracy.png')
if __name__ == '__main__':
    main()