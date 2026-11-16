# Hackathon 6 Part 2
from pathlib import Path
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds

SEED = 479
DATA_DIR = str(Path.home() / 'tensorflow-datasets')
TRAIN_SAMPLES = 12000
BATCH_SIZE = 32
EPOCHS = 3
MAX_SEQ_LEN = 128
MAX_TOKENS = 5000
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


def perturb(text, method, rng):
    words = text.split()
    if len(words) < 2:
        return text
    if method == 'Adjacent swap':
        i = rng.randrange(len(words) - 1)
        words[i], words[i + 1] = words[i + 1], words[i]
    elif method == 'Reverse':
        words.reverse()
    elif method == 'Full shuffle':
        rng.shuffle(words)
    return ' '.join(words)


def make_model(vectorize_layer):
    text_input = tf.keras.Input(shape=(), dtype=tf.string)
    tokens = vectorize_layer(text_input)
    vocab_size = len(vectorize_layer.get_vocabulary())
    embedding_layer = tf.keras.layers.Embedding(vocab_size, int(np.sqrt(vocab_size)))
    embeddings = embedding_layer(tokens)
    query_layer = tf.keras.layers.Conv1D(filters=100, kernel_size=4, padding='same')
    value_layer = tf.keras.layers.Conv1D(filters=100, kernel_size=4, padding='same')
    query = query_layer(embeddings)
    value = value_layer(embeddings)
    attention = tf.keras.layers.Attention()
    attended_values, attention_scores = attention([query, value], return_attention_scores=True)
    concat = tf.keras.layers.Concatenate()
    attended = concat([query, attended_values])
    cells = [tf.keras.layers.LSTMCell(64), tf.keras.layers.LSTMCell(32)]
    rnn = tf.keras.layers.RNN(cells)
    logits = tf.keras.layers.Dense(2)(rnn(attended))
    model = tf.keras.Model(text_input, logits)
    attention_model = tf.keras.Model(text_input, attention_scores)
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model, attention_model, vectorize_layer


def infer(texts, model):
    logits = model(tf.constant(texts), training=False)
    probs = tf.nn.softmax(logits, axis=-1).numpy()
    labels = tf.math.argmax(probs, axis=-1).numpy()
    return labels, probs


def main():
    print('Loading IMDB data...', flush=True)
    ds = tfds.load('imdb_reviews', data_dir=DATA_DIR)
    train_ds = ds['train'].shuffle(25000, seed=SEED,
                                   reshuffle_each_iteration=False).take(TRAIN_SAMPLES)
    vectorize_layer = tf.keras.layers.TextVectorization(
        max_tokens=MAX_TOKENS, output_sequence_length=MAX_SEQ_LEN)
    vectorize_layer.adapt(train_ds.map(lambda x: x['text']))
    model, attention_model, vectorize_layer = make_model(vectorize_layer)
    train_batches = train_ds.map(lambda x: (x['text'], x['label'])).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    print('Training TensorFlow LSTM with attention...', flush=True)
    model.fit(train_batches, epochs=EPOCHS)

    candidates = []
    for item in tfds.as_numpy(ds['test']):
        text = item['text'].decode('utf-8')
        word_count = len(text.split())
        if 15 <= word_count <= 65:
            candidates.append((text, int(item['label'])))
        if len(candidates) >= 10:
            break
    if len(candidates) < 10:
        raise RuntimeError('Not enough short IMDB reviews found.')

    texts = [t for t, _ in candidates]
    truths = [y for _, y in candidates]
    methods = ['Adjacent swap', 'Reverse', 'Full shuffle']
    rows = []
    changed_example = None
    for i, (text, truth) in enumerate(zip(texts, truths), 1):
        original_label, original_probs = infer([text], model)
        original_label = int(original_label[0])
        original_prob = float(original_probs[0, original_label])
        rows.append({'example': i, 'condition': 'Original', 'text': text,
                     'true_label': truth, 'prediction': original_label,
                     'predicted_probability': original_prob,
                     'positive_probability': float(original_probs[0, 1]),
                     'changed_prediction': False})
        for method in methods:
            changed_text = perturb(text, method, random.Random(SEED + i * 17 + methods.index(method)))
            label, probs = infer([changed_text], model)
            label = int(label[0])
            flipped = label != original_label
            rows.append({'example': i, 'condition': method, 'text': changed_text,
                         'true_label': truth, 'prediction': label,
                         'predicted_probability': float(probs[0, label]),
                         'positive_probability': float(probs[0, 1]),
                         'changed_prediction': flipped})
            if flipped and changed_example is None:
                changed_example = (i, text, changed_text, method)

    results = pd.DataFrame(rows)
    results.to_csv('N06P2_examples.csv', index=False)
    print('\nPart 2 results (0=negative, 1=positive):', flush=True)
    print(results[['example', 'condition', 'prediction', 'predicted_probability',
                   'positive_probability', 'changed_prediction']].to_string(index=False))
    print('\nChanged predictions:', int(results['changed_prediction'].sum()), flush=True)

    if changed_example is None:
        print('\nNo prediction changed among the 10 examples. For the required attention'
              ' comparison, select other short reviews or stronger perturbations and rerun.')
        return

    i, original, modified, method = changed_example
    import seaborn as sns
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, text, label in zip(axes, [original, modified], ['Original', method]):
        scores = attention_model(tf.constant([text]), training=False).numpy()[0]
        token_ids = vectorize_layer(tf.constant([text])).numpy()[0]
        valid = min(int(np.count_nonzero(token_ids)), 25)
        vocab = vectorize_layer.get_vocabulary()
        tick_labels = [vocab[int(t)] for t in token_ids[:valid]]
        sns.heatmap(scores[:valid, :valid], xticklabels=tick_labels,
                    yticklabels=tick_labels, cmap='plasma', ax=ax)
        ax.set_title(f'Example {i}: {label}')
        ax.set_xlabel('Key tokens')
        ax.set_ylabel('Query tokens')
    fig.tight_layout()
    fig.savefig('N06P2_attention.png', dpi=180)
    plt.show()
    print('\nSaved N06P2_examples.csv and N06P2_attention.png')

if __name__ == '__main__':
    main()
