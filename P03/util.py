# Project 3: IMDB sentiment analysis - shared helper functions.
#
# This module handles everything that is not model architecture:
# loading data through TensorFlow Datasets, building/adapting the
# TextVectorization layer, turning text into batched integer-sequence
# datasets, computing evaluation metrics (including Wilson confidence
# intervals), and saving plots/CSVs for the report.

import random
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

# ---------------------------------------------------------------------------
# Constants / common hyperparameters shared by every model in this project.
# ---------------------------------------------------------------------------
SEED = 479
# Use a dataset directory outside OneDrive to avoid Windows permission issues.
DATA_DIR = str(Path.home() / 'tensorflow-datasets')
RESULTS_DIR = Path(__file__).resolve().parent / 'results'

ORIGINAL_TRAIN_SIZE = 25000
TRAIN_SPLIT_SIZE = 20000       # 80% of the original 25,000 training reviews
VALIDATION_SPLIT_SIZE = 5000   # 20% of the original 25,000 training reviews

VOCAB_SIZE = 10000
MAX_SEQ_LEN = 200
EMBEDDING_DIM = 64
LSTM_UNITS = 64
BATCH_SIZE = 32
MAX_EPOCHS = 30
PATIENCE = 5
L2_REG = 1e-4


def set_seed(seed=SEED):
    """Fix every relevant random seed so the experiment is reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def load_imdb_data(data_dir=DATA_DIR, seed=SEED):
    """Load IMDB reviews with tensorflow_datasets and split the 25,000
    training reviews into 80% train / 20% validation. The 25,000 test
    reviews are returned untouched and must stay held out until the end.
    """
    print('Loading IMDB reviews from TensorFlow Datasets...', flush=True)
    (full_train, test_ds), info = tfds.load(
        'imdb_reviews', split=['train', 'test'], data_dir=data_dir,
        as_supervised=True, with_info=True)

    # Shuffle once with a fixed seed so the train/validation split is
    # reproducible and does not change between runs.
    full_train = full_train.shuffle(ORIGINAL_TRAIN_SIZE, seed=seed,
                                     reshuffle_each_iteration=False)
    train_ds = full_train.take(TRAIN_SPLIT_SIZE)
    validation_ds = full_train.skip(TRAIN_SPLIT_SIZE).take(VALIDATION_SPLIT_SIZE)

    print(f'Train reviews: {TRAIN_SPLIT_SIZE}, '
          f'Validation reviews: {VALIDATION_SPLIT_SIZE}, '
          f'Test reviews: {info.splits["test"].num_examples}', flush=True)
    return train_ds, validation_ds, test_ds


def build_vectorizer(train_ds, vocab_size=VOCAB_SIZE, seq_len=MAX_SEQ_LEN):
    """Create and adapt a TextVectorization layer using ONLY training
    reviews. Validation and test reviews must never be used here.
    """
    vectorizer = tf.keras.layers.TextVectorization(
        max_tokens=vocab_size, output_sequence_length=seq_len)
    train_text = train_ds.map(lambda text, label: text)
    print('Adapting TextVectorization on training reviews only...', flush=True)
    vectorizer.adapt(train_text)
    return vectorizer


def make_datasets(train_ds, validation_ds, test_ds, vectorizer, batch_size=BATCH_SIZE):
    """Convert raw (text, label) datasets into fixed-length integer-sequence
    datasets. Training is shuffled and batched; validation/test are batched
    only (no shuffling), keeping all three splits completely separate.
    """
    def vectorize(text, label):
        return vectorizer(text), label

    train_batches = (train_ds
                      .map(vectorize, num_parallel_calls=tf.data.AUTOTUNE)
                      .shuffle(TRAIN_SPLIT_SIZE, seed=SEED)
                      .batch(batch_size)
                      .prefetch(tf.data.AUTOTUNE))
    validation_batches = (validation_ds
                           .map(vectorize, num_parallel_calls=tf.data.AUTOTUNE)
                           .batch(batch_size)
                           .prefetch(tf.data.AUTOTUNE))
    test_batches = (test_ds
                     .map(vectorize, num_parallel_calls=tf.data.AUTOTUNE)
                     .batch(batch_size)
                     .prefetch(tf.data.AUTOTUNE))
    return train_batches, validation_batches, test_batches


def get_predictions(model, dataset):
    """Run the model over a batched dataset and return true labels and
    predicted positive-class probabilities as flat NumPy arrays.
    """
    y_true = []
    y_prob = []
    for x, y in dataset:
        probs = model.predict(x, verbose=0)
        y_true.append(y.numpy())
        y_prob.append(probs.reshape(-1))
    return np.concatenate(y_true), np.concatenate(y_prob)


def wilson_confidence_interval(successes, n, confidence=0.95):
    """95% Wilson score confidence interval for a binary accuracy proportion."""
    if n == 0:
        return 0.0, 0.0
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    phat = successes / n
    denom = 1 + (z ** 2) / n
    center = phat + (z ** 2) / (2 * n)
    margin = z * np.sqrt((phat * (1 - phat) / n) + (z ** 2) / (4 * n ** 2))
    lower = (center - margin) / denom
    upper = (center + margin) / denom
    return max(0.0, float(lower)), min(1.0, float(upper))


def evaluate_predictions(y_true, y_prob, threshold=0.5):
    """Compute accuracy, precision, recall, F1, a 95% Wilson CI on accuracy,
    and a row-normalized confusion matrix for a set of predictions.
    """
    y_pred = (y_prob >= threshold).astype(int)
    n = len(y_true)
    correct = int(np.sum(y_pred == y_true))
    accuracy = correct / n
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    ci_lower, ci_upper = wilson_confidence_interval(correct, n)
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'n_samples': n,
        'confusion_matrix': cm,
    }


def save_csv(rows, path):
    """Save a list of dict rows (or an existing DataFrame) to a CSV file."""
    df = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return df


def plot_training_curves(histories, out_path):
    """Plot training/validation accuracy and loss curves for every model."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for name, history in histories.items():
        epochs = range(1, len(history['loss']) + 1)
        axes[0].plot(epochs, history['accuracy'], marker='o', label=f'{name} (train)')
        axes[0].plot(epochs, history['val_accuracy'], marker='s', linestyle='--', label=f'{name} (val)')
        axes[1].plot(epochs, history['loss'], marker='o', label=f'{name} (train)')
        axes[1].plot(epochs, history['val_loss'], marker='s', linestyle='--', label=f'{name} (val)')

    axes[0].set_title('Training and Validation Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend(fontsize=7)

    axes[1].set_title('Training and Validation Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_model_comparison_bar(df, out_path):
    """Bar chart comparing validation accuracy (with 95% CI error bars)."""
    lower_err = df['val_accuracy'] - df['val_ci_lower']
    upper_err = df['val_ci_upper'] - df['val_accuracy']

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(df['name'], df['val_accuracy'], yerr=[lower_err, upper_err],
           capsize=6, color='steelblue')
    ax.set_ylabel('Validation Accuracy')
    ax.set_title('Model Comparison: Validation Accuracy with 95% CI')
    ax.set_ylim(0, 1)
    plt.xticks(rotation=15, ha='right')
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_confusion_matrices(matrices, out_path, ncols=2):
    """Plot a grid of row-normalized confusion matrices, one per model.
    `matrices` maps a model name to a 2x2 normalized confusion matrix.
    """
    names = list(matrices.keys())
    nrows = int(np.ceil(len(names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows), squeeze=False)
    axes = axes.reshape(-1)
    labels = ['Negative (0)', 'Positive (1)']

    for ax, name in zip(axes, names):
        cm = matrices[name]
        im = ax.imshow(cm, cmap='Blues', vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f'{cm[i, j]:.2f}', ha='center', va='center',
                        color='white' if cm[i, j] > 0.5 else 'black')
        ax.set_xticks([0, 1])
        ax.set_xticklabels(labels)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(labels)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title(name)
        fig.colorbar(im, ax=ax, fraction=0.046)

    for ax in axes[len(names):]:
        ax.axis('off')

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
