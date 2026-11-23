# Project 3: IMDB sentiment analysis - model architectures.
#
# Both architectures use the Functional API (needed because attention takes
# the LSTM sequence output as both query and value). Padding tokens (id 0
# from TextVectorization) are masked via Embedding(mask_zero=True), and that
# mask propagates automatically through the LSTM layers, the Attention
# layer, and the mask-aware GlobalAveragePooling1D used to pool attention
# outputs into a single vector per review.

import tensorflow as tf
from tensorflow.keras import layers, regularizers, Model


def _attention_classifier_head(sequence_output, dropout_rate, l2_reg):
    """Shared tail: self-attention over LSTM outputs, masked pooling,
    dropout, and a regularized sigmoid classification layer.
    """
    # Self-attention over the LSTM sequence output. The padding mask
    # produced by the Embedding layer propagates through here automatically,
    # so padded tokens cannot be attended to and do not distort the scores.
    attention_out = layers.Attention(name='attention')([sequence_output, sequence_output])

    # Mask-aware pooling: GlobalAveragePooling1D ignores masked (padded)
    # timesteps when it receives a propagated mask.
    pooled = layers.GlobalAveragePooling1D(name='masked_average_pool')(attention_out)

    pooled = layers.Dropout(dropout_rate, name='dropout')(pooled)
    outputs = layers.Dense(
        1, activation='sigmoid',
        kernel_regularizer=regularizers.l2(l2_reg),
        name='classifier')(pooled)
    return outputs


def build_single_lstm(vocab_size=10000, seq_len=200, embedding_dim=64,
                       lstm_units=64, dropout_rate=0.2, l2_reg=1e-4):
    """Embedding -> single LSTM -> attention -> dropout -> classifier."""
    inputs = layers.Input(shape=(seq_len,), dtype='int64', name='token_ids')
    x = layers.Embedding(vocab_size, embedding_dim, mask_zero=True, name='embedding')(inputs)
    lstm_out = layers.LSTM(
        lstm_units, return_sequences=True,
        kernel_regularizer=regularizers.l2(l2_reg),
        name='lstm')(x)

    outputs = _attention_classifier_head(lstm_out, dropout_rate, l2_reg)

    model = Model(inputs, outputs, name='single_lstm_attention')
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=['accuracy'])
    return model


def build_stacked_lstm(vocab_size=10000, seq_len=200, embedding_dim=64,
                        lstm_units=64, dropout_rate=0.2, l2_reg=1e-4):
    """Embedding -> two stacked LSTMs -> attention -> dropout -> classifier."""
    inputs = layers.Input(shape=(seq_len,), dtype='int64', name='token_ids')
    x = layers.Embedding(vocab_size, embedding_dim, mask_zero=True, name='embedding')(inputs)
    x = layers.LSTM(
        lstm_units, return_sequences=True,
        kernel_regularizer=regularizers.l2(l2_reg),
        name='lstm_1')(x)
    lstm_out = layers.LSTM(
        lstm_units, return_sequences=True,
        kernel_regularizer=regularizers.l2(l2_reg),
        name='lstm_2')(x)

    outputs = _attention_classifier_head(lstm_out, dropout_rate, l2_reg)

    model = Model(inputs, outputs, name='stacked_lstm_attention')
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=['accuracy'])
    return model
