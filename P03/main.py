# Project 3: IMDB sentiment analysis - main experiment script.
#
# Trains four LSTM + attention configurations (two architectures x two
# dropout settings), selects the best model using validation loss only,
# evaluates that one model once on the held-out test set, and saves all
# CSVs/figures needed for the report into results/.

import tensorflow as tf
import pandas as pd

import util
import model as model_lib

# The four required configurations: architecture x dropout rate.
CONFIGS = [
    {'name': 'Model 1', 'architecture': 'Single-layer LSTM + Attention',
     'dropout': 0.2, 'builder': model_lib.build_single_lstm},
    {'name': 'Model 2', 'architecture': 'Single-layer LSTM + Attention',
     'dropout': 0.5, 'builder': model_lib.build_single_lstm},
    {'name': 'Model 3', 'architecture': 'Stacked LSTM + Attention',
     'dropout': 0.2, 'builder': model_lib.build_stacked_lstm},
    {'name': 'Model 4', 'architecture': 'Stacked LSTM + Attention',
     'dropout': 0.5, 'builder': model_lib.build_stacked_lstm},
]


def main():
    util.set_seed(util.SEED)
    util.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    saved_models_dir = util.RESULTS_DIR / 'saved_models'
    saved_models_dir.mkdir(parents=True, exist_ok=True)

    # --- Load and preprocess data -----------------------------------------
    train_ds, validation_ds, test_ds = util.load_imdb_data()
    vectorizer = util.build_vectorizer(train_ds)
    train_batches, validation_batches, test_batches = util.make_datasets(
        train_ds, validation_ds, test_ds, vectorizer)

    trained_models = {}
    histories = {}
    comparison_rows = []
    val_confusion_matrices = {}

    # --- Train each of the four configurations -----------------------------
    for config in CONFIGS:
        name = config['architecture'] + f" (dropout={config['dropout']})"
        print(f'\n{"=" * 70}', flush=True)
        print(f'Training {config["name"]}: {name}', flush=True)
        print(f'{"=" * 70}', flush=True)

        keras_model = config['builder'](
            vocab_size=util.VOCAB_SIZE, seq_len=util.MAX_SEQ_LEN,
            embedding_dim=util.EMBEDDING_DIM, lstm_units=util.LSTM_UNITS,
            dropout_rate=config['dropout'], l2_reg=util.L2_REG)
        keras_model.summary()

        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=util.PATIENCE, restore_best_weights=True)

        history = keras_model.fit(
            train_batches, validation_data=validation_batches,
            epochs=util.MAX_EPOCHS, callbacks=[early_stop], verbose=2)

        epochs_trained = len(history.history['loss'])
        print(f'{config["name"]} stopped after {epochs_trained} epochs '
              f'(best weights restored).', flush=True)

        # Evaluate with the restored best weights on validation only; the
        # test set is never touched during model selection.
        val_loss, val_accuracy = keras_model.evaluate(validation_batches, verbose=0)
        y_val_true, y_val_prob = util.get_predictions(keras_model, validation_batches)
        val_metrics = util.evaluate_predictions(y_val_true, y_val_prob)

        print(f'{config["name"]} validation: loss={val_loss:.4f}, '
              f'accuracy={val_accuracy:.4f}, precision={val_metrics["precision"]:.4f}, '
              f'recall={val_metrics["recall"]:.4f}, f1={val_metrics["f1"]:.4f}, '
              f'95% CI=[{val_metrics["ci_lower"]:.4f}, {val_metrics["ci_upper"]:.4f}]',
              flush=True)

        trained_models[config['name']] = keras_model
        histories[config['name']] = history.history
        val_confusion_matrices[config['name']] = val_metrics['confusion_matrix']
        comparison_rows.append({
            'name': config['name'],
            'architecture': config['architecture'],
            'dropout': config['dropout'],
            'epochs_trained': epochs_trained,
            'val_loss': val_loss,
            'val_accuracy': val_accuracy,
            'val_precision': val_metrics['precision'],
            'val_recall': val_metrics['recall'],
            'val_f1': val_metrics['f1'],
            'val_ci_lower': val_metrics['ci_lower'],
            'val_ci_upper': val_metrics['ci_upper'],
            'val_n_samples': val_metrics['n_samples'],
        })

        model_path = saved_models_dir / f'{config["name"].replace(" ", "_").lower()}.keras'
        keras_model.save(model_path)

    comparison_df = pd.DataFrame(comparison_rows)
    util.save_csv(comparison_df, util.RESULTS_DIR / 'model_comparison.csv')

    # --- Select the best model using validation loss only -------------------
    best_row = comparison_df.loc[comparison_df['val_loss'].idxmin()]
    best_name = best_row['name']
    best_model = trained_models[best_name]
    print(f'\nSelected {best_name} as the best model '
          f'(lowest validation loss = {best_row["val_loss"]:.4f}).', flush=True)

    # --- Evaluate the selected model ONCE on the held-out test set ----------
    test_loss, test_accuracy = best_model.evaluate(test_batches, verbose=0)
    y_test_true, y_test_prob = util.get_predictions(best_model, test_batches)
    test_metrics = util.evaluate_predictions(y_test_true, y_test_prob)

    test_row = {
        'selected_model': best_name,
        'architecture': best_row['architecture'],
        'dropout': best_row['dropout'],
        'test_loss': test_loss,
        'test_accuracy': test_accuracy,
        'test_precision': test_metrics['precision'],
        'test_recall': test_metrics['recall'],
        'test_f1': test_metrics['f1'],
        'test_ci_lower': test_metrics['ci_lower'],
        'test_ci_upper': test_metrics['ci_upper'],
        'test_n_samples': test_metrics['n_samples'],
    }
    util.save_csv([test_row], util.RESULTS_DIR / 'test_results.csv')

    # --- Print summary tables ------------------------------------------------
    print('\n' + '=' * 70, flush=True)
    print('Model comparison summary (validation set):', flush=True)
    print(comparison_df.to_string(index=False, float_format=lambda v: f'{v:.4f}'), flush=True)
    print('\nFinal selected model performance on the test set:', flush=True)
    print(pd.DataFrame([test_row]).to_string(index=False, float_format=lambda v: f'{v:.4f}'), flush=True)

    # --- Save figures ----------------------------------------------------------
    print('\nGenerating plots...', flush=True)
    util.plot_training_curves(histories, util.RESULTS_DIR / 'training_curves.png')
    util.plot_model_comparison_bar(comparison_df, util.RESULTS_DIR / 'model_comparison.png')
    util.plot_confusion_matrices(val_confusion_matrices,
                                  util.RESULTS_DIR / 'validation_confusion_matrices.png', ncols=2)
    util.plot_confusion_matrices({best_name: test_metrics['confusion_matrix']},
                                  util.RESULTS_DIR / 'test_confusion_matrix.png', ncols=1)

    print(f'\nAll results saved to {util.RESULTS_DIR}', flush=True)


if __name__ == '__main__':
    main()
