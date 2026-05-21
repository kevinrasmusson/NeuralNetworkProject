import os
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
import yaml

from dataset import load_data
from model import create_model

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau


if __name__ == "__main__":
    # -----------------------------
    # Settings / hyperparameters
    # -----------------------------
    DATA_PATH = "text-ds.csv"

    SAMPLE_SIZE = 5000
    MAX_WORDS = 10000
    SEQUENCE_LENGTH = 15

    EMBEDDING_DIM = 128 # Size of each word vector
    RNN_UNITS = 128 # Number of units in the RNN layer
    DENSE_UNITS = 128 # Number of units in the Dense layer before the output layer
    DROPOUT_RATE = 0.3

    EPOCHS = 20
    BATCH_SIZE = 64
    RANDOM_STATE = 42

    # -----------------------------
    # Load and sample data
    # -----------------------------
    df = load_data(DATA_PATH)

    # Random sample makes training faster and more representative than df.head(...)
    if SAMPLE_SIZE is not None and SAMPLE_SIZE < len(df):
        df = df.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE)

    print("Dataset shape used for training:", df.shape)

    # -----------------------------
    # Tokenize text
    # -----------------------------
    tokenizer = Tokenizer(
        num_words=MAX_WORDS,
        oov_token="<UNK>" # Rare words
    )

    tokenizer.fit_on_texts(df["text_clean"])
    sequences = tokenizer.texts_to_sequences(df["text_clean"])

    print("Sample tokenized sequence:", sequences[0][:50])

    # -----------------------------
    # Create input/output sequences
    # X = previous SEQUENCE_LENGTH words
    # y = next word
    # -----------------------------
    X = []
    y = []

    for seq in sequences:
        for i in range(SEQUENCE_LENGTH, len(seq)):
            target_word = seq[i]

            # Skip examples where target is <UNK>
            if target_word == 1:
                continue

            X.append(seq[i - SEQUENCE_LENGTH:i])
            y.append(target_word)

    X = np.array(X)
    y = np.array(y)

    print("X shape:", X.shape)
    print("y shape:", y.shape)

    if len(X) == 0:
        raise ValueError("No training sequences were created. Try lowering SEQUENCE_LENGTH or checking the dataset.")

    print("Sample input sequence:", X[0])
    print("Sample target:", y[0])

    # -----------------------------
    # Shuffle and split data
    # -----------------------------
    np.random.seed(RANDOM_STATE)
    indices = np.arange(len(X))
    np.random.shuffle(indices)

    X = X[indices]
    y = y[indices]

    split_index = int(0.8 * len(X))

    X_train, X_val = X[:split_index], X[split_index:]
    y_train, y_val = y[:split_index], y[split_index:]

    print("Training examples:", len(X_train))
    print("Validation examples:", len(X_val))

    # -----------------------------
    # Create model
    # -----------------------------
    model = create_model(
        vocab_size=MAX_WORDS,
        sequence_length=SEQUENCE_LENGTH,
        embedding_dim=EMBEDDING_DIM,
        rnn_units=RNN_UNITS,
        dense_units=DENSE_UNITS,
        dropout_rate=DROPOUT_RATE
    )

    model.summary()

    # -----------------------------
    # Create experiment folder
    # -----------------------------
    experiment_name = (
        f"bilstm_seq{SEQUENCE_LENGTH}_"
        f"vocab{MAX_WORDS}_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    experiment_dir = os.path.join("checkpoints", experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)

    # -----------------------------
    # Save hyperparameters
    # -----------------------------
    hyperparameters = {
        "data_path": DATA_PATH,
        "sample_size": SAMPLE_SIZE,
        "max_words": MAX_WORDS,
        "sequence_length": SEQUENCE_LENGTH,
        "embedding_dim": EMBEDDING_DIM,
        "rnn_units": RNN_UNITS,
        "dense_units": DENSE_UNITS,
        "dropout_rate": DROPOUT_RATE,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "random_state": RANDOM_STATE,
        "model_type": "Embedding + Bidirectional LSTM + Dense + Dropout + Softmax",
        "loss": "sparse_categorical_crossentropy",
        "checkpoint_monitor": "val_top_5_accuracy"
    }

    with open(os.path.join(experiment_dir, "hyperparameters.yaml"), "w", encoding="utf-8") as f:
        yaml.dump(hyperparameters, f, allow_unicode=True, sort_keys=False)

    # -----------------------------
    # Save tokenizer
    # -----------------------------
    with open(os.path.join(experiment_dir, "tokenizer.pkl"), "wb") as f:
        pickle.dump(tokenizer, f)

    # -----------------------------
    # Callbacks
    # -----------------------------
    early_stopping = EarlyStopping(
        monitor="val_top_5_accuracy",
        patience=3,
        mode="max",
        restore_best_weights=True,
        verbose=1
    )

    checkpoint = ModelCheckpoint(
        filepath=os.path.join(experiment_dir, "best_model.keras"),
        monitor="val_top_5_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1
    )
    reduce_lr = ReduceLROnPlateau(
    monitor="val_top_5_accuracy",
    factor=0.5,
    patience=2,
    mode="max",
    min_lr=0.00001,
    verbose=1
    )

    # -----------------------------
    # Train model
    # -----------------------------
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stopping, checkpoint, reduce_lr]
    )

    # -----------------------------
    # Save training history
    # -----------------------------
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(os.path.join(experiment_dir, "training_history.csv"), index=False)

    # -----------------------------
    # Save results
    # -----------------------------
    best_epoch_index = history_df["val_top_5_accuracy"].idxmax()

    results = {
        "best_epoch": int(best_epoch_index + 1),
        "final_epoch": int(len(history_df)),

        "best_val_accuracy": float(history_df.loc[best_epoch_index, "val_accuracy"]),
        "best_val_top_3_accuracy": float(history_df.loc[best_epoch_index, "val_top_3_accuracy"]),
        "best_val_top_5_accuracy": float(history_df.loc[best_epoch_index, "val_top_5_accuracy"]),
        "best_val_loss": float(history_df.loc[best_epoch_index, "val_loss"]),

        "final_train_accuracy": float(history_df["accuracy"].iloc[-1]),
        "final_train_top_3_accuracy": float(history_df["top_3_accuracy"].iloc[-1]),
        "final_train_top_5_accuracy": float(history_df["top_5_accuracy"].iloc[-1]),
        "final_train_loss": float(history_df["loss"].iloc[-1]),

        "num_total_sequences": int(len(X)),
        "num_training_examples": int(len(X_train)),
        "num_validation_examples": int(len(X_val)),

        "experiment_dir": experiment_dir
    }

    with open(os.path.join(experiment_dir, "results.yaml"), "w", encoding="utf-8") as f:
        yaml.dump(results, f, allow_unicode=True, sort_keys=False)

    print("\nTraining complete.")
    print(f"Saved experiment to: {experiment_dir}")
    print(f"Best validation top-5 accuracy: {results['best_val_top_5_accuracy']:.4f}")
    print(f"Best validation accuracy: {results['best_val_accuracy']:.4f}")