from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam


def create_model(
    vocab_size: int,
    sequence_length: int = 5,
    embedding_dim: int = 100,
    lstm_units: int = 64,
    dense_units: int = 64,
    dropout_rate: float = 0.5,
    learning_rate: float = 0.001
):
    model = Sequential()

    # Turns word IDs into dense vector representations
    model.add(
        Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            input_length=sequence_length
        )
    )

    # Reads the sequence both forwards and backwards
    model.add(
        Bidirectional(
            LSTM(
                lstm_units,
                return_sequences=False,
                dropout=0.1,
                recurrent_dropout=0.1
            )
        )
    )

    # Fully connected layer
    model.add(
        Dense(
            dense_units,
            activation="relu"
        )
    )

    # Regularization to reduce overfitting
    model.add(
        Dropout(dropout_rate)
    )

    # Output layer: one probability per word in vocabulary
    model.add(
        Dense(
            vocab_size,
            activation="softmax"
        )
    )

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model