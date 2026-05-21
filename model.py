import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, Dense, Dropout, GRU
from tensorflow.keras.optimizers import Adam



def create_model(
    vocab_size,
    sequence_length=15,
    embedding_dim=100,
    rnn_units=64,
    dense_units=64,
    dropout_rate=0.5,
    learning_rate = 0.0005
):
    model = Sequential()

    model.add(Input(shape=(sequence_length,)))

    model.add(
        Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim
        )
    )

    model.add(
        Bidirectional(
            LSTM(
                rnn_units,
                return_sequences=False,
                dropout=0.1,
                recurrent_dropout=0.1
            )
        )
    )

    model.add(Dense(dense_units, activation="relu"))
    model.add(Dropout(dropout_rate))
    model.add(Dense(vocab_size, activation="softmax"))

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top_3_accuracy"), tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top_5_accuracy")],
    )

    return model