import numpy as np
import tensorflow as tf
from gensim.models import KeyedVectors
from keras import Model, layers, models, backend, initializers
from keras.optimizers import Adam
from optuna.integration import TFKerasPruningCallback
from sklearn.model_selection import train_test_split
from tensorflow import keras


class ModelHandler:
    def __init__(self, embedding_layer) -> None:
        self.embedding_layer = embedding_layer
        self.lstm_first_layer_size = None
        self.lstm_second_layer_size = None
        self.lstm_third_layer_size = None
        self.learning_rate: float

    def new_trail(self, trail):
        self.lstm_first_layer_size = trail.suggest_int("lstm_first_layer_size", 16, 256)
        self.lstm_second_layer_size = trail.suggest_int(
            "lstm_second_layer_size", 16, 256
        )
        self.lstm_third_layer_size = trail.suggest_int("lstm_third_layer_size", 16, 256)
        self.learning_rate = trail.suggest_float("learning_rate", 1e-5, 1e-1, log=True)

    @staticmethod
    def split_dataset(x, y, test_size=0.15):
        return train_test_split(x, y, test_size=test_size, random_state=42)

    def create_basic_model(self):
        model = models.Sequential(
            [
                self.embedding_layer,
                layers.LSTM(self.lstm_first_layer_size),
                layers.Dense(1, activation="sigmoid"),
            ]
        )
        return self.compile_model(model)

    def create_in_series_model(self) -> Model:
        model = models.Sequential(
            [
                self.embedding_layer,
                layers.LSTM(self.lstm_first_layer_size, return_sequences=True),
                layers.LSTM(self.lstm_second_layer_size, return_sequences=True),
                layers.LSTM(self.lstm_third_layer_size),
                layers.Dense(1, activation="sigmoid"),
            ]
        )
        return self.compile_model(model)

    def create_cnn_lstm_model(self) -> Model:
        model = models.Sequential(
            [
                self.embedding_layer,
                layers.Lambda(lambda x: tf.expand_dims(x, 1)),
                layers.Conv2D(100, (2, 2), activation="relu", padding="same"),
                layers.MaxPooling2D(pool_size=1),
                layers.Flatten(),
                layers.Reshape((-1, 100)),
                layers.LSTM(self.lstm_first_layer_size),
                layers.Dense(1, activation="sigmoid"),
            ]
        )
        return self.compile_model(model)

    def compile_model(self, model) -> Model:
        return model.compile(
            loss="binary_crossentropy",
            optimizer=Adam(learning_rate=self.learning_rate),
            metrics=["acc"],
        )

    def create_model(self, which) -> Model:
        if which == "basic":
            return self.create_basic_model()
        elif which == "in_series":
            return self.create_in_series_model()
        elif which == "cnn_lstm":
            return self.create_cnn_lstm_model()
        else:
            raise NotImplementedError

    def objective(self, trail, which, x_train, y_train, x_valid, y_valid):
        # Clear clutter from previous session graphs.
        backend.clear_session()
        # Generate our trial model.
        self.new_trail(trail)
        model = self.create_model(which)
        # Fit the model on the training data.
        # The KerasPruningCallback checks for pruning condition every epoch.
        model.fit(
            x_train,
            y_train,
            batch_size=BATCH_SIZE,
            callbacks=[TFKerasPruningCallback(trial, "val_acc")],
            epochs=EPOCHS,
            validation_data=(x_valid, y_valid),
            verbose=1,
        )

        # Evaluate the model accuracy on the validation set.
        score = model.evaluate(x_valid, y_valid, verbose=0)
        return score[1]


def create_embedding_layer(voc, shape, model):
    word_index = dict(zip(voc, range(len(voc))))
    num_tokens = len(voc) + 2
    embedding_dim = shape[1]  # dimension of vectors
    hits = 0
    misses = 0

    embedding_matrix = np.zeros((num_tokens, embedding_dim))
    for word, i in word_index.items():
        try:
            embedding_vector = model.get_vector(word)
            embedding_matrix[i] = embedding_vector
            hits += 1
        except KeyError:
            misses += 1

    embedding_layer = layers.Embedding(
        num_tokens,
        embedding_dim,
        embeddings_initializer=initializers.Constant(embedding_matrix),
        trainable=False,
        input_shape=[None],
        mask_zero=True,
    )
    return embedding_layer


def load_model(models, file_name):
    file = models[file_name]
    print(file.name)
    return KeyedVectors.load_word2vec_format(file, binary=False)
