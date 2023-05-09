import numpy as np
import tensorflow as tf
from gensim.models import KeyedVectors
from keras import Model, backend, initializers, layers, models
from keras.optimizers import Adam
from optuna.integration import TFKerasPruningCallback
from sklearn.model_selection import train_test_split
from tensorflow import keras


class ModelHandler:
    def __init__(self, embedding_layer, batch_size, epochs) -> None:
        self.embedding_layer = embedding_layer
        self.batch_size = batch_size
        self.epochs = epochs
        self.first_layer_size: int
        self.second_layer_size: int
        self.third_layer_size: int
        self.learning_rate: float
        self.x_valid = None
        self.y_valid = None
        self.x_train = None
        self.y_train = None

    def new_trial(self, trial):
        self.first_layer_size = trial.suggest_int("lstm_first_layer_size", 16, 256)
        self.second_layer_size = trial.suggest_int("lstm_second_layer_size", 16, 256)
        self.third_layer_size = trial.suggest_int("lstm_third_layer_size", 16, 256)
        self.learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)

    def set_split_dataset(self, x, y, test_size=0.15):
        self.x_valid, self.x_train, self.y_valid, self.y_train = train_test_split(
            x, y, test_size=test_size, random_state=42
        )

    def create_basic_model(self):
        model = models.Sequential(
            [
                self.embedding_layer,
                layers.LSTM(self.first_layer_size),
                layers.Dense(1, activation="sigmoid"),
            ]
        )
        return self.compile_model(model)

    def create_in_series_model(self) -> Model:
        model = models.Sequential(
            [
                self.embedding_layer,
                layers.LSTM(self.first_layer_size, return_sequences=True),
                layers.LSTM(self.second_layer_size, return_sequences=True),
                layers.LSTM(self.third_layer_size),
                layers.Dense(1, activation="sigmoid"),
            ]
        )
        return self.compile_model(model)

    def create_cnn_lstm_model(self) -> Model:
        model = models.Sequential(
            [
                self.embedding_layer,
                layers.Bidirectional(
                    layers.LSTM(
                        self.first_layer_size,
                        return_sequences=True,
                    ),
                ),
                layers.Conv1D(
                    self.second_layer_size,
                    kernel_size=3,
                    activation="relu",
                    padding="same",
                ),
                layers.GlobalMaxPooling1D(),
                layers.Dense(1, activation="sigmoid"),
            ]
        )
        return self.compile_model(model)

    def make_bidirectional_lstm_cnn_nlp_keras_model(self):
        words = layers.Input(shape=(None,))
        x = self.embedding_layer(words)
        x = layers.SpatialDropout1D(0.3)(x)
        x = layers.Bidirectional(layers.LSTM(self.first_layer_size, return_sequences=True))(x)
        x = layers.Bidirectional(layers.LSTM(self.second_layer_size, return_sequences=True))(x)

        hidden = layers.concatenate(
            [
                layers.GlobalMaxPooling1D()(x),
                layers.GlobalAveragePooling1D()(x),
            ]
        )
        # hidden = layers.add([hidden, layers.Dense(self.third_layer_size, activation="relu")(hidden)])
        # hidden = layers.add([hidden, layers.Dense(DENSE_HIDDEN_UNITS, activation="relu")(hidden)])
        result = layers.Dense(1, activation="sigmoid")(hidden)
        aux_result = layers.Dense(1, activation="sigmoid")(hidden)

        model = Model(inputs=words, outputs=[result, aux_result])
        model.compile(loss="binary_crossentropy", optimizer="adam")

        return model

    def compile_model(self, model) -> Model:
        model.compile(
            loss="binary_crossentropy",
            optimizer=Adam(learning_rate=self.learning_rate),
            metrics=["acc"],
        )
        return model

    def create_model(self, which) -> Model:
        if which == "basic":
            return self.create_basic_model()
        elif which == "in_series":
            return self.create_in_series_model()
        elif which == "cnn_lstm":
            return self.create_cnn_lstm_model()
        else:
            raise NotImplementedError

    def objective(self, trial, which):
        # Clear clutter from previous session graphs.
        backend.clear_session()
        # Generate our trial model.
        self.new_trial(trial)
        model = self.create_model(which)
        # Fit the model on the training data.
        # The KerasPruningCallback checks for pruning condition every epoch.
        model.fit(
            self.x_train,
            self.y_train,
            batch_size=self.batch_size,
            callbacks=[TFKerasPruningCallback(trial, "val_acc")],
            epochs=self.epochs,
            validation_data=(self.x_valid, self.y_valid),
            verbose="1",
        )

        # Evaluate the model accuracy on the validation set.
        score = model.evaluate(self.x_valid, self.y_valid, verbose="0")
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


def load_model(file):
    print(file.name)
    return KeyedVectors.load_word2vec_format(file, binary=False)
