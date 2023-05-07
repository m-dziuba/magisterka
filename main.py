from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
import tensorflow as tf
from gensim.models import KeyedVectors
from keras.layers import TextVectorization
from optuna.trial import TrialState
from sklearn.preprocessing import LabelBinarizer
from tensorflow import keras

from models import ModelHandler, create_embedding_layer, load_model
from readers import generate_datasets, get_finished_embedding_models, get_embedding_models_paths


def main(which, x, y, word2vec_model, n_trails=20):
    shape = word2vec_model.vectors.shape
    vectorizer = TextVectorization(
        max_tokens=shape[0], output_sequence_length=int(x.str.split().str.len().max())
    )
    vectorizer.adapt(x)
    # dict mapping words to their indices
    voc = vectorizer.get_vocabulary()

    # create embedding layer
    vectorized_x = vectorizer(np.array([[s] for s in x])).numpy()
    embedding_layer = create_embedding_layer(voc, shape, word2vec_model)

    nn_model = ModelHandler(
        embedding_layer=embedding_layer,
        batch_size=256,
        epochs=100,
    )
    # create model

    # test train split
    nn_model.set_split_dataset(vectorized_x, y)

    func = lambda trail: nn_model.objective(trail, which)
    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(),
        storage="sqlite:///db.sqlite3",
    )
    study.optimize(func, n_trials=n_trails)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    best_trail = study.best_trial

    print("  Value: ", best_trail.value)
    return study


def training_loop(which):
    tf.get_logger().setLevel("INFO")
    df = pd.concat(
        [
            pd.read_csv("data/dane treningowe_I etap.csv"),
            pd.read_csv("data/dane testowe.csv"),
        ]
    )
    label_binarizer = LabelBinarizer()

    embedding_models_paths = get_embedding_models_paths()
    bin_y = label_binarizer.fit_transform(df["class"])
    dataset = generate_datasets(df)
    embedding_models_cache = {}

    results_so_far = get_finished_embedding_models(f"./results/{which}")

    for file_path in embedding_models_paths.keys():
        result = {}
        file = embedding_models_paths[file_path]

        if file_path in results_so_far.keys():
            continue

        if file.name not in embedding_models_cache.keys():
            embedding_models_cache[file.name] = load_model(file) 

        for data in dataset.columns:
            if "lemmas" not in data and "lemmas" not in file_path:
                print(data, file_path)
                result = main(which, dataset[data], bin_y, embedding_models_cache[file.name])
            elif "lemmas" in data and "forms" not in file_path:
                print(data, file_path)
                result = main(which, dataset[data], bin_y, embedding_models_cache[file.name])
            if result != {}:
                dest_folder_path = Path(f"./results/{which}/{file_path}/")
                dest_folder_path.mkdir(parents=True, exist_ok=True)
                dest_path = dest_folder_path / (data + ".pkl")
                with dest_path.open("wb") as dest_file:
                    joblib.dump(result, dest_file)
