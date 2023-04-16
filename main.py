from tensorflow import keras
from keras.layers import TextVectorization
from sklearn.preprocessing import LabelBinarizer
import pandas as pd
from pathlib import Path
import joblib
import optuna
from optuna.trial import TrialState
import numpy as np
import tensorflow as tf
from models import objective, create_embedding_layer, split_dataset
from readers import get_models, get_calculated_models, generate_dataset
from gensim.models import KeyedVectors


def main(which, x, y, word2vec_model, n_trails=5):
    shape = word2vec_model.vectors.shape
    vectorizer = TextVectorization(max_tokens=shape[0], output_sequence_length=int(x.str.split().str.len().max()))
    vectorizer.adapt(x)
    # dict mapping words to their indices
    voc = vectorizer.get_vocabulary()

    # create embedding layer
    vectorized_x = vectorizer(np.array([[s] for s in x])).numpy()
    embedding_layer = create_embedding_layer(voc, shape, word2vec_model)

    #create model

    #test train split
    x_train, x_valid, y_train, y_valid = split_dataset(vectorized_x, y)

    func = lambda trail: objective(trail, which, embedding_layer, x_train, y_train, x_valid, y_valid)
    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(),
        storage="sqlite:///db.sqlite3"
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


if __name__ == "__main__":

    tf.get_logger().setLevel('INFO')
    df = pd.read_csv("data/dane treningowe_I etap.csv")
    df_2 = pd.read_csv("data/dane testowe.csv")
    df = pd.concat([df, df_2])
    label_binarizer = LabelBinarizer()

    models = get_models()
    bin_y = label_binarizer.fit_transform(df["class"])
    dataset = generate_dataset()
    BATCH_SIZE = 255
    EPOCHS = 100
    which = "cnn_lstm"
    modelss = {}

    curr_results = get_calculated_models(f"./results/{which}")

    for file_name in models.keys():
        result = {}
        file = models[file_name]

        if file_name in curr_results.keys():
            continue
        if file.name not in modelss.keys():
            if str(file).endswith(".txt"):
                print(file.name)
                modelss[file.name] = KeyedVectors.load_word2vec_format(file, binary=False)
            elif str(file_name).endswith(".bin"):
                print(file.name)
                modelss[file.name] = KeyedVectors.load(str(file))

        for data in dataset.columns:
            if data == "no_stopwords":
                continue
            if "lemmas" not in data and "lemmas" not in file_name:
                print(data, file_name)
                result =  main(which, dataset[data], bin_y, modelss[file.name])
            elif "lemmas" in data and "forms" not in file_name:
                print(data, file_name)
                result = main(which, dataset[data], bin_y, modelss[file.name])
            if result != {}:
                dest_folder_path = Path(f"./results/{which}/{file_name}/")
                dest_folder_path.mkdir(parents=True, exist_ok=True)
                dest_path = dest_folder_path / (data + ".pkl")
                with dest_path.open('wb') as dest_file:
                    joblib.dump(result, dest_file)
