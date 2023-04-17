import pandas as pd
from utils.preprocessing import clean, remove_stopwords, lemmatize
from itertools import combinations_with_replacement
from pathlib import Path


def generate_datasets(input_df):
    res = pd.DataFrame()
    possible_functions = {
        "clean": clean,
        "no_stopwords": remove_stopwords,
        "lemmas": lemmatize,
    }
    possible_datasets = set()

    for comb in combinations_with_replacement(possible_functions, 3):
        possible_datasets.add(tuple(sorted(tuple(set(comb)))))

    for func_comb in possible_datasets:
        print(func_comb)
        resulting_df = input_df
        dataset_name = "+".join(func_comb)
        for func in func_comb:
            if "lemmas" not in dataset_name:
                resulting_df = possible_functions[func](resulting_df)
        res[dataset_name] = resulting_df["text"]
    return res


def get_embedding_models_paths(models_path="./models"):
    models_dir = {}
    cwd = Path(models_path)
    for path in cwd.iterdir():
        if path.is_dir():
            for file in path.iterdir():
                models_dir[file.name] = file
    return models_dir


def get_finished_embedding_models(models_path):
    models_dir = {}
    cwd = Path(models_path)
    for path in cwd.iterdir():
        if path.is_dir():
            models_dir[path.name] = []
            for file in path.iterdir():
                models_dir[path.name].append(file.name)
    return models_dir
