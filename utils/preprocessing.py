import spacy


def clean(df, col_name="text"):
    # whitespaces
    df.loc[:, col_name] = df.loc[:, col_name].replace("(\s)+", " ", regex=True)

    # symbol
    df.loc[:, col_name] = df.loc[:, col_name].replace("§ \d+", " ", regex=True).replace("§[\d|.]*", " ", regex=True)

    # subitems
    df.loc[:, col_name] = df.loc[:, col_name].replace(
        "^M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})\. ",
        " ",
        regex=True
    )
    df.loc[:, col_name] = df.loc[:, col_name].replace("\ \d+\)", " ", regex=True)
    df.loc[:, col_name] = df.loc[:, col_name].replace(
        "(\ |^)+[A-Za-z]\.",
        " ", regex=True
    ).replace("(\ |^)[A-Za-z]\)", "", regex=True)

    # "Dziennik ustaw"
    df.loc[:, col_name] = df[col_name].str.replace(
        "((\())(dziennik ustaw|dz.u.|dz. u.|DzU).*(późniejszymi zmianami|późn. zm.|t.j.)(\))",
        "<odniesienie>",
        regex=True,
        case=False
    )

    # Empty brackets
    df.loc[:, col_name] = df.loc[:, col_name].replace("(\[\])|(\(\))", "", regex=True)

    # Spaces for filling in on paper
    df.loc[:, col_name] = df.loc[:, col_name].replace("\([\.|\ |…]+\)", "", regex=True).replace("_{2,}", "", regex=True)

    # Additional dots
    df.loc[:, col_name] = df.loc[:, col_name].replace("(\s)+", " ", regex=True)
    df.loc[:, col_name] = df.loc[:, col_name].replace("[\.|\ ][\.|\ ]+", ". ", regex=True)

    # Additional punctation marks on lines' ends
    df.loc[:, col_name] = df.loc[:, col_name].replace("[\ .?!:-]+$", ".", regex=True)

    # Short (max 3 words) sentences started on the end of obserwation
    df.loc[:, col_name] = df.loc[:, col_name].replace("[.?!](\ )\w+(\s\w+){,2}[^.]$", "", regex=True)

    # Missing spaces
    df.loc[:, col_name] = df.loc[:, col_name].replace('\B(?<=\d)(?!\d)', " ", regex=True)
    df.loc[:, col_name] = df.loc[:, col_name].replace("(?<=[\)])(?=[^\ ])", r" ", regex=True)

    # Unnecessary spaces vol2
    df.loc[:, col_name] = df.loc[:, col_name].replace("(\s)+", " ", regex=True)

    return df


def tokenize(df, col_name="text"):
    df[col_name] = df[col_name].split("")
    return df


def remove_stopwords(df, col_name="text"):
    pl = spacy.load("pl_core_news_lg")
    stopwords = pl.Defaults.stop_words
    print(stopwords)
    df[col_name] = df[col_name].apply(lambda x: " ".join([word for word in x.split() if word not in stopwords]))
    return df


def lemmatize(df, col_name="text"):
    pl = spacy.load("pl_core_news_lg")
    df[col_name] = df[col_name].apply(lambda x: pl(x))
    df[col_name] = df[col_name].apply(lambda x: " ".join([token.lemma_ for token in x]))
    return df
