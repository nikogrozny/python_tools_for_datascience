import os
import re
from typing import Dict
import pandas as pd
import nltk
from nltk.corpus import stopwords

path_data = os.getcwd()
stop_words = stopwords.words("english") + ["gt", "lt", "one", "st", "nd", "rd", "th", "br", "se", "ms", "ye", "wl",
                                           "az", "ca", "pl", "si", "na", "co", "em", "mo", "za"]
wnl = nltk.WordNetLemmatizer()
lem_stop_words = [wnl.lemmatize(w) for w in stop_words]
dic_eng = open(os.path.join(path_data, "words_alpha.txt"), "r", encoding="utf-8")
english_words = set(dic_eng.read().split("\n"))
dic_eng.close()


def document_features(doc) -> Dict[str, bool]:
    doc_words = set(doc.split(" "))
    features = dict()
    for word in train_words:
        features['contains({})'.format(word)] = (word in doc_words)
    return features


def build_classify(train: pd.DataFrame):
    print([(row["Extrait"], row["is_metaphore"]) for i, row in train.iterrows()][0])
    train_set = [(document_features(row["Extrait"]), row["is_metaphore"]) for i, row in train.iterrows()]
    print(train_set[0])
    classifier = nltk.NaiveBayesClassifier.train(train_set)
    print(classifier.show_most_informative_features(100))

    test = dict()
    for game in os.listdir(os.path.join(path_data, "tutos_by_hand")):
        print(game)
        test[game] = list()
        f = open(os.path.join(path_data, "tutos_by_hand", game), "r")
        for sentence in f:
            if len(sentence) > 15:
                test[game].append(sentence)
        f.close()
        results = [(t, classifier.prob_classify(document_features(t)).prob(1)) for t in test[game]]
        results.sort(key=lambda z: z[1])
        print(results[:10])
        print(results[-10:])


if __name__ == "__main__":
    tags = pd.read_csv(os.path.join(path_data, "metaphores.csv"), sep=";", encoding="utf-8", header=0)
    tags.Extrait = tags.Extrait.apply(lambda z: re.sub("[.,;:?!0-9/\"&=\{\}()\[\]_]", "", z.lower()))
    tags.is_metaphore = tags.is_metaphore.apply(lambda z: 0 if z == "Non" else 1)
    train_words = list({word for line in tags.Extrait.to_list() for word in line.split()
                        if word in english_words and word not in stop_words})
    print(train_words)
    build_classify(tags)
