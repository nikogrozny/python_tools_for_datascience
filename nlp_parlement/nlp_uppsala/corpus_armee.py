import os
from typing import Tuple, List, Dict, Any, TextIO
import pandas as pd
import numpy as np
from nltk import FreqDist, NaiveBayesClassifier
from matplotlib import pyplot as plt
import re
from random import shuffle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, roc_curve, auc
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

path_data = os.getcwd()
stop_words = set(open(os.path.join(path_data, "french_stopwords.txt"), "r").read().split("\n"))
fr_words = set(open(os.path.join(path_data, "french_words.txt"), "r").read().split("\n"))


def most_freq(df: pd.DataFrame, lmax: int) -> List[str]:
    print("building dictionnary")
    texte1 = " ".join(df.loc[df.Y == 1, "objective"].tolist()).split(" ")
    texte0 = " ".join(df.loc[~(df.Y == 1), "objective"].tolist()).split(" ")
    fdist1 = FreqDist(texte1).most_common(lmax)
    fdist0 = FreqDist(texte0).most_common(lmax)
    print("removing stopwords")
    words1 = set([x[0] for x in fdist1]).intersection(fr_words).difference(stop_words)
    words = set([x[0] for x in fdist0]).intersection(fr_words).difference(stop_words).union(words1)
    words = [w for w in words if len(w) > 3]
    open(os.path.join(path_data, f"topwords.txt"), 'w').write("\n".join(words))
    return words


def tableau_sans_etiquetage(longueur: int):
    fichiers: List[str] = os.listdir(os.path.join(path_data, "ocr_sorted"))
    shuffle(fichiers)
    fichiers = fichiers[:2000]
    data = pd.DataFrame(columns=["source", "text", "is_army"])
    for nom_fichier in fichiers:
        print(nom_fichier)
        fichier: TextIO = open(os.path.join(path_data, "ocr_sorted", nom_fichier), "r", encoding="Utf-8")
        content: str = fichier.read()
        fichier.close()
        nb_blocs: int = len(content)//longueur
        if nb_blocs > 2:
            blocs: List[str] = [content[i*longueur:(i+1)*longueur] for i in range(nb_blocs)]
            for bloc in blocs:
                data = data.append({"source": nom_fichier, "text": bloc, "is_army": np.nan}, ignore_index=True)
    data.to_csv(os.path.join(path_data, "etiquetage", "premieres_annees.csv"),
                    sep=";", encoding="utf-8", index=False)


def tableau_sans_etiquetage_maj(longueur: int):
    fichiers = os.listdir(os.path.join(path_data, "ocr_sorted"))
    for nom_fichier in fichiers:
        print(nom_fichier)
        fichier: TextIO = open(os.path.join(path_data, "ocr_sorted", nom_fichier), "r", encoding="Utf-8")
        content: str = fichier.read()
        fichier.close()
        blocs = [bloc for bloc in re.split(r"[A-ZÉÈÀ\s\-]{6,}", content) if len(bloc) > longueur]
        for i, bloc in enumerate(blocs):
            fichier_sortie: TextIO = open(os.path.join(path_data, "blocs", f"{nom_fichier[:-4]}-{i}.txt"), "w",
                                          encoding="Utf-8")
            fichier_sortie.write(bloc)
            fichier_sortie.close()
        fichier.close()


def extract_material() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print("building dataset")
    labeled_data: pd.DataFrame = pd.read_csv(os.path.join(path_data, "etiquetage", "a_classer.csv"),
                                             sep=";", encoding="utf-8")
    labeled_data = labeled_data.rename({"is_army": "Y", "text": "objective"}, axis=1)
    labeled_data = labeled_data.loc[~labeled_data.Y.isna(), :]
    labeled_data.loc[:, "objective"] = labeled_data.objective.apply(clean_text)
    labeled_data.loc[:, "Y"] = labeled_data.Y.apply(int)

    unlabeled_data = pd.read_csv(os.path.join(path_data, "etiquetage", "premieres_annees.csv"),
                                             sep=";", encoding="utf-8")
    unlabeled_data = unlabeled_data.rename({"is_army": "Y", "text": "objective"}, axis=1)
    print(labeled_data.head())
    print(unlabeled_data.head())

    return labeled_data.iloc[[i for i in range(labeled_data.shape[0]) if i % 3 in [0, 1]], :], \
           labeled_data.iloc[[i for i in range(labeled_data.shape[0]) if i % 3 == 2], :], unlabeled_data


def clean_text(field: Any) -> str:
    if not isinstance(field, str):
        return ""
    else:
        return " ".join(re.findall(r"[a-zéèàçùâêûîôÄëïöü\-]+", field.lower()))


def build_matrices(df_train: pd.DataFrame, df_test: pd.DataFrame, df_apply: pd.DataFrame):
    print("building X")
    train_set = df_train.copy()
    test_set = df_test.copy()
    apply_set = df_apply.copy()
    train_set.loc[:, [f"nb_{word}" for word in top_words]] = train_set.objective.apply(
        lambda z: pd.Series([z.count(word) for word in top_words], index=[f"nb_{word}" for word in top_words]))
    print(train_set.head())
    test_set.loc[:, [f"nb_{word}" for word in top_words]] = test_set.objective.apply(
        lambda z: pd.Series([z.count(word) for word in top_words], index=[f"nb_{word}" for word in top_words]))
    apply_set.loc[:, [f"nb_{word}" for word in top_words]] = apply_set.objective.apply(
        lambda z: pd.Series([z.count(word) for word in top_words], index=[f"nb_{word}" for word in top_words]))

    train_set.drop("objective", axis=1).to_csv(os.path.join(path_data, "train.csv"),
                                               sep=";", encoding="utf-8", index=False)
    test_set.drop("objective", axis=1).to_csv(os.path.join(path_data, "test.csv"),
                                              sep=";", encoding="utf-8", index=False)
    apply_set.drop("objective", axis=1).to_csv(os.path.join(path_data, "apply.csv"),
                                               sep=";", encoding="utf-8", index=False)


def build_classify() -> Tuple[MultinomialNB, SVC]:
    train_set = pd.read_csv(os.path.join(path_data, "train.csv"), sep=";", encoding="utf-8")
    test_set = pd.read_csv(os.path.join(path_data, "test.csv"), sep=";", encoding="utf-8")
    apply_set = pd.read_csv(os.path.join(path_data, "apply.csv"), sep=";", encoding="utf-8")

    train_set.loc[:, [f"nb_{w}" for w in top_words]] = train_set.loc[:, [f"nb_{w}" for w in top_words]].apply(
        lambda z: z.apply(lambda w: 1 if w > 0 else 0), axis=1)
    test_set.loc[:, [f"nb_{w}" for w in top_words]] = test_set.loc[:, [f"nb_{w}" for w in top_words]].apply(
        lambda z: z.apply(lambda w: 1 if w > 0 else 0), axis=1)
    apply_set.loc[:, [f"nb_{w}" for w in top_words]] = apply_set.loc[:, [f"nb_{w}" for w in top_words]].apply(
        lambda z: z.apply(lambda w: 1 if w > 0 else 0), axis=1)

    classifierNB = NaiveBayesClassifier.train([(row[[f"nb_{w}" for w in top_words]], row["Y"])
                                               for i, row in train_set.iterrows()])
    test_set0 = [(row[[f"nb_{w}" for w in top_words]], row["Y"]) for i, row in test_set.iterrows()]
    test_result = test_set.loc[:, [f"nb_{w}" for w in top_words]].apply(lambda z: classifierNB.classify(z), axis=1)
    print(classifierNB.show_most_informative_features(30))
    print(test_set.loc[(test_result == 1) & (test_set.Y == 1), :].shape)
    print(test_set.loc[(test_result == 1) & (test_set.Y == 0), :].shape)
    print(test_set.loc[(test_result == 0) & (test_set.Y == 1), :].shape)
    print(test_set.loc[(test_result == 0) & (test_set.Y == 0), :].shape)

    mnB = MultinomialNB()
    mnB.fit(train_set.loc[:, [f"nb_{w}" for w in top_words]], train_set.Y)
    print(mnB.score(train_set.loc[:, [f"nb_{w}" for w in top_words]], train_set.Y))
    svm = SVC(probability=True)
    svm.fit(train_set.loc[:, [f"nb_{w}" for w in top_words]], train_set.Y)
    print(svm.score(train_set.loc[:, [f"nb_{w}" for w in top_words]], train_set.Y))
    rfc = RandomForestClassifier()
    rfc.fit(train_set.loc[:, [f"nb_{w}" for w in top_words]], train_set.Y)
    print(rfc.score(train_set.loc[:, [f"nb_{w}" for w in top_words]], train_set.Y))

    fpr: Dict[str, Any] = dict()
    tpr: Dict[str, Any] = dict()
    algos: Dict[str, Any] = {"multiNB": mnB, "SVM": svm, "RndFor": rfc}
    for algo in algos:
        X_test: pd.DataFrame = test_set.loc[:, [f"nb_{w}" for w in top_words]]
        Y_test: pd.Series = test_set.Y
        Y_pred: np.ndarray[int] = algos[algo].predict(X_test)
        acc_score = accuracy_score(Y_test, Y_pred)
        pre_score = precision_score(Y_test, Y_pred)
        rec_score = recall_score(Y_test, Y_pred)
        conf_mat = confusion_matrix(Y_test, Y_pred)
        fpr[algo], tpr[algo], _ = roc_curve(Y_test, algos[algo].predict_proba(X_test)[:, 1], pos_label=1)
        roc_auc = auc(fpr[algo], tpr[algo])
        print(acc_score, pre_score, rec_score, roc_auc)
        print(conf_mat)

    for i in range(1):
        plt.figure()
        for algo in algos:
            plt.plot(fpr[algo], tpr[algo], label=algo)
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC raw_material")
        plt.legend()
        plt.savefig(os.path.join(path_data, f"roc_rawmat_{i}.png"))
    for i in range(2):
        feature_counts = np.argsort(mnB.feature_count_[i, :])[-20:]
        print(i, [top_words[i] for i in feature_counts])

    for algo in algos:
        apply_set.loc[:, "Y"] = algos[algo].predict(apply_set.loc[:, [f"nb_{w}" for w in top_words]])
        print(apply_set.groupby("Y").count())
        predictions: pd.DataFrame = pd.concat([train_set, test_set, apply_set],
                                              axis=0).drop([f"nb_{w}" for w in top_words], axis=1)
        predictions.loc[predictions.Y == 1, :].to_csv(os.path.join(path_data, f"mines_{algo}.csv"), sep=";",
                                                      encoding="utf-8", index=False)

    apply_set.loc[:, "Y"] = apply_set.loc[:, [f"nb_{w}" for w in top_words]].apply(lambda z: classifierNB.classify(z),
                                                                                   axis=1)
    print(apply_set.groupby("Y").count())
    predictions = pd.concat([train_set, test_set, apply_set], axis=0).drop([f"nb_{w}" for w in top_words], axis=1)
    predictions.loc[predictions.Y == 1, :].to_csv(os.path.join(path_data, f"mines_NBnltk.csv"), sep=";",
                                                  encoding="utf-8", index=False)

    return mnB, svm


def stats_and_graphs():
    result1 = pd.read_csv(os.path.join(path_data, "mines_multiNB.csv"), sep=";", encoding="utf-8")
    result2 = pd.read_csv(os.path.join(path_data, "mines_NBnltk.csv"), sep=";", encoding="utf-8")
    local_multiNB = set(result1.index)
    local_NBnltk = set(result2.index)
    print("only multiNB", len(local_multiNB.difference(local_NBnltk)))
    print("only NBnltk", len(local_NBnltk.difference(local_multiNB)))
    print("both", len(local_multiNB.intersection(local_NBnltk)))


if __name__ == "__main__":
    # tableau_sans_etiquetage(6000)
    # train_data, test_data, apply_data = extract_material()
    # if os.path.exists(os.path.join(path_data, f"topwords.txt")):
    #     top_words = open(os.path.join(path_data, f"topwords.txt"), 'r').read().split("\n")
    # else:
    #     top_words = most_freq(train_data, 800)
    # build_matrices(train_data, test_data, apply_data)
    # build_classify()
    stats_and_graphs()