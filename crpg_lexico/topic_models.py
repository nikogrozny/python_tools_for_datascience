import os
from typing import List, Tuple
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

path_data = os.getcwd()
nb_topics = 100
nb_runs = 7
words_in_topic = 25
stop_words = stopwords.words("english") + ["gt", "lt", "one", "st", "nd", "rd", "th", "br", "se", "ms", "ye", "wl",
                                           "az", "ca", "pl", "si", "na", "co", "em", "mo", "za"]


def topic_models(from_rep: str, to_rep: str) -> None:
    print("Lemmatization des textes")
    vectorizer = CountVectorizer(input="filename", stop_words=stop_words)
    os.chdir(os.path.join(path_data, from_rep))
    fichiers_source: List[str] = os.listdir()
    print(f"Fichiers source: {fichiers_source}")
    X: np.ndarray = vectorizer.fit_transform(fichiers_source)
    clefs = vectorizer.get_feature_names()
    blocs = [fi.split(".")[0] + "_" + fi.split(".")[1][3:] for fi in fichiers_source]
    print("Lemmatization terminée")
    print("Topic modeling")
    for n_it in range(nb_runs):
        lda = LatentDirichletAllocation(n_components=nb_topics)
        topic_to_game: np.ndarray = lda.fit_transform(X)
        corr_ttg: pd.DataFrame = pd.DataFrame(np.vectorize(lambda z: f"{z:.3f}")(topic_to_game),
                                              columns=range(nb_topics), index=blocs)
        corr_ttg.to_csv(os.path.join(path_data, to_rep, f"blocs_jv_toptogames_{n_it}.csv"), encoding="utf-8", sep=";",
                        index=True)
        topics = pd.DataFrame({f"Topic{i}": [clefs[w] for w in top.argsort()[-words_in_topic:]]
                               for i, top in enumerate(lda.components_)})
        topics.to_csv(os.path.join(path_data, to_rep, f"blocs_jv_topics_{n_it}.csv"), encoding="utf-8", sep=";",
                      index=False)
        print(f"TM{n_it} terminée")
    print("Topic modeling terminé")


def identify_topics(rep: str):
    print("Identification des topics")
    topic_sets: List[pd.DataFrame] = list()
    seuil_similarite: int = 19
    for run in range(nb_runs):
        fichier_run = os.path.join(path_data, rep, f"blocs_jv_topics_{run}.csv")
        if os.path.isfile(fichier_run):
            topic_sets.append(pd.read_csv(fichier_run, encoding="utf-8", sep=";"))
    matches: List[Tuple[int, int, int, int]] = \
        [(index1, topic1, index2, topic2)
         for index1, df1 in enumerate(topic_sets) for topic1 in range(nb_topics)
         for index2, df2 in enumerate(topic_sets) for topic2 in range(nb_topics)
         if (index1 < index2 or index1 == index2 and topic1 < topic2)
         and len(set(df1.loc[:, f"Topic{topic1}"].to_list())
                 & set(df2.loc[:, f"Topic{topic2}"].to_list())) > seuil_similarite]
    print(matches)
    for j, df_topic_set in enumerate(topic_sets):
        for num_topic in range(nb_topics):
            mym1: List[Tuple[int, int, int, int]] = [m for m in matches if m[0] == j and m[1] == num_topic]
            mym2: List[Tuple[int, int, int, int]] = [m for m in matches if m[2] == j and m[3] == num_topic]
            if len(mym1) + len(mym2) > 0:
                ser0: pd.Series = df_topic_set.loc[:, f"Topic{num_topic}"]
                sers1: List[pd.Series] = [topic_sets[m[2]].loc[:, f"Topic{m[3]}"] for m in mym1]
                sers2: List[pd.Series] = [topic_sets[m[0]].loc[:, f"Topic{m[1]}"] for m in mym2]
                df_b: pd.DataFrame = pd.concat([ser0] + sers1 + sers2, axis=1)
                df_b.to_csv(os.path.join(path_data, rep, "recurrents", f"run{j}_topic{num_topic}.csv"))
    print("Identification terminée")
