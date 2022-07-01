from time import time
from typing import Dict, List
import os
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import Normalizer
from matplotlib import pyplot as plt

path_data = os.getcwd()

def build_data():
    data_init: Dict[str, pd.DataFrame] = {adr: pd.read_csv(os.path.join(path_data, "peaks", adr), sep=";",
                                                           encoding="utf-8")
                                          for adr in os.listdir(os.path.join(path_data, "peaks"))}
    all_texts: Dict[str, str] = dict()
    for dat in data_init:
        previous, num = 0, 0
        data_init[dat].loc["n_pic"] = 0
        for i, row in data_init[dat].iterrows():
            if row["timing"] - previous > 360:
                num += 1
            data_init[dat].loc[i, "n_pic"] = num
            previous = row["timing"]
        blocs = [data_init[dat].loc[data_init[dat].loc[:, "n_pic"].apply(int) == i, "texte"].to_list()
                 for i in range(1, num + 1)]
        for j, b in enumerate(blocs):
            all_texts[f"{dat}_{j}.txt"] = " ".join([str(l) for l in b])

    for pic in all_texts:
        f = open(os.path.join(path_data, "peaks_raw", pic), "w", encoding="utf-8")
        f.write(all_texts[pic])
        f.close()


def tfidf_acp():
    t0 = time()
    vectorizer = TfidfVectorizer(max_df=0.5, max_features=5000, min_df=2, stop_words="english", input="filename")
    clefs = os.listdir(os.path.join(path_data, "peaks_raw"))
    X = vectorizer.fit_transform([os.path.join(path_data, "peaks_raw", adr) for adr in clefs])
    print("done in %fs" % (time() - t0))
    print("n_samples: %d, n_features: %d" % X.shape)

    n_components = 10
    t0 = time()
    normalizer = Normalizer(copy=False)
    X = normalizer.fit_transform(X)

    print("done in %fs" % (time() - t0))
    km = KMeans(n_clusters=10, max_iter=100)
    print("Clustering sparse data with %s" % km)
    t0 = time()
    km.fit(X)
    print("done in %0.3fs" % (time() - t0))
    print()
    print(X.shape)
    print(len(km.labels_))
    print(km.labels_.shape)

    clusters: List[List[str]] = [list() for i in range(n_components)]
    for i in range(n_components):
        clusters[i] = [clefs[j] for j in range(len(clefs)) if km.labels_[j] == i]
    print(clusters)

    print(km.cluster_centers_)

    print("Top terms per cluster:")
    order_centroids = km.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names()
    for i in range(n_components):
        print("Cluster %d:" % i, end="")
        for ind in order_centroids[i, :20]:
            print(" %s" % terms[ind], end="")
            print()

    acp = PCA()
    Xtr = acp.fit_transform(X.toarray())
    plt.figure(figsize=(15, 15))
    plt.scatter(Xtr[:, 0], Xtr[:, 1], c=km.labels_, cmap="tab10")
    for i in range(len(km.labels_)):
        if clusters[km.labels_[i]][0] == clefs[i]:
            plt.annotate(clefs[i], (Xtr[i, 0] + 0.05, Xtr[i, 1] + 0.05))
    plt.savefig(os.path.join(path_data, "acp_clusters.png"))


if __name__ == "__main__":
    build_data()
    tfidf_acp()
