import os
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import AgglomerativeClustering

path_data = os.getcwd()
dest = "topics_par_periode"
nb_runs = 7
nb_topics = 100
words_in_topic = 50
titres = {
    "B1": "Baldurs Gate II_EN-FR.xml", "B2": "Baldurs Gate I_EN-FR.xml",
    "CC1": "Creation Club (Fallout 4)_En-FR.xml", "CC2": "Creation Club (Skyrim)_En-Fr.xml",
    "DD": "Darkest Dungeon_EN-FR.xml", "DOS": "Divinity Original Sin 2_EN-FR.xml",
    "F1": "Fallout 1_EN-FR.xml", "F2": "Fallout 2_EN-FR.xml",
    "F3": "Fallout 3_EN-FR.xml", "F4": "Fallout 4_EN-FR.xml",
    "FNV": "Fallout NV_EN-FR.xml", "PE": "Pillars of Eternity_EN-FR.xml",
    "ES3": "TES Morrowind_EN-FR.xml", "ES4": "TES Oblivion_EN-FR.xml",
    "ES5": "TES Skyrim_EN-FR.xml", "TP": "Torment Planetscape_EN-FR.xml",
    "TN": "Torment Tides of Numenera_EN-FR.xml", "U7": "Ultima 7_EN-FR.xml",
    "U8": "Ultima 8_EN-FR.xml", "U9": "Ultima 9_EN-FR.xml",
    "WA2": "Wasteland 2_EN-FR.xml", "WI2": "Witcher 2_EN-FR.xml",
    "WI3": "Witcher 3_EN-FR.xml"
}
dates = {
    titres["B1"]: 2000, titres["B2"]: 1998, titres["CC1"]: 2017, titres["CC2"]: 2017, titres["DD"]: 2016,
    titres["DOS"]: 2017, titres["F1"]: 1997, titres["F2"]: 1998, titres["F3"]: 2008, titres["F4"]: 2015,
    titres["FNV"]: 2010, titres["PE"]: 2015, titres["ES3"]: 2002, titres["ES4"]: 2006, titres["ES5"]: 2011,
    titres["TP"]: 1999, titres["TN"]: 2017, titres["U7"]: 1992, titres["U8"]: 1994, titres["U9"]: 1999,
    titres["WA2"]: 2014, titres["WI2"]: 2011, titres["WI3"]: 2015
}


def graphe_topics(rep: str) -> None:
    print("Construction graphe entre topics")
    proximity: np.ndarray = np.loadtxt(os.path.join(path_data, rep, "prox.csv"), delimiter=",")
    edges_per_dist_min: List[np.ndarray[bool, bool]] = \
        [np.vectorize(lambda z: z >= n and z != words_in_topic)(proximity) for n in range(words_in_topic)]
    graph_dist_min: List[csr_matrix] = [csr_matrix(e) for e in edges_per_dist_min]
    connect_components: List[Tuple[int, np.ndarray[int]]] = [connected_components(g) for g in graph_dist_min]
    # chaque Tuple de connect_components : (nombre de composantes, tableau des étiquettes des composantes)
    # le tout indexé par la distance minimale pour qu'une arête soit affichée
    cc_list: List[List[Tuple[np.ndarray[int]]]] = \
        [[np.asarray(cc[1] == num).nonzero() for num in range(cc[0])] for cc in connect_components]
    runs_par_cc: List[List[np.ndarray[int]]] = \
        [[np.unique(np.vectorize(lambda z: z // nb_topics)(c)) for c in lc] for lc in cc_list]
    robustes: List[List[np.ndarray[int]]] = [[c for c in r if len(c) >= nb_runs // 2 + 1] for r in runs_par_cc]

    edges_inter_classes: np.ndarray[int, int] = \
        np.array([[i // nb_topics != j // nb_topics for i in range(nb_topics * nb_runs)]
                  for j in range(nb_topics * nb_runs)])
    edges_inter_per_dist_min: List[np.ndarray[int, int]] = [np.asarray(edges_per_dist_min[n] & edges_inter_classes)
                                                            for n in range(words_in_topic)]
    neighb: List[List[np.ndarray]] = [[edges_inter_per_dist_min[n][v, :].nonzero()[0]
                                       for v in range(nb_topics * nb_runs)] for n in range(words_in_topic)]

    x_vect: List[int] = list(range(50))
    plt.figure(figsize=(12, 8))
    y_edges: List[float] = [np.sum(e) / (nb_topics * nb_runs) for e in edges_inter_per_dist_min]
    y_cc: List[int] = [c[0] for c in connect_components]
    plt.plot(x_vect, y_edges, label="densité d'arêtes")
    plt.plot(x_vect, y_cc, label="composantes connexes")
    plt.legend()
    plt.xlabel("seuil de distance retenu")
    plt.savefig(os.path.join(path_data, rep, "densite_et _composantes_connexes.png"))

    plt.figure(figsize=(12, 8))
    y_nb_cc_robustes = [len(r) for r in robustes]
    y_dens = [np.sum(edges_per_dist_min[n]) * 10 / max(1, sum([len(c[0]) * (len(c[0]) - 1) for c in cc_list[n]]))
              for n in range(words_in_topic)]
    plt.yticks(range(0, 18, 2))
    plt.plot(x_vect, y_nb_cc_robustes, label="composantes connexes robustes")
    plt.plot(x_vect, y_dens, label="densité des composantes connexes")
    plt.xlabel("proximité minimale entre sommets reliés")
    plt.legend()
    plt.savefig(os.path.join(path_data, rep, "densite_et_cc_robustes.png"))
    plt.close()
    print("Construction terminée")


def topics_with_time(cha: AgglomerativeClustering, topic2game: pd.DataFrame, topic2word: pd.DataFrame, rep: str):
    loc_rep: str = "series_temp"
    print("Calcul intensité temporelle topics")
    temps: List[int] = sorted(dates.values())
    jeux_par_annee: List[List[int]] = [[g for g in dates if dates[g] == d] for d in temps]
    robustes: Dict[int, List[str]] = dict()
    words: Dict[int, pd.DataFrame] = dict()

    for i in range(cha.n_clusters_):
        cluster_names: List[str] = [f"{j // nb_topics}-{j % nb_topics}" for j in range(len(cha.labels_))
                                    if cha.labels_[j] == i]
        print(f"Cluster {i}: {cluster_names}")

        cluster_games: pd.DataFrame = topic2game.loc[:, cluster_names]
        cluster_games.to_csv(os.path.join(path_data, rep, loc_rep, f"cluster_{i}.csv"), encoding="utf-8", sep=";")
        cluster_words: pd.DataFrame = topic2word.loc[:, cluster_names]
        cluster_words.to_csv(os.path.join(path_data, rep, loc_rep, f"cluster_{i}_words.csv"), encoding="utf-8", sep=";")
        topic_intensite: List[int] = [sum(topic2game.loc[g[:-4], cluster_names].sum() for g in z)
                                      for z in jeux_par_annee]
        plt.figure(figsize=(15, 10))
        plt.plot(temps, topic_intensite)
        plt.xticks(temps, labels=temps)
        plt.ylabel("Intensité")
        plt.xlabel("année")
        plt.title(f"Topics {','.join(cluster_names)}")
        plt.savefig(os.path.join(path_data, rep, loc_rep, f"cluster_{i}.png"))
        plt.close()

        if len(set(j // nb_topics for j in range(len(cha.labels_)) if cha.labels_[j] == i)) > nb_runs // 2:
            robustes[i] = cluster_names
            words[i] = pd.concat([cluster_words.loc[:, t] for t in cluster_names], ignore_index=True).value_counts()
            words[i] = words[i][words[i] > 2].reset_index().loc[:, "index"]
            print(words[i])
    pd.concat(words, axis=1).to_csv(os.path.join(path_data, rep, "series_temp_robust", f"all_words.csv"),
                                    encoding="utf-8", sep=";")
    print("Calcul terminé")