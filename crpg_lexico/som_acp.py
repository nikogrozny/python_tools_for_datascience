import os
import pickle
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

path_data = os.getcwd()
dest = "topics_par_periode"
nb_runs = 7
nb_topics = 100
words_in_topic = 25
distance_max = 8
dic_eng = open(os.path.join(path_data, "words_alpha.txt"), "r", encoding="utf-8")
english_words = set(dic_eng.read().split("\n"))
dic_eng.close()
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
colormap = ["lightcoral", "gold", "yellowgreen", "palegreen", "lightskyblue", "plum", "lightpink", "sandybrown",
            "orange", "khaki", "dodgerblue"]


def so_map(rep: str):
    print("Génération de la SOM")
    top2game_run: List[pd.DataFrame] = list()
    nb_neurons = 12
    for r in range(nb_runs):
        top2game_run.append(pd.read_csv(os.path.join(path_data, rep, f"blocs_jv_toptogames_{r}.csv"), sep=";",
                                        index_col=0).reset_index())
        top2game_run[-1].loc[:, "jeu"] = top2game_run[-1].loc[:, "index"].apply(lambda z: z.split("__")[0])
        top2game_run[-1] = top2game_run[-1].drop("index", axis=1).groupby("jeu").sum().drop("Battle Front 2_EN-FR",
                                                                                            axis=0)
        top2game_run[-1] = top2game_run[-1].div(top2game_run[-1].sum(axis=0), axis=1)
        top2game_run[-1] = top2game_run[-1].rename({i: f"{r * nb_topics + int(i)}" for i in top2game_run[-1].columns},
                                                   axis=1)
    topics2games: pd.DataFrame = pd.concat(top2game_run, axis=1).reset_index()
    topics2games.loc[:, "date"] = topics2games.jeu.apply(lambda z: dates[f"{z}.xml"])
    topics2date: pd.DataFrame = topics2games.copy()
    for k in range(nb_runs * nb_topics):
        topics2games.loc[:, str(k)] = topics2games.loc[:, [str(k), "date"]].apply(lambda z: z[str(k)] * z["date"],
                                                                                  axis=1)
    print("Topics par date", topics2date.head())

    robustes_files: List[str] = [fi for fi in os.listdir(os.path.join(path_data, rep, "series_temp_robust"))
                                 if fi[:8] == "cluster_"]
    robustes_nums: List[str] = [adr.split("_")[1] for adr in robustes_files]
    print(f"Classes robustes {robustes_nums}")

    superclasses_file = open(os.path.join(path_data, rep, "superclasses.csv"), "r")
    superclasses: List[List[str]] = [li.split(",") for li in superclasses_file.read().split("\n")]
    superclasses_file.close()
    print(f"Superclasses {superclasses}")

    rob_to_sc: Dict[str, List[int]] = dict()
    for r in robustes_nums:
        rob_to_sc[r]: List[int] = [s for s in range(len(superclasses)) if r in superclasses[s]]
    print(f"Classes robustes par superclasse {rob_to_sc}")

    with open(os.path.join(path_data, rep, "SOM_clusters.p"), 'rb') as infile:
        som = pickle.load(infile)
    data: np.ndarray[float, float] = np.loadtxt(os.path.join(path_data, rep, "prox_clusters.csv"), delimiter=",")
    data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
    data = data[:, :-1]
    classes_in_case: Dict[Tuple[int, int], List[int]] = {(xn, yn): list() for xn in range(nb_neurons)
                                                         for yn in range(nb_neurons)}
    superclasses_in_case: Dict[Tuple[int, int], List[List[int]]] = {(xn, yn): list() for xn in range(nb_neurons)
                                                                    for yn in range(nb_neurons)}
    for num_donnee, donnee in enumerate(data):
        classes_in_case[som.winner(donnee)].append(num_donnee)
        if str(num_donnee) in robustes_nums:
            superclasses_in_case[som.winner(donnee)].append(rob_to_sc[str(num_donnee)])
    print(classes_in_case)
    print([(i, superclasses_in_case[i]) for i in superclasses_in_case if len(superclasses_in_case[i]) > 0])

    plt.figure(figsize=(12, 12))
    plt.xlim(0, 12)
    plt.ylim(0, 12)
    plt.yticks([])
    plt.xticks([])
    size = 1950
    for c in classes_in_case:
        if len(classes_in_case[c]) > 0:
            for num_donnee, t in enumerate(classes_in_case[c]):
                plt.annotate(xy=(c[0] + .15 + 0.42 * (num_donnee % 2), c[1] + .7 - (num_donnee // 2) / 4), s=t,
                             fontsize="small", fontweight="heavy" if str(t) in robustes_nums else "normal")
        if len(superclasses_in_case[c]) >= 1:
            t = [x for co in superclasses_in_case[c] for x in co]
            if len(list(set(t))) == 1:
                plt.scatter(c[0] + .5, c[1] + .5, color=colormap[t[0]], s=size, marker="s")
            else:
                plt.scatter(c[0] + .5, c[1] + .5, color=colormap[t[0]], s=size, marker="s")
                plt.scatter(c[0] + .5, c[1] + .5, color=colormap[t[1]], s=size / 3, marker="s")
        else:
            plt.scatter(c[0] + .5, c[1] + .5, color="lightgray", s=size, marker="s")
    plt.savefig(os.path.join(path_data, rep, "som_classes_colored_v2.png"))
    print("Génération terminée")


def do_acp() -> None:
    print("Génération de l'ACP")
    top2game_run: List[pd.DataFrame] = list()
    for r in range(nb_runs):
        top2game_run.append(pd.read_csv(os.path.join(path_data, "topics_par_periode",
                                                     f"blocs_jv_toptogames_{r}.csv"), sep=";",
                                        index_col=0).reset_index())
        top2game_run[-1].loc[:, "jeu"] = top2game_run[-1].loc[:, "index"].apply(lambda z: z.split("__")[0])
        top2game_run[-1] = top2game_run[-1].drop("index", axis=1).groupby("jeu").sum().drop("Battle Front 2_EN-FR",
                                                                                            axis=0)
        top2game_run[-1] = top2game_run[-1].div(top2game_run[-1].sum(axis=0), axis=1)
        top2game_run[-1] = top2game_run[-1].rename({i: f"{r * nb_topics + int(i)}" for i in top2game_run[-1].columns},
                                                   axis=1)
    topics2games = pd.concat(top2game_run, axis=1).reset_index()
    print(topics2games.head())

    t2g_data: pd.DataFrame = topics2games.dropna(axis=1, how="all")
    features: List[str] = [c for c in t2g_data.columns if c != "jeu"]
    for fi in features:
        t2g_data.loc[:, fi] = t2g_data.loc[:, fi].astype(float)
    print(f"Data pour ACP {t2g_data.info()}")
    X: np.ndarray[float, float] = t2g_data.loc[:, features].values
    print(f"Shape {X.shape}")
    StandardScaler().fit_transform(X)

    pca = PCA()
    XP: np.ndarray[float, float] = pca.fit_transform(X)
    print(f"Ratios de variance {pca.explained_variance_ratio_}")

    plt.figure(figsize=(10, 10))
    ax = plt.subplot(111)
    ax.scatter(XP[:, 0], XP[:, 1], edgecolors="none", marker="o", c="red")
    for i, j in enumerate(t2g_data.jeu):
        plt.annotate(j, xy=(XP[i, 0], XP[i, 1]))
    plt.savefig(os.path.join(path_data, "acp.png"))

    trans_X: np.ndarray[float, float] = X.transpose()
    XP: np.ndarray[float, float] = pca.fit_transform(trans_X)
    print(f"Ratios de variance {pca.explained_variance_ratio_}")

    f = open(os.path.join(path_data, "topics_par_periode", "superclasses.csv"), "r")
    superclasses: List[List[str]] = [li.split(",") for li in f.read().split("\n")]
    f.close()

    plt.figure(figsize=(10, 10))
    ax = plt.subplot(111)
    for i in range(nb_runs * nb_topics):
        ax.scatter(XP[i, 0], XP[i, 1], edgecolors="none", marker="o", c="red")
        plt.annotate(i, xy=(XP[i, 0], XP[i, 1]))
    plt.savefig(os.path.join(path_data, "acp2.png"))

    t2g_data: np.ndarray[float, float] = np.loadtxt(os.path.join(path_data, "topics_par_periode", "prox_clusters.csv"),
                                                delimiter=",")
    t2g_data: np.ndarray[float, float] = (t2g_data - np.mean(t2g_data, axis=0)) / np.std(t2g_data, axis=0)
    XP: np.ndarray[float, float] = pca.fit_transform(t2g_data)
    print(pca.explained_variance_ratio_)
    plt.figure(figsize=(10, 10))
    plt.subplot(111)
    for i in range(t2g_data.shape[0]):
        if str(i) in [c for su in superclasses for c in su]:
            plt.scatter(XP[i, 0], XP[i, 1], edgecolors="none", marker="o", c=colormap[findsup(i, superclasses)])
            plt.annotate(i, xy=(XP[i, 0] + 0.1, XP[i, 1] + 0.1))
        else:
            plt.scatter(XP[i, 0], XP[i, 1], edgecolors="none", marker="o", c="lightgray")
    plt.savefig(os.path.join(path_data, "acp3.png"))
    print("Génération terminée")


def findsup(i: int, supcla: List[List[str]]) -> int:
    isinto: List[bool] = [str(i) in s for s in supcla]
    return min([j for j in range(len(isinto)) if isinto[j]])
