import os
from statistics import mean
from typing import List, Tuple, Set, Iterable

import minisom
import pandas as pd
import numpy as np
from networkx import Graph
from sklearn.cluster import AgglomerativeClustering
from matplotlib import pyplot as plt
import pickle

path_data = os.getcwd()
nb_topics = 100
nb_runs = 7
words_in_topic = 50
distance_max = 8
path_dot = "insert_path_to_dot"


def topics_games_graph(adr: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    print("Construction du graphe")
    matching_topics: str = open(os.path.join(path_data, adr, "matching_topics.txt"), "r").read()
    mtops: List[List[int]] = [[int(n) for n in mtopics.split(",")] for mtopics in matching_topics.split(";")]
    print("Matching topics", mtops)

    topic_games: List[pd.DataFrame] = list()
    for num_run in range(nb_runs):
        topic_dist_file: str = os.path.join(path_data, adr, f"blocs_jv_toptogames_{num_run}.csv")
        if os.path.isfile(topic_dist_file):
            topic_games.append(pd.read_csv(topic_dist_file, encoding="utf-8", sep=";"))
            topic_games[-1] = topic_games[-1].rename({"Unnamed: 0": "game"}, axis=1)
            topic_games[-1] = topic_games[-1].rename({str(num_topic): f"{num_run}-{num_topic}"
                                                      for num_topic in range(nb_topics)}, axis=1)
            topic_games[-1].game = topic_games[-1].game.apply(lambda z: z.split("__")[0])
            topic_games[-1] = topic_games[-1].groupby("game").mean()
            prevalence: pd.Series = topic_games[-1].sum(axis=0)
            for topic in topic_games[-1].columns:
                topic_games[-1].loc[:, topic] = topic_games[-1].loc[:, topic].div(prevalence[topic])
    df_distr_top: pd.DataFrame = pd.concat(topic_games, axis=1)
    print("Distribution des topics par jeu", df_distr_top.info())

    topics: List[pd.DataFrame] = list()
    for num_run in range(nb_runs):
        topic_file: str = os.path.join(path_data, adr, f"blocs_jv_topics_{num_run}.csv")
        if os.path.isfile(topic_file):
            topics.append(pd.read_csv(topic_file, encoding="utf-8", sep=";"))
            topics[-1] = topics[-1].rename({f"Topic{num_topic}": f"{num_run}-{num_topic}"
                                            for num_topic in range(nb_topics)}, axis=1)
    df_words_top: pd.DataFrame = pd.concat(topics, axis=1)
    print("Mots principaux des topics", df_words_top.info())

    gr_main: Graph = Graph(format="png")
    gr_topics: Graph = Graph(format="png")
    for r1, t1, r2, t2 in mtops:
        gr_main.edge(f"{r1}-{t1}", f"{r2}-{t2}")
        gr_topics.edge(f"{r1}-{t1}", f"{r2}-{t2}")
    for run in range(nb_runs):
        for topic in range(nb_topics):
            topic2game: pd.DataFrame = df_distr_top.loc[df_distr_top.loc[:, f"{run}-{topic}"] > 0.25, f"{run}-{topic}"]
            for edge in topic2game.index:
                gr_main.edge(edge, f"{run}-{topic}")
    gr_main.save(os.path.join(path_data, adr, "main_graph_prev.dot"))
    gr_topics.save(os.path.join(path_data, adr, "topics_graph_prev.dot"))

    print("Positionnement des éléments")
    for tool in ["neato", "sfdp"]:
        for graph in ["main_graph_prev", "topics_graph_prev"]:
            os.system(f"{path_dot}{tool}.exe -Tpng {graph}.dot -o {graph}_{tool}.png")

    print("Construction terminée")
    return df_distr_top, df_words_top


def topics_proxy(rep: str):
    print("Calcul de la matrice de proximité")
    topic_sets: List[pd.DataFrame] = list()
    vocabulary: Set[pd.Series] = set()
    comptages: List[pd.DataFrame] = list()
    for i in range(nb_runs):
        fi: str = os.path.join(path_data, rep, f"blocs_jv_topics_{i}.csv")
        if os.path.isfile(fi):
            topic_sets.append(pd.read_csv(fi, encoding="utf-8", sep=";"))
    for tset in topic_sets:
        for i in range(nb_topics):
            vocabulary = vocabulary.union(tset.loc[:, f"Topic{i}"])
    for num_set, tset in enumerate(topic_sets):
        compte_run: pd.DataFrame = pd.concat([pd.DataFrame(
            [tset.loc[:, f"Topic{i}"].value_counts()]) for i in range(nb_topics)], ignore_index=True)
        print(f"Comptage {num_set}", compte_run.info())
        comptages.append(compte_run)
    comptage_global: pd.DataFrame = pd.concat(comptages, ignore_index=True)
    comptage_global.to_csv(os.path.join(path_data, rep, "comptage.csv"), sep=";")

    proximity: np.ndarray = np.array([
        [len(set(df1.loc[:, f"Topic{nt1}"].to_list()) & set(df2.loc[:, f"Topic{nt2}"].to_list()))
         for j1, df1 in enumerate(topic_sets) for nt1 in range(nb_topics)]
        for j2, df2 in enumerate(topic_sets) for nt2 in range(nb_topics)])
    np.savetxt(os.path.join(path_data, rep, "prox.csv"), proximity, delimiter=",")
    print("Calcul terminé")


def topics_cha(rep: str) -> AgglomerativeClustering:
    print("Classification hiérarchique")
    proximity: np.ndarray = np.loadtxt(os.path.join(path_data, rep, "prox.csv"), delimiter=",")
    distances: np.ndarray = np.vectorize(lambda x: words_in_topic - x)(proximity)

    cha = AgglomerativeClustering(affinity="precomputed", linkage="complete",
                                  distance_threshold=words_in_topic - distance_max, n_clusters=None)
    cha.fit(distances)
    print(f"Nombre de clusters : {cha.n_clusters_}")
    print(f"Labels : {cha.labels_}")
    clusters: List[List[int]] = [[x] for x in range(len(cha.labels_))]
    active: List[bool] = [True for x in range(len(cha.labels_))]
    poids: List[int] = [0]
    taille: List[int] = [0]
    max_distance: List[int] = [0]
    classes_robustes: List[int] = [0]
    num_iteration: int = 0
    f = open(os.path.join(path_data, "cha.txt"), "w", encoding="utf-8")
    while num_iteration < nb_runs * nb_topics - 1:
        v1, v2 = cha.children_[num_iteration]
        clusters.append(clusters[v1] + clusters[v2])
        active.append(True)
        active[v1] = False
        active[v2] = False
        active_clusters: List[List[int]] = [c for i, c in enumerate(clusters) if active[i]]
        f.write("\n************" + str(num_iteration) + "\n")
        f.write("\n".join([f"{i} {cluster}" for i, cluster in enumerate(active_clusters)]))
        poids.append(max([0 if len(cluster) == 1
                          else mean(distances[n_topic1, n_topic2]
                                    for n_topic1 in cluster for n_topic2 in cluster if n_topic2 < n_topic1)
                          for cluster in active_clusters]))
        max_distance.append(max([0 if len(cluster) == 1
                                 else max(distances[n_topic1, n_topic2]
                                          for n_topic1 in cluster for n_topic2 in cluster if n_topic2 < n_topic1)
                                 for cluster in active_clusters]))
        taille.append(max([len(cluster) for cluster in active_clusters]))
        classes_robustes.append(len([cluster for cluster in active_clusters
                                     if len(set([num_top // nb_topics for num_top in cluster])) >= nb_runs // 2 + 1]))
        num_iteration += 1
    f.close()

    plt.figure(figsize=(12, 8))
    xvect: Iterable = range(len(poids))
    plt.plot(xvect, max_distance, label="pire distance intra")
    plt.plot(xvect, taille, label="plus gros cluster")
    plt.plot(xvect, classes_robustes, label="nombre clusters robustes")
    plt.grid()
    plt.ylim(0, 80)
    plt.xlim(0, nb_topics * nb_runs)
    plt.legend()
    plt.xlabel("nombre d'étapes")
    plt.savefig(os.path.join(path_data, rep, "acp_dist.png"))
    plt.close()

    data: np.ndarray = (proximity - np.mean(proximity, axis=0)) / np.std(proximity, axis=0)
    data = data[:, :-1]
    n_neurons: int = 14
    m_neurons: int = 14
    som = minisom.MiniSom(n_neurons, m_neurons, data.shape[1], sigma=1.5, learning_rate=.5,
                          neighborhood_function='gaussian', random_seed=0)
    som.pca_weights_init(data)
    som.train(data, 2000, verbose=True)
    with open(os.path.join(path_data, rep, "SOM_topics.p"), 'wb') as outfile:
        pickle.dump(som, outfile)

    prox_clusters: pd.DataFrame = pd.DataFrame()
    for i in range(cha.n_clusters_):
        topics: List[int] = [j for j in range(len(cha.labels_)) if cha.labels_[j] == i]
        prox_clusters = prox_clusters.append(pd.Series(proximity[topics, :].max(axis=0)), ignore_index=True)
    np.savetxt(os.path.join(path_data, rep, "prox_clusters.csv"), prox_clusters, delimiter=",")

    data = prox_clusters.values
    data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
    data = data[:, :-1]
    n_neurons = 12
    m_neurons = 12

    som = minisom.MiniSom(n_neurons, m_neurons, data.shape[1], sigma=1.5, learning_rate=.5,
                          neighborhood_function='gaussian', random_seed=0)
    som.pca_weights_init(data)
    som.train(data, 2000, verbose=True)
    with open(os.path.join(path_data, rep, "SOM_clusters.p"), 'wb') as outfile:
        pickle.dump(som, outfile)

    return cha
