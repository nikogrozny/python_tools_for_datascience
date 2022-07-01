import os
from math import sqrt
from typing import List
import seaborn as sns
import numpy as np
import networkx as nx
import pandas as pd
import re
import plotly.graph_objects as go
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import LatentDirichletAllocation, PCA
from sklearn.feature_extraction.text import CountVectorizer
from sklearn_extra.cluster import KMedoids
import spacy
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE, MDS

nlp = spacy.load("fr_core_news_md")
# matplotlib.use('Qt4Agg')
path_data = os.path.join(os.getcwd(), "data")
path_exports = os.path.join(os.getcwd(), "exports")
list_data_dir: List[str] = ["pixel_war", "corpus_lausanne"]


def compute_frequences(pixelwar: str):
    if pixelwar == "Non":
        emissions: List[str] = sorted([adr for adr in os.listdir(os.path.join(path_data, "dataframe_comments"))
                                       if "pixel_war" not in adr])
    elif pixelwar == "Uniquement":
        emissions: List[str] = sorted([adr for adr in os.listdir(os.path.join(path_data, "dataframe_comments"))
                                       if "pixel_war" in adr])
    else:
        emissions: List[str] = sorted([adr for adr in os.listdir(os.path.join(path_data, "dataframe_comments"))])
    all_data: pd.DataFrame = pd.DataFrame(columns=["date", "streamer", "message"])
    for adr in emissions:
        df = pd.read_csv(os.path.join(path_data, "dataframe_comments", adr), sep=";", encoding="utf-8", header=0)
        dico = {"date": df.date.loc[0], "streamer": df.streamer.loc[0],
                "message": " ".join(df.message.apply(str).to_list())}
        all_data = all_data.append(dico, ignore_index=True)
    print(all_data.info())
    all_data.loc[:, "message"] = all_data.loc[:, "message"].apply(
        lambda z: " ".join(re.findall(r"[a-zéèçàùâêûîôäëïöü\-]+", z.lower())))
    cv: CountVectorizer = CountVectorizer(min_df=25, max_df=0.75)
    X: np.ndarray = cv.fit_transform(all_data.message).toarray()
    clefs: List[str] = cv.get_feature_names_out()
    print(X.shape)
    print(len(clefs))
    pd.DataFrame(columns=clefs, data=X).to_csv(os.path.join(path_exports, "count_all_words.csv"),
                                               sep=";", encoding="utf-8", index=False)


def tokenize():
    frequences: pd.DataFrame = pd.read_csv(os.path.join(path_exports, "count_all_words.csv"), sep=";", encoding="utf-8")
    frequences.loc["nom", :] = [tok.lemma_ for tok in nlp(" ".join(frequences.columns))]
    frequences = frequences.transpose()
    frequences = frequences.groupby("nom").sum().transpose()
    frequences.to_csv(os.path.join(path_exports, "count_all_tokens.csv"), sep=";", encoding="utf-8", index=False)


def compute_topics():
    words_per_topic: int = 20
    frequences: pd.DataFrame = pd.read_csv(os.path.join(path_exports, "count_all_tokens.csv"),
                                           sep=";", encoding="utf-8", index_col=None)
    print(frequences.head())
    X: np.ndarray = frequences.values
    clefs: List[str] = list(frequences.columns)
    print(X.shape)
    for nb_topics in range(8, 22, 2):
        lda = LatentDirichletAllocation(n_components=nb_topics)
        topic_to_text = lda.fit_transform(X)
        topics: pd.DataFrame = pd.DataFrame({f"Topic{i}": [clefs[w] for w in top.argsort()[-words_per_topic:]]
                                             for i, top in enumerate(lda.components_)})
        table_topics_to_texts: pd.DataFrame = pd.DataFrame(np.vectorize(lambda z: f"{z:.3f}")(topic_to_text),
                                                           columns=range(nb_topics))
        topics.to_excel(os.path.join(path_exports, f"topics_{nb_topics}.xlsx"), encoding="utf-8", index=False)
        table_topics_to_texts.to_excel(os.path.join(path_exports, f"corpus_topics_{nb_topics}.xlsx"), encoding="utf-8",
                                       index=True)


def visu_topics(pixelwar: str):
    all_meta: pd.DataFrame = pd.concat([pd.read_csv(os.path.join(path_data, "par_auteur", adr),
                                                    sep=";", encoding="utf-8", header=0).drop(["message"], axis=1)
                                        for adr in os.listdir(os.path.join(path_data, "par_auteur"))],
                                       axis=0).reset_index().drop("index", axis=1)
    if pixelwar == "Non":
        all_meta = all_meta.loc[all_meta.event != "pixel_war", :].reset_index().drop("index", axis=1)
        list_data_dir.remove("pixel_war")
    if pixelwar == "Uniquement":
        all_meta = all_meta.loc[all_meta.event == "pixel_war", :].reset_index().drop("index", axis=1)
        list_data_dir.remove("corpus_lausanne")
    print(all_meta.info())

    for nb_topics in range(8, 22, 2):
        print(nb_topics)
        table_top2text: pd.DataFrame = pd.read_excel(os.path.join(path_exports, f"corpus_topics_{nb_topics}.xlsx"),
                                                     index_col=0)
        table_top2text = pd.concat([table_top2text, all_meta], axis=1)
        for data_dir in list_data_dir:
            print(data_dir)
            print(table_top2text.info())
            print(table_top2text.tail(10))
            top2text_event: pd.DataFrame = table_top2text.loc[table_top2text.event == data_dir, :] \
                .drop(["date", "event"], axis=1)
            print(top2text_event.info())
            top2text_event = top2text_event.groupby("streamer").sum()
            top2text_event.loc[:, "total"] = top2text_event.sum(axis=1)
            top2text_event = top2text_event.apply(lambda z: z / z["total"], axis=1).drop("total", axis=1)
            print(top2text_event.tail(10))

            kmeans: KMeans = KMeans(n_clusters=4)
            kmeans.fit_predict(top2text_event.values)
            cha: AgglomerativeClustering = AgglomerativeClustering()
            cha.fit_predict(top2text_event.values)

            for cl, clustering in enumerate([kmeans, cha]):
                acp: PCA = PCA()
                Xtr = acp.fit_transform(top2text_event.values)
                print(Xtr.shape, top2text_event.shape)
                plt.figure(figsize=(12, 12))
                plt.scatter(Xtr[:, 0], Xtr[:, 1], c=clustering.labels_)
                for i, nom in enumerate(top2text_event.index):
                    plt.annotate(text=nom, xy=(Xtr[i, 0] + 0.01, Xtr[i, 1]))
                loadings = acp.components_.T * np.sqrt(acp.explained_variance_)
                for i in range(nb_topics):
                    if abs(loadings[i, 0]) + abs(loadings[i, 1]) > 0.08:
                        plt.annotate(text=f"Topic {i}", xy=(loadings[i, 0], loadings[i, 1]), color="red")
                plt.title(f"Axes 1 et 2 d'une ACP sur {nb_topics} topics - {data_dir}")
                plt.savefig(os.path.join(path_exports, "img", f"ACP-12-{nb_topics}-{data_dir}-{pixelwar}.png"))

                plt.figure(figsize=(12, 12))
                plt.scatter(Xtr[:, 2], Xtr[:, 3], c=clustering.labels_)
                for i, nom in enumerate(top2text_event.index):
                    plt.annotate(text=nom, xy=(Xtr[i, 2] + 0.01, Xtr[i, 3]))
                loadings = acp.components_.T * np.sqrt(acp.explained_variance_)
                for i in range(nb_topics):
                    if abs(loadings[i, 2]) + abs(loadings[i, 3]) > 0.08:
                        plt.annotate(text=f"Topic {i}", xy=(loadings[i, 2], loadings[i, 3]), color="red")
                plt.title(f"Axes 3 et 4 d'une ACP sur {nb_topics} topics - {data_dir}")
                plt.savefig(os.path.join(path_exports, "img", f"ACP-34-{nb_topics}-{data_dir}-{pixelwar}.png"))

                tsne: TSNE = TSNE(n_components=2, learning_rate='auto', init='pca', perplexity=Xtr.shape[0] // 3)
                Xtr = tsne.fit_transform(top2text_event.values)
                print(Xtr.shape, top2text_event.shape)
                plt.figure(figsize=(12, 12))
                plt.scatter(Xtr[:, 0], Xtr[:, 1], c=clustering.labels_)
                for i, nom in enumerate(top2text_event.index):
                    plt.annotate(text=nom, xy=(Xtr[i, 0] + 2, Xtr[i, 1]))
                plt.title(f"t-SNE sur {nb_topics} topics - {data_dir}")
                plt.savefig(os.path.join(path_exports, "img", f"tSNE-{nb_topics}-{data_dir}--{cl}-{pixelwar}.png"))


def compute_followers():
    all_data: pd.DataFrame = pd.concat([pd.read_csv(os.path.join(path_data, "par_auteur_vw", adr),
                                                    sep=";", encoding="utf-8", header=0) for adr
                                        in os.listdir(os.path.join(path_data, "par_auteur_vw"))],
                                       axis=0).drop("date", axis=1)
    for data_dir in list_data_dir:
        local_data: pd.DataFrame = all_data.loc[all_data.event == data_dir, :].drop("event", axis=1)
        local_data = local_data.groupby(['streamer'])['author'].apply(lambda x: ' '.join(x)).reset_index()
        local_data.loc[:, "author"] = local_data.loc[:, "author"].apply(
            lambda z: ' '.join(sorted(list(set(z.split())))))
        streamers = local_data.streamer.unique()
        local_data.set_index("streamer", inplace=True)
        matrix: pd.DataFrame = pd.DataFrame(index=streamers, columns=streamers)
        for streamer1 in streamers:
            for streamer2 in streamers:
                matrix.loc[streamer1, streamer2] = len(set(local_data.loc[streamer1, "author"].split())
                                                       .intersection(set(local_data.loc[streamer2, "author"].split())))
        print(matrix)

        print("***Réduction dimensionelle***")
        distance: pd.DataFrame = pd.DataFrame(index=streamers, columns=streamers)
        for streamer1 in streamers:
            for streamer2 in streamers:
                distance.loc[streamer1, streamer2] = 1 - 2 * matrix.loc[streamer1, streamer2] \
                                                     / (matrix.loc[streamer1, streamer1] + matrix.loc[
                    streamer2, streamer2])

        kmed: KMedoids = KMedoids(n_clusters=4)
        kmed.fit_predict(distance.values)
        cha: AgglomerativeClustering = AgglomerativeClustering(affinity="precomputed", linkage="complete")
        cha.fit_predict(distance.values)

        for cl, clustering in enumerate([kmed, cha]):
            mds: MDS = MDS(dissimilarity="precomputed")
            Xtr = mds.fit_transform(distance.values)
            plt.figure(figsize=(12, 12))
            plt.scatter(Xtr[:, 0], Xtr[:, 1], c=clustering.labels_)
            for i, nom in enumerate(distance.index):
                plt.annotate(text=nom, xy=(Xtr[i, 0] + 0.01, Xtr[i, 1]))
            plt.title(f"MDS sur followers - {data_dir}")
            plt.savefig(os.path.join(path_exports, "img", f"MDS-followers-{data_dir}.png"))

            tsne: TSNE = TSNE(n_components=2, learning_rate='auto', perplexity=Xtr.shape[0] // 3, metric="precomputed")
            Xtr = tsne.fit_transform(distance.values)
            plt.figure(figsize=(12, 12))
            plt.scatter(Xtr[:, 0], Xtr[:, 1], c=clustering.labels_)
            for i, nom in enumerate(distance.index):
                plt.annotate(text=nom, xy=(Xtr[i, 0] + 0.01, Xtr[i, 1]))
            plt.title(f"tSNE sur followers - {data_dir}")
            plt.savefig(os.path.join(path_exports, "img", f"tSNE-followers-{data_dir}--{cl}.png"))

        print("***Graphes****")

        threshold_edge = {"pixel_war": 800, "corpus_lausanne": 800}
        graphe_viewers: nx.Graph = nx.Graph()
        graphe_viewers.add_nodes_from([s for s in streamers
                                       if matrix.loc[matrix.loc[:, s] > threshold_edge[data_dir], :].shape[0] > 1])
        graphe_viewers.add_weighted_edges_from([(s1, s2, matrix.loc[s1, s2])
                                                for s1 in streamers for s2 in streamers if s1 != s2
                                                and matrix.loc[s1, s2] > threshold_edge[data_dir]])
        pos = nx.spring_layout(graphe_viewers)
        edge_trace: List[go.Scatter] = list()
        for edge in graphe_viewers.edges.data("weight"):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            weight = edge[2]
            edge_trace.append(go.Scatter(x=(x0, x1), y=(y0, y1), hoverinfo='none', mode='lines',
                                         line=dict(width=sqrt(weight) // 10, color="black")))
        node_x: List[float] = [pos[node][0] for node in graphe_viewers.nodes()]
        node_y: List[float] = [pos[node][1] for node in graphe_viewers.nodes()]
        node_labels = list(graphe_viewers.nodes())
        node_trace = go.Scatter(x=node_x, y=node_y, mode='markers', hoverinfo='text', text=node_labels,
                                marker=dict(
                                    color=cha.labels_,
                                    size=[sqrt(matrix.loc[l, l]) // 10 for i, l in enumerate(node_labels)]
                                ))
        fig = go.Figure(data=edge_trace + [node_trace],
                        layout=go.Layout(
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=20, l=5, r=5, t=40),
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                        )
        fig.write_html(os.path.join(path_exports, "graphes", f"viewers-{data_dir}.html"))
        plt.close()

        print("***Heatmaps***")

        cha: AgglomerativeClustering = AgglomerativeClustering(n_clusters=8)
        cha.fit_predict(distance.values)
        unsorted_streamers: List[str] = list(matrix.columns)
        sorted_streamers = sorted(unsorted_streamers, key=lambda z: cha.labels_[unsorted_streamers.index(z)])
        plt.figure(figsize=(12, 12))
        heater: pd.DataFrame = matrix.astype(int).loc[sorted_streamers, sorted_streamers]
        for streamer in heater.columns:
            heater.loc[:, streamer] = heater.loc[:, streamer].div(heater.loc[streamer, streamer])
        print(heater)
        heatmap: sns.heatmap = sns.heatmap(heater, vmin=0, vmax=0.3)
        heatmap.get_figure().savefig(os.path.join(path_exports, "img", f"heatmap-vw--{data_dir}.png"),
                                     bbox_inches="tight")


if __name__ == "__main__":
    compute_frequences(pixelwar="Non")
    tokenize()
    compute_topics()
    visu_topics(pixelwar="Non")
    compute_followers()
