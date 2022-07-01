from typing import List, Tuple, Dict, TextIO
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import networkx as nx
import pandas as pd
import numpy as np
import os
import shutil

path_data = os.getcwd()
classes = {
    "names of MP": [0, 5, 10, 14, 18, 23, 35, 37, 39],
    "government/parliament": [1, 6, 9, 13, 17, 19, 22, 36, 38, 41, 45, 46, 49],
    "economy": [2, 4, 16],
    "working class": [7, 8, 31, 34],
    "army": [11, 48],
    "department": [12],
    "trains/communications": [15, 44],
    "local politics": [20, 33],
    "law inforcement": [21, 40],
    "school": [24],
    "alcohol": [25],
    "budget": [26, 29, 30, 43],
    "colonies": [28],
    "navy": [32],
    "building works": [27, 42],
    "foreign affairs": [47],
    "junk": [3]
}


def build_classes(cls: Dict[str, List[int]]):
    top2texts: pd.DataFrame = pd.read_excel(os.path.join(path_data, "corpus_topics.xlsx"), index_col=0)
    colonnes: List[str] = top2texts.columns.to_list()
    print(top2texts.head())
    for classe in cls:
        top2texts.loc[:, classe] = top2texts.iloc[:, [t for t in cls[classe]]].sum(axis=1)
    top2texts.drop(colonnes, axis=1, inplace=True)
    top2texts.to_excel(os.path.join(path_data, "corpus_classes.xlsx"))

    liste_fichiers: List[str] = os.listdir(os.path.join(path_data, "blocs"))
    sizes: pd.Series = pd.Series(list(range(top2texts.shape[0]))).apply(lambda z: liste_fichiers[z])
    comptes: List[int] = list()
    for i, v in sizes.iteritems():
        fichier: TextIO = open(os.path.join(path_data, "blocs", v), "r", encoding="utf-8")
        comptes.append(len(fichier.read()))
        fichier.close()
    top2texts = top2texts.reset_index()
    for c in top2texts.columns:
        top2texts.loc[:, c] = top2texts.loc[:, c].multiply(pd.Series(comptes))
    print(top2texts.sum(axis=0).div(pd.Series(comptes).sum()).sum())


def matrice_corr() -> np.ndarray:
    top2texts: pd.DataFrame = pd.read_excel(os.path.join(path_data, "corpus_classes.xlsx"), index_col=0)
    cocomptage: np.ndarray = np.array([[
        top2texts.loc[(top2texts.loc[:, top1] > 0.15) & (top2texts.loc[:, top2] > 0.15)].shape[0]
        for top1 in top2texts.columns] for top2 in top2texts.columns])
    np.savetxt(os.path.join(path_data, "cocomptage_80.csv"), cocomptage, delimiter=";", fmt='%i')
    cofrequence: np.ndarray = np.array([[
        cocomptage[i, j] * 2 / (cocomptage[i, i] + cocomptage[j, j])
        for i in range(cocomptage.shape[0])] for j in range(cocomptage.shape[1])])
    np.savetxt(os.path.join(path_data, "cofrequence_80.csv"), cofrequence, delimiter=";", fmt='%f')
    cocomptage_trie: pd.DataFrame = pd.DataFrame(cocomptage, index=classes.keys(), columns=classes.keys())
    cocomptage_trie = cocomptage_trie.drop("junk", axis=1).drop("junk", axis=0)
    cocomptage_trie = cocomptage_trie.reindex(["army"] + [c for c in cocomptage_trie.columns if c != "army"], axis=1)
    cocomptage_trie = cocomptage_trie.reindex(cocomptage_trie.columns, axis=0)
    cocomptage_trie.to_excel(os.path.join(path_data, "cofrequence_8090.xlsx"))
    return cofrequence


def draw_graph(cofrequences: np.ndarray):
    top2clas: pd.DataFrame = pd.read_excel(os.path.join(path_data, "corpus_classes.xlsx"), index_col=0)
    graphe_cofreq: nx.Graph = nx.Graph()
    threshold: float = 0.05
    ntopics: int = cofrequences.shape[0]
    edges: List[Tuple[int, int]] = [(i, j) for i in range(ntopics) for j in range(i + 1, ntopics)
                                    if cofrequences[i, j] > threshold]
    is_connected: List[bool] = [any([e[0] == i or e[1] == i for e in edges]) for i in range(ntopics)]
    node_labels: List[str] = [top2clas.columns[i] for i in range(ntopics)]
    edges = [e for e in edges if top2clas.columns[e[0]] != "junk" and top2clas.columns[e[1]] != "junk"]
    graphe_cofreq.add_edges_from(edges)
    pos = nx.spring_layout(graphe_cofreq)

    edge_trace: List[go.Scatter] = list()
    for edge in edges:
        print(top2clas.columns[edge[0]], top2clas.columns[edge[1]])
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace.append(go.Scatter(x=(x0, x1), y=(y0, y1), mode='lines', hoverinfo='none',
                                     line=dict(width=2, color="gray")))
    node_x: List[float] = [pos[node][0] for node in graphe_cofreq.nodes]
    node_y: List[float] = [pos[node][1] for node in graphe_cofreq.nodes]
    colornode: List[str] = ["red" if label == "army" else "green" for label in node_labels]
    node_trace = go.Scatter(x=node_x, y=node_y, mode='markers+text', hoverinfo="skip",
                            text=[node_labels[node] for node in graphe_cofreq.nodes],
                            marker=dict(showscale=True, color=[colornode[node] for node in graphe_cofreq.nodes],
                                        size=20, line_width=2),
                            textposition="top right")
    fig = go.Figure(data=edge_trace + [node_trace],
                    layout=go.Layout(
                        title=f"Co-frequency of Topics in Texts",
                        plot_bgcolor='rgba(0,0,0,0)',
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, visible=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, visible=False))
                    )
    fig.write_html(os.path.join(path_data, f"topics_correlation_graphe.html"))
    fig.write_image(os.path.join(path_data, f"topics_correlation_graphe.jpg"), width=1000, height=1000)


def examples_corr():
    top2texts: pd.DataFrame = pd.read_excel(os.path.join(path_data, "corpus_classes.xlsx"), index_col=0).reset_index()
    liste_fichiers: List[str] = os.listdir(os.path.join(path_data, "blocs"))
    top2texts.loc[:, "nom_fichier"] = pd.Series(data=range(top2texts.shape[0])).apply(lambda z: liste_fichiers[z])
    corr_to_army: pd.DataFrame = top2texts.loc[top2texts.army > 0.2, :]
    for classe in [c for c in classes if c not in ["junk", "army"]]:
        if not os.path.isdir(os.path.join(path_data, "corr_army", classe)):
            os.makedirs(os.path.join(path_data, "corr_army", classe))
        local_corr: pd.DataFrame = corr_to_army.loc[corr_to_army.loc[:, classe] > 0.2, ["army", classe, "nom_fichier"]]
        for i, row in local_corr.sample(min(10, local_corr.shape[0])).iterrows():
            shutil.copy(os.path.join(path_data, "blocs", row["nom_fichier"]),
                        os.path.join(path_data, "corr_army", classe, row["nom_fichier"]))


def timeline_corr():
    top2texts: pd.DataFrame = pd.read_excel(os.path.join(path_data, "corpus_classes.xlsx"), index_col=0).reset_index()
    liste_fichiers: List[str] = os.listdir(os.path.join(path_data, "blocs"))
    top2texts.loc[:, "nom_fichier"] = pd.Series(data=range(top2texts.shape[0])).apply(lambda z: liste_fichiers[z])
    corr_to_army: pd.DataFrame = top2texts.loc[top2texts.army > 0.15, :]
    for classe in [c for c in classes if c not in ["junk", "army"]]:
        local_corr: pd.DataFrame = corr_to_army.loc[corr_to_army.loc[:, classe] > 0.15, ["army", classe, "nom_fichier"]]
        local_corr = local_corr.loc[~local_corr.nom_fichier.isna(), :]
        local_corr.loc[:, "year"] = local_corr.nom_fichier.apply(lambda z: int(z.split("-")[0]))
        local_corr.loc[:, "month"] = local_corr.nom_fichier.apply(lambda z: int(z.split("-")[1]))
        local_corr.loc[:, "day"] = local_corr.nom_fichier.apply(lambda z: int(z.split("-")[2]))
        plt.figure()
        plt.bar(range(1881, 1900), [local_corr.loc[local_corr.year == y, :].shape[0] for y in range(1881, 1900)],
                color="lightgray")
        plt.xticks(range(1881, 1900), rotation=90)
        plt.title(f"army & {classe}")
        plt.savefig(os.path.join(path_data, "graphes", f"army & {classe[:6]}.jpg"))


if __name__ == "__main__":
    # build_classes(classes)
    # co_frequences = matrice_corr()
    # draw_graph(co_frequences)
    examples_corr()
    # timeline_corr()
