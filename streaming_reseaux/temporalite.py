from datetime import datetime, timedelta
import os
from typing import List, Dict, Set, Union
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt, patches, colors
import networkx as nx
from sklearn.cluster import AgglomerativeClustering

path_data = os.path.join(os.getcwd(), "data")
path_exports = os.path.join(os.getcwd(), "exports")
list_data_dir: List[str] = ["pixel_war", "corpus_lausanne"]


def data_with_time():
    total: int = len(os.listdir(os.path.join(path_data, "dataframe_comments")))
    for data_dir in list_data_dir:
        auteurs = sorted(list(set([adr.split("__")[1].strip()
                                   for adr in os.listdir(os.path.join(path_data, "dataframe_comments")) if
                                   data_dir in adr])))
        for auteur in auteurs:
            print(auteur)
            subset: List[str] = [adr for adr in os.listdir(os.path.join(path_data, "dataframe_comments"))
                                 if data_dir in adr and adr.split("__")[1].strip() == auteur]
            dataframe: pd.DataFrame = pd.concat([pd.read_csv(os.path.join(path_data, "dataframe_comments", adr),
                                                             sep=";", encoding="utf-8") for adr in subset], axis=0)
            dataframe = dataframe.loc[:, ["date", "time", "author"]].dropna()
            dataframe.loc[:, "streamer"] = auteur
            dataframe.loc[:, "event"] = data_dir
            dataframe.to_csv(os.path.join(path_data, "heure_auteur", f"{data_dir}_{auteur}.csv"), sep=";",
                             encoding="utf-8", index=False)
            total -= len(subset)
    print(total)


def compute_flows() -> pd.DataFrame:
    pixelwar_data: List[str, pd.DataFrame] = [pd.read_csv(os.path.join(path_data, "heure_auteur", adr),
                                                          sep=";", encoding="utf-8", header=0) for adr
                                              in os.listdir(os.path.join(path_data, "heure_auteur"))
                                              if "pixel_war" in adr]
    streamers: List[str] = [df.streamer.unique()[0] for df in pixelwar_data]
    matrix_flow: pd.DataFrame = pd.DataFrame(columns=streamers, index=streamers)
    for df_streamer1 in pixelwar_data:
        viewers1: List[str] = df_streamer1.author.unique()
        streamer1: str = df_streamer1.streamer.unique()[0]
        print(f"From {streamer1}", len(viewers1))
        for df_streamer2 in pixelwar_data:
            streamer2: str = df_streamer2.streamer.unique()[0]
            if streamer2 != streamer1:
                viewers2: List[str] = df_streamer2.author.unique()
                common_viewers: List[str] = list(set(viewers1).intersection(set(viewers2)))
                df_common_str: pd.DataFrame = pd.concat([df_streamer1.loc[df_streamer1.author.isin(common_viewers), :],
                                                         df_streamer2.loc[df_streamer2.author.isin(common_viewers), :]],
                                                        axis=0)
                df_common_str.loc[:, "datetime"] = pd.to_datetime(df_common_str['date'] + ' ' + df_common_str['time'])
                df_common_str.loc[:, "day"] = pd.to_datetime(df_common_str['date'])
                df_common_str = df_common_str.drop(["date", "event", "time"], axis=1)
                df_common_str = df_common_str.sort_values(by=["author", "datetime"])
                df_common_str = df_common_str.drop_duplicates(["author", "streamer", "day"], keep="first")
                flow_to: Set[str] = set()
                prev_author = ""
                prev_streamer = ""
                prev_time: datetime = datetime(1, 1, 1)
                for i, row in df_common_str.iterrows():
                    if i > 0 and row["author"] == prev_author and prev_streamer == streamer1 \
                            and row["streamer"] == streamer2 \
                            and (row["datetime"] - prev_time) < timedelta(hours=24):
                        flow_to.add(prev_author)
                    prev_streamer = row["streamer"]
                    prev_author = row["author"]
                    prev_time = row["datetime"]
                print(f"______to {streamer2}", len(common_viewers), len(flow_to))
                matrix_flow.loc[streamer1, streamer2] = len(flow_to)
            else:
                matrix_flow.loc[streamer1, streamer2] = 0
    print(matrix_flow)
    matrix_flow.to_csv(os.path.join(path_exports, "flots_pixelwar_24H.csv"), sep=";", encoding="utf-8")


def build_digraph():
    matrix = pd.read_csv(os.path.join(path_exports, "flots_pixelwar_24H.csv"), sep=";", encoding="utf-8", index_col=0,
                         header=0)

    streamers: List[str] = matrix.columns
    threshold_edge: int = 400
    graphe_viewers: nx.DiGraph = nx.DiGraph(directed=True)
    graphe_viewers.add_nodes_from(
        [s for s in streamers if matrix.loc[matrix.loc[:, s] > threshold_edge, :].shape[0] > 1])
    graphe_viewers.add_weighted_edges_from([(s1, s2, matrix.loc[s1, s2])
                                            for s1 in streamers for s2 in streamers if s1 != s2
                                            and matrix.loc[s1, s2] > threshold_edge])
    pos = nx.spring_layout(graphe_viewers)
    plt.figure(figsize=(16, 12))
    edge_colors = [matrix.loc[e[0], e[1]] for e in graphe_viewers.edges]
    edges = nx.draw_networkx_edges(
        graphe_viewers,
        pos,
        node_size=40,
        arrowstyle="->",
        arrowsize=16,
        edge_color=edge_colors,
        edge_cmap=plt.cm.Greys,
        width=1,
    )
    nx.draw_networkx_nodes(graphe_viewers, pos, node_size=40, node_color="indigo")
    nx.draw_networkx_labels(graphe_viewers, {p: (pos[p][0], pos[p][1] + 0.1) for p in pos})
    pc = matplotlib.collections.PatchCollection(edges, cmap=plt.cm.Greys)
    pc.set_array(edge_colors)
    plt.colorbar(pc)
    ax = plt.gca()
    ax.set_axis_off()
    plt.savefig(os.path.join(path_exports, "graphes", "viewers_pixel_war_oriented.jpg"))
    plt.close()


def build_sankey():
    emissions: List[str] = [adr for adr in os.listdir(os.path.join(path_data, "dataframe_comments"))
                            if "pixel_war" in adr]
    auteurs: List[str] = sorted(list(set([adr.split("__")[1].strip() for adr in emissions])))
    matrix: pd.DataFrame = pd.read_csv(os.path.join(path_exports, "flots_pixelwar_24H.csv"), sep=";", encoding="utf-8")
    matrix = matrix.set_index("Unnamed: 0")
    print(matrix.info())

    distance: pd.DataFrame = pd.DataFrame(index=auteurs, columns=auteurs)
    for streamer1 in auteurs:
        for streamer2 in auteurs:
            if streamer2 != streamer1:
                distance.loc[streamer1, streamer2] = 1 - 2 * matrix.loc[streamer1, streamer2] \
                                                     / (matrix.loc[streamer1, streamer1]
                                                        + matrix.loc[streamer2, streamer2] + 1)
            else:
                distance.loc[streamer1, streamer2] = 0
    cha: AgglomerativeClustering = AgglomerativeClustering(n_clusters=6)
    cha.fit_predict(distance.values)
    auteurs = sorted(auteurs, key=lambda z: cha.labels_[auteurs.index(z)])

    blocs: List[Dict[str, Union[str, datetime, Set[str]]]] = list()
    for emission in emissions:
        data_emission = {"auteur": emission.split("__")[1].strip()}
        df = pd.read_csv(os.path.join(path_data, "dataframe_comments", emission), sep=";", encoding="utf-8")
        df.loc[:, "datetime"] = pd.to_datetime(df['date'] + ' ' + df['time'])
        df = df.loc[:, ["datetime", "author"]].dropna()
        data_emission["start"] = df.datetime.iloc[0]
        data_emission["end"] = df.datetime.iloc[len(df.datetime) - 1]
        data_emission["viewers"] = set(df.author.unique())
        blocs.append(data_emission)
    blocs.sort(key=lambda z: z["start"])

    threshold = 2000
    h = 0.5
    cmap = plt.cm.tab20
    cNorm = colors.Normalize(vmin=0, vmax=len(auteurs))
    scalarMap = matplotlib.cm.ScalarMappable(norm=cNorm, cmap=cmap)

    fig: plt.Figure = plt.figure(figsize=(15, 12))
    ax = fig.add_subplot()
    plt.plot()
    for i, bloc in enumerate(blocs):
        xs = bloc["start"].timestamp()
        xe = bloc["end"].timestamp()
        y = auteurs.index(bloc["auteur"]) - h / 2
        ax.add_patch(
            patches.Rectangle(
                (xs, y), xe - xs, h,
                edgecolor='blue', facecolor='red', fill=False
            ))
        next24 = [b for b in blocs[i + 1:] if b["start"] - bloc["start"] < timedelta(hours=24)]
        common_viewers: List[int] = [len(list(bloc["viewers"].intersection(further_em["viewers"])))
                                     for further_em in next24]
        for j in range(len(common_viewers)):
            if common_viewers[j] > threshold:
                color = scalarMap.to_rgba(auteurs.index(bloc["auteur"]))
                plt.plot([(xe + xs) / 2,
                          (blocs[i + j + 1]["start"].timestamp() + blocs[i + j + 1]["end"].timestamp()) / 2],
                         [y + h / 2, auteurs.index(blocs[i + j + 1]["auteur"])],
                         color=color, marker="o")

    plt.yticks(range(len(auteurs)), auteurs)
    plt.xticks([datetime(year=2022, month=4, day=i, hour=12).timestamp() for i in range(1, 7)],
               [f"2022-04-0{i}" for i in range(1, 7)])
    plt.savefig(os.path.join(path_exports, "graphes", "pixelwar_flot.jpg"))
    plt.close()


def data_word(words: List[str]):
    all_data: pd.DataFrame = pd.concat([pd.read_csv(os.path.join(path_data, "dataframe_comments", adr), sep=";",
                                                    encoding="utf-8", header=0)
                                        for adr in os.listdir(os.path.join(path_data, "dataframe_comments"))
                                       if "pixel_war" in adr], axis=0)
    all_data = all_data.dropna()
    all_data.loc[:, "datetime"] = pd.to_datetime(all_data['date'] + ' ' + all_data['time'])
    xmin: datetime = all_data.datetime.min()
    xmax: datetime = all_data.datetime.max()
    nb_intervals: int = 50
    x = [xmin + (xmax - xmin) * i / nb_intervals for i in range(nb_intervals)]
    plt.figure(figsize=(15, 10))
    for i, word in enumerate(words):
        print(f"***{word}***")
        relevant_data = all_data.loc[all_data.message.apply(lambda z: word in z)]
        relevant_data = relevant_data.drop(["message", "date", "time"], axis=1)
        y = [relevant_data.loc[(relevant_data.datetime >= dt)
                               & (relevant_data.datetime < dt + (xmax - xmin) / nb_intervals), :].shape[0] for dt in x]
        if i == 0:
            plt.bar(x, y, width=(xmax - xmin) * 0.9 / nb_intervals, linewidth=1, color="red", alpha=0.4)
        else:
            plt.bar(x, y, width=(xmax - xmin) * 0.9 / nb_intervals, linewidth=1, fill=False)
    plt.xticks([datetime(year=2022, month=4, day=i, hour=12) for i in range(1, 7)],
               [f"2022-04-0{i}" for i in range(1, 7)])
    plt.show()


def sankey_word(words: List[str]):
    for word in words:
        emissions: List[str] = [adr for adr in os.listdir(os.path.join(path_data, "dataframe_comments"))
                                if "pixel_war" in adr]
        all_data: pd.DataFrame = pd.concat([pd.read_csv(os.path.join(path_data, "dataframe_comments", adr), sep=";",
                                                        encoding="utf-8", header=0) for adr in emissions], axis=0)
        all_data = all_data.dropna()
        all_data.message = all_data.message.apply(lambda z: z.lower())
        all_data.loc[:, "datetime"] = pd.to_datetime(all_data['date'] + ' ' + all_data['time'])
        auteurs: List[str] = sorted(list(set([adr.split("__")[1].strip() for adr in emissions])))
        all_data.loc[all_data.streamer == "otplol_", "streamer"] = "otplol"

        matrix: pd.DataFrame = pd.read_csv(os.path.join(path_exports, "flots_pixelwar_24H.csv"), sep=";", encoding="utf-8")
        matrix = matrix.set_index("Unnamed: 0")
        print(matrix.info())

        distance: pd.DataFrame = pd.DataFrame(index=auteurs, columns=auteurs)
        for streamer1 in auteurs:
            for streamer2 in auteurs:
                if streamer2 != streamer1:
                    distance.loc[streamer1, streamer2] = 1 - 2 * matrix.loc[streamer1, streamer2] \
                                                         / (matrix.loc[streamer1, streamer1]
                                                            + matrix.loc[streamer2, streamer2] + 1)
                else:
                    distance.loc[streamer1, streamer2] = 0
        cha: AgglomerativeClustering = AgglomerativeClustering(n_clusters=6)
        cha.fit_predict(distance.values)
        auteurs = sorted(auteurs, key=lambda z: cha.labels_[auteurs.index(z)])

        blocs: List[Dict[str, Union[str, datetime, Set[str]]]] = list()
        for emission in emissions:
            data_emission = {"auteur": emission.split("__")[1].strip()}
            df = pd.read_csv(os.path.join(path_data, "dataframe_comments", emission), sep=";", encoding="utf-8")
            df.loc[:, "datetime"] = pd.to_datetime(df['date'] + ' ' + df['time'])
            df = df.loc[:, ["datetime", "author"]].dropna()
            data_emission["start"] = df.datetime.iloc[0]
            data_emission["end"] = df.datetime.iloc[len(df.datetime) - 1]
            data_emission["viewers"] = set(df.author.unique())
            blocs.append(data_emission)
        blocs.sort(key=lambda z: z["start"])

        h = 0.5
        fig: plt.Figure = plt.figure(figsize=(15, 12))
        ax = fig.add_subplot()
        plt.plot()
        for i, bloc in enumerate(blocs):
            xs = bloc["start"].timestamp()
            xe = bloc["end"].timestamp()
            y = auteurs.index(bloc["auteur"]) - h / 2
            ax.add_patch(
                patches.Rectangle(
                    (xs, y), xe - xs, h,
                    edgecolor='blue', facecolor='red', fill=False
                ))
        relevant_data = all_data.loc[all_data.message.apply(lambda z: word in z)]
        relevant_data = relevant_data.drop(["message", "date", "time"], axis=1)

        for i, com in relevant_data.iterrows():
            plt.plot([com["datetime"].timestamp(), com["datetime"].timestamp()],
                     [auteurs.index(com["streamer"]) - h / 2, auteurs.index(com["streamer"]) + h / 2],
                     color="red", marker=None)
        plt.yticks(range(len(auteurs)), auteurs)
        plt.xticks([datetime(year=2022, month=4, day=i, hour=12).timestamp() for i in range(1, 7)],
                   [f"2022-04-0{i}" for i in range(1, 7)])
        plt.savefig(os.path.join(path_exports, "graphes", f"diffusion_pixelwar_{word}.jpg"))
        plt.close()


def sankey_topics(nb_topics: int):
    topic_dist: pd.DataFrame = pd.read_excel(os.path.join(path_exports, "corpus_topics_12.xlsx"), header=0, index_col=0)
    matrix: pd.DataFrame = pd.read_csv(os.path.join(path_exports, "flots_lausanne_24H.csv"), sep=";",
                                       encoding="utf-8")
    matrix = matrix.set_index("Unnamed: 0")
    print(matrix.info())
    emissions: List[str] = sorted([adr for adr in os.listdir(os.path.join(path_data, "dataframe_comments"))
                            if "pixel_war" not in adr])
    print(len(emissions))
    all_data: pd.DataFrame = pd.concat([pd.read_csv(os.path.join(path_data, "dataframe_comments", adr), sep=";",
                                                    encoding="utf-8", header=0) for adr in emissions], axis=0)
    all_data = all_data.dropna()
    all_data.message = all_data.message.apply(lambda z: z.lower())
    all_data.loc[:, "datetime"] = pd.to_datetime(all_data['date'] + ' ' + all_data['time'])
    auteurs: List[str] = sorted(list(set([adr.split("__")[1].strip() for adr in emissions])))
    all_data.loc[all_data.streamer == "otplol_", "streamer"] = "otplol"
    distance: pd.DataFrame = pd.DataFrame(index=auteurs, columns=auteurs)
    for streamer1 in auteurs:
        for streamer2 in auteurs:
            if streamer2 != streamer1:
                distance.loc[streamer1, streamer2] = 1 - 2 * matrix.loc[streamer1, streamer2] \
                                                     / (matrix.loc[streamer1, streamer1]
                                                        + matrix.loc[streamer2, streamer2] + 1)
            else:
                distance.loc[streamer1, streamer2] = 0
    cha: AgglomerativeClustering = AgglomerativeClustering(n_clusters=6)
    cha.fit_predict(distance.values)
    auteurs = sorted(auteurs, key=lambda z: cha.labels_[auteurs.index(z)])

    blocs: List[Dict[str, Union[str, datetime, Set[str], List[float]]]] = list()
    for i, emission in enumerate(emissions):
        data_emission = {"auteur": emission.split("__")[1].strip()}
        df = pd.read_csv(os.path.join(path_data, "dataframe_comments", emission), sep=";", encoding="utf-8")
        df.loc[:, "datetime"] = pd.to_datetime(df['date'] + ' ' + df['time'])
        df = df.loc[:, ["datetime", "author"]].dropna()
        data_emission["start"] = df.datetime.iloc[0]
        data_emission["end"] = df.datetime.iloc[len(df.datetime) - 1]
        data_emission["viewers"] = set(df.author.unique())
        data_emission["topics"] = topic_dist.loc[i, :].to_list()
        blocs.append(data_emission)
    blocs.sort(key=lambda z: z["start"])
    for topic in topic_dist.columns:
        print(f"Topic {topic}")

        h = 0.5
        fig: plt.Figure = plt.figure(figsize=(15, 12))
        ax = fig.add_subplot()
        plt.plot()
        for i, bloc in enumerate(blocs):
            xs = bloc["start"].timestamp()
            xe = bloc["end"].timestamp()
            y = auteurs.index(bloc["auteur"]) - h / 2
            ax.add_patch(
                patches.Rectangle(
                    (xs, y), xe - xs, h,
                    edgecolor='blue', facecolor=(1, 0, 0, bloc["topics"][topic]), fill=True
                ))

        plt.yticks(range(len(auteurs)), auteurs)
        plt.xticks([datetime(year=2022, month=4, day=i, hour=12).timestamp() for i in range(11, 18)],
                   [f"2022-04-1{i}" for i in range(1, 8)])
        plt.savefig(os.path.join(path_exports, "graphes", f"difftopics_nopixelwar_{topic}-{nb_topics}.jpg"))
        plt.close()


if __name__ == "__main__":
    # data_with_time()
    # matrix_flow = compute_flows()
    # build_digraph()
    # build_sankey()
    # data_word(["espagne", "match"])
    # sankey_word(["kcorp", "donde", "bots", "htylaser", "poncefleur", "locklearcringo", "zlan", "trackmania",
    #              "popcorn", "smash", "tournoi", "elden", "macron", "feur", "ramadan"])
    sankey_topics(12)
