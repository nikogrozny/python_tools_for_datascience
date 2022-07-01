from typing import Dict, List
import pandas as pd
from matplotlib import pyplot as plt
import os
from matplotlib.patches import Ellipse
from matplotlib import lines

path_data = os.getcwd()
dest = "topics_par_periode"
nb_runs = 1
nb_topics = 50
words_in_topic = 25
distance_max = 8
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
devs = {
    titres["B1"]: "Bioware", titres["B2"]: "Bioware", titres["CC1"]: "Bethesda Game Studios",
    titres["CC2"]: "Bethesda Game Studios", titres["DD"]: "Red Hook Studios",
    titres["DOS"]: "Larian Studios NV", titres["F1"]: "Interplay Productions",
    titres["F2"]: "Black Isle Studios", titres["F3"]: "Bethesda Game Studios", titres["F4"]: "Bethesda Game Studios",
    titres["FNV"]: "Obsidian Entertainment", titres["PE"]: "Obsidian Entertainment",
    titres["ES3"]: "Bethesda Game Studios", titres["ES4"]: "Bethesda Game Studios",
    titres["ES5"]: "Bethesda Game Studios", titres["TP"]: "Black Isle Studios", titres["TN"]: "inXile Entertainment",
    titres["U7"]: "Origin Systems", titres["U8"]: "Origin Systems", titres["U9"]: "Origin Systems",
    titres["WA2"]: "inXile Entertainment", titres["WI2"]: "CD Projekt RED", titres["WI3"]: "CD Projekt RED"
}
studios = [
    "Bioware", "Interplay Productions",
    "Black Isle Studios", "Larian Studios NV",
    "inXile Entertainment", "Obsidian Entertainment",
    "Bethesda Game Studios", "Origin Systems",
    "Red Hook Studios", "CD Projekt RED"
]
editors = {
    titres["B1"]: [studios[2], studios[1]], titres["B2"]: [studios[2], studios[1]], titres["CC1"]: [studios[6]],
    titres["CC2"]: [studios[6]], titres["DD"]: [studios[8]],
    titres["DOS"]: [studios[3]], titres["F1"]: [studios[1]],
    titres["F2"]: [studios[1]], titres["F3"]: [studios[6]], titres["F4"]: [studios[6]],
    titres["FNV"]: [studios[6]], titres["PE"]: ["Paradox Interactive"],
    titres["ES3"]: [studios[6]], titres["ES4"]: [studios[6]],
    titres["ES5"]: [studios[6], "2K Games"], titres["TP"]: [studios[1]], titres["TN"]: ["Techland Publishing"],
    titres["U7"]: [studios[7]], titres["U8"]: [studios[7]], titres["U9"]: ["Electronic Arts"],
    titres["WA2"]: ["Deep Silver"], titres["WI2"]: [studios[9]], titres["WI3"]: [studios[9]]
}
display_names = {
    titres["B1"]: "BG 1", titres["B2"]: "BG 2", titres["CC1"]: "CCF", titres["CC2"]: "CCES",
    titres["DD"]: "DarkDg", titres["DOS"]: "Divinity", titres["F1"]: "Fallout 1",
    titres["F2"]: "Fallout 2", titres["F3"]: "Fallout 3", titres["F4"]: "Fallout 4",
    titres["FNV"]: "Fallout NV", titres["PE"]: "Pillars", titres["ES3"]: "ES 3", titres["ES4"]: "ES 4",
    titres["ES5"]: "ES 5", titres["TP"]: "Planescape", titres["TN"]: "Tides", titres["U7"]: "Ultima 7",
    titres["U8"]: "Ultima 8", titres["U9"]: "Ultima 8", titres["WA2"]: "Wastld 2",
    titres["WI2"]: "Witcher 2", titres["WI3"]: "Witcher 3"
}
timewords = {
    " crossbow ": -1, " gunpowder ": 0, " pistol ": 1, " rifle ": 1, " nuclear ": 2, " scroll ": -1,
    " paper ": 1, " print ": 1, " car ": 1, " factory ": 1, "full plate": -1, " cotton ": 1,
    " newspaper ": 1, " vaccine ": 1, " phone ": 1, " truck ": 1, " airplane ": 2,
    " bank ": 1, " sewage ": 1, " hospital ": 1, " university ": 0, " computer ": 2, " spaceship ": 3
}
timestops = [1990, 1995, 2000, 2010, 2015, 2020]


def graphe_contraint(rep: str):
    print("Construction des graphes temporels")
    nb_pts_baryctr = 3
    all_words: pd.DataFrame = pd.read_csv(os.path.join(path_data, rep, "series_temp_robust", "all_words.csv"),
                                          sep=";", index_col=0)
    classes_robustes: pd.Index = all_words.columns
    print(f"Nombre de classes robustes: {len(classes_robustes)}")
    distribution_par_classe: Dict[str, pd.DataFrame] = dict()
    for classe_rob in classes_robustes:
        print(f"Classe {classe_rob}")
        distr_topics_jeux: pd.DataFrame = pd.read_csv(os.path.join(path_data, rep, "series_temp",
                                                                   f"cluster_{classe_rob}.csv"), sep=";", index_col=0)
        distr_topics_jeux = distr_topics_jeux.loc[distr_topics_jeux.index != 'Battle Front 2_EN-FR', :].sum(axis=1) \
            .sort_values(ascending=False)
        distribution_par_classe[classe_rob] = distr_topics_jeux.div(distr_topics_jeux.sum()).reset_index()
        distribution_par_classe[classe_rob].game = distribution_par_classe[classe_rob].game.apply(lambda z: z + ".xml")
        print(distribution_par_classe[classe_rob].head())

    devnames: List[str] = list(devs.values())
    devnames_uniques: List[str] = sorted(list(set(devnames)))
    y_dev: Dict[str, int] = {jeu: studios.index(devs[jeu]) for jeu in devs}

    plt.figure(figsize=(15, 10))
    plt.scatter(dates.values(), [devnames_uniques.index(n) for n in devnames], c="blue")
    for jeu in dates:
        plt.annotate(display_names[jeu], xy=(dates[jeu] + 0.1, y_dev[jeu] + 0.1))
    x_classe: Dict[str, float] = dict()
    y_classe: Dict[str, float] = dict()
    for classe_rob in distribution_par_classe:
        x_classe[classe_rob] = sum([dates[distribution_par_classe[classe_rob].loc[i, "game"]]
                                    * distribution_par_classe[classe_rob].loc[i, 0] for i in range(nb_pts_baryctr)]) \
                               / sum([distribution_par_classe[classe_rob].loc[i, 0] for i in range(nb_pts_baryctr)])
        y_classe[classe_rob] = sum([devnames_uniques.index(devs[distribution_par_classe[classe_rob].loc[i, "game"]])
                                    * distribution_par_classe[classe_rob].loc[i, 0] for i in range(nb_pts_baryctr)]) \
                               / sum([distribution_par_classe[classe_rob].loc[i, 0] for i in range(nb_pts_baryctr)])
        plt.annotate(classe_rob, xy=(x_classe[classe_rob] + 0.1, y_classe[classe_rob] + 0.1))
    plt.scatter(x_classe.values(), y_classe.values(), c="red")

    plt.yticks(range(len(devnames_uniques)), labels=devnames_uniques)
    plt.savefig(os.path.join(path_data, rep, "series_temp_robust", f"time_vs_studio.png"))
    plt.close()

    fig, ax = plt.subplots(figsize=(15, 10))
    plt.scatter(dates.values(), [devnames_uniques.index(n) for n in devnames], c="blue")
    for jeu in dates:
        ax.annotate(display_names[jeu], xy=(dates[jeu] + 0.1, y_dev[jeu] + 0.06))
    x_classe, y_classe = dict(), dict()
    for classe_rob in distribution_par_classe:
        x_classe[classe_rob] = sum([dates[distribution_par_classe[classe_rob].loc[i, "game"]]
                                    * distribution_par_classe[classe_rob].loc[i, 0] for i in range(nb_pts_baryctr)]) \
                               / sum([distribution_par_classe[classe_rob].loc[i, 0] for i in range(nb_pts_baryctr)])
        y_classe[classe_rob] = sum([devnames_uniques.index(devs[distribution_par_classe[classe_rob].loc[i, "game"]])
                                    * distribution_par_classe[classe_rob].loc[i, 0] for i in range(nb_pts_baryctr)]) \
                               / sum([distribution_par_classe[classe_rob].loc[i, 0] for i in range(nb_pts_baryctr)])
        ax.annotate(classe_rob, xy=(x_classe[classe_rob] + 0.1, y_classe[classe_rob] + 0.06))
        for gm in distribution_par_classe[classe_rob].loc[distribution_par_classe[classe_rob].loc[:, 0] > 0.1, "game"]:
            line = lines.Line2D((x_classe[classe_rob], dates[gm]),
                                (y_classe[classe_rob], devnames_uniques.index(devs[gm])), lw=1, color='grey', axes=ax)
            ax.add_line(line)
    plt.scatter(x_classe.values(), y_classe.values(), c="red")
    plt.yticks(range(len(devnames_uniques)), labels=devnames_uniques)
    plt.savefig(os.path.join(path_data, rep, "series_temp_robust", f"time_vs_studio_graph.png"))
    plt.close()

    age_diegese: pd.DataFrame = pd.read_csv(os.path.join(path_data, "timer.csv"), sep=";").loc[:, ["game", "score"]]
    print(age_diegese.head())
    decalage = {
        "Baldurs Gate II_EN-FR.xml": (0.1, 0.03), "Baldurs Gate I_EN-FR.xml": (-0.2, 0.05),
        "Creation Club (Fallout 4)_En-FR.xml": (0.1, 0.03), "Creation Club (Skyrim)_En-Fr.xml": (0.1, 0.03),
        "Darkest Dungeon_EN-FR.xml": (0.35, 0.01),
        "Divinity Original Sin 2_EN-FR.xml": (-0.65, -0.08), "Fallout 1_EN-FR.xml": (-1, 0.04),
        "Fallout 2_EN-FR.xml": (0.1, 0.03), "Fallout 3_EN-FR.xml": (-0.5, 0.04),
        "Fallout 4_EN-FR.xml": (0.1, 0.03), "Fallout NV_EN-FR.xml": (-0.8, 0.09),
        "Pillars of Eternity_EN-FR.xml": (0.1, -0.08), "TES Morrowind_EN-FR.xml": (-0.7, 0.03),
        "TES Oblivion_EN-FR.xml": (0.38, -0.025), "TES Skyrim_EN-FR.xml": (-1.3, -0.02),
        "Torment Planetscape_EN-FR.xml": (-1.6, 0.04), "Torment Tides of Numenera_EN-FR.xml": (0.1, 0.03),
        "Ultima 7_EN-FR.xml": (-0.5, 0.04), "Ultima 8_EN-FR.xml": (-0.9, 0.03),
        "Ultima 9_EN-FR.xml": (0.1, -0.08), "Wasteland 2_EN-FR.xml": (0.1, 0.03),
        "Witcher 2_EN-FR.xml": (-1.2, 0.04), "Witcher 3_EN-FR.xml": (-1, 0.1),
        "92": (-0.5, -0.06), "74": (-0.1, -0.06), "6": (-0.1, -0.07), "194": (-0.5, -0.07), "42": (0.15, 0.03),
        "72": (-0.5, -0.06), "48": (-0.5, -0.06), "2": (-0.4, -0.05), "83": (-0.5, -0.06), "0": (-0.4, 0.03),
        "3": (0.22, 0.03), "131": (0.25, 0), "67": (0.25, 0), "124": (-0.9, -0.02), "23": (0, -0.07),
        "119": (0.2, -0.05), "41": (0.15, 0.03), "8": (-0.5, -0.06)
    }
    fig, ax = plt.subplots(figsize=(15, 10))
    plt.scatter(dates.values(), [age_diegese.loc[age_diegese.game == gm, "score"].iloc[0] for gm in dates.keys()],
                c="blue")
    for jeu in dates:
        ax.annotate(display_names[jeu],
                    xy=(dates[jeu] + decalage[jeu][0], age_diegese.loc[age_diegese.game == jeu, "score"].iloc[0] \
                        + decalage[jeu][1]))
    x_classe: Dict[str, float] = dict()
    y_classe: Dict[str, float] = dict()
    for classe_rob in distribution_par_classe:
        x_classe[classe_rob] = sum(
            [dates[distribution_par_classe[classe_rob].loc[i, "game"]] * distribution_par_classe[classe_rob].loc[i, 0]
             for i in range(nb_pts_baryctr)]) \
                               / sum([distribution_par_classe[classe_rob].loc[i, 0] for i in range(nb_pts_baryctr)])
        y_classe[classe_rob] = sum([age_diegese.loc[age_diegese.game == distribution_par_classe[classe_rob].loc[
            i, "game"], "score"].iloc[0] * distribution_par_classe[classe_rob].loc[i, 0]
                                    for i in range(nb_pts_baryctr)]) / sum(
            [distribution_par_classe[classe_rob].loc[i, 0] for i in range(nb_pts_baryctr)])
        ax.annotate(classe_rob, xy=(x_classe[classe_rob] + (decalage[classe_rob][0] if classe_rob in decalage else 0.1),
                                    y_classe[classe_rob] + (
                                        decalage[classe_rob][1] if classe_rob in decalage else 0.03)))
        for gm in distribution_par_classe[classe_rob].loc[distribution_par_classe[classe_rob].loc[:, 0] > 0.1, "game"]:
            line: lines.Line2D = lines.Line2D((x_classe[classe_rob], dates[gm]),
                                              (y_classe[classe_rob],
                                               age_diegese.loc[age_diegese.game == gm, "score"].iloc[0]),
                                              lw=1, color='grey', axes=ax)
            ax.add_line(line)
    plt.scatter(x_classe.values(), y_classe.values(), c="red")
    ax.tick_params(axis=u'both', which=u'both', length=0)
    plt.yticks([-0.9, 0.35, 1.6], ["Dark Ages", "Renaissance", "Futuriste"], rotation=90)
    plt.xlabel("année de parution")
    plt.ylabel("époque dans la diégèse")
    plt.savefig(os.path.join(path_data, rep, "series_temp_robust", f"time_vs_epoch_graph.png"))
    plt.close()

    parts = {"1.1": [[0, 6, 21, 47, 59, 74, 83, 88, 144], [28, 41, 42, 55, 126]],
             "1.2": [[3, 9, 48, 62, 66, 138, 166], [12, 90]],
             "2.1": [[104, 119, 124, 165]],
             "2.2": [[7, 11, 17, 23, 75, 131], [14, 40, 46, 48], [5, 67, 76, 105, 120, 130, 134, 191, 194],
                     [22, 39, 96, 98]],
             "3": [[1, 2, 44, 45, 59, 61, 65, 72, 79, 146, 178], [8, 15, 18, 92, 182]]
             }
    for p in parts:
        fig, ax = plt.subplots(figsize=(15, 10))
        plt.scatter(dates.values(), [age_diegese.loc[age_diegese.game == gm, "score"].iloc[0] for gm in dates.keys()],
                    c="blue")
        for jeu in dates:
            ax.annotate(display_names[jeu],
                        xy=(dates[jeu] + decalage[jeu][0], age_diegese.loc[age_diegese.game == jeu, "score"].iloc[0] \
                            + decalage[jeu][1]))
        for c, superclasse in enumerate(parts[p]):
            x_classe, y_classe = dict(), dict()
            for icl in superclasse:
                classe_rob = str(icl)
                x_classe[classe_rob] = sum([dates[distribution_par_classe[classe_rob].loc[i, "game"]] *
                                            distribution_par_classe[classe_rob].loc[i, 0] for i in
                                            range(nb_pts_baryctr)]) \
                                       / sum(
                    [distribution_par_classe[classe_rob].loc[i, 0] for i in range(nb_pts_baryctr)])
                y_classe[classe_rob] = sum(
                    [age_diegese.loc[
                         age_diegese.game == distribution_par_classe[classe_rob].loc[i, "game"], "score"].iloc[0] *
                     distribution_par_classe[classe_rob].loc[i, 0]
                     for i in range(3)]) / sum(
                    [distribution_par_classe[classe_rob].loc[i, 0] for i in range(nb_pts_baryctr)])
                ax.annotate(classe_rob,
                            xy=(x_classe[classe_rob] + (decalage[classe_rob][0] if classe_rob in decalage else 0.1),
                                y_classe[classe_rob] + (decalage[classe_rob][1] if classe_rob in decalage else 0.03)))
                for gm in distribution_par_classe[classe_rob].loc[
                    distribution_par_classe[classe_rob].loc[:, 0] > 0.1, "game"]:
                    line = lines.Line2D((x_classe[classe_rob], dates[gm]),
                                        (
                                            y_classe[classe_rob],
                                            age_diegese.loc[age_diegese.game == gm, "score"].iloc[0]),
                                        lw=1, color='grey', axes=ax)
                    ax.add_line(line)
                for gm in distribution_par_classe[classe_rob].loc[
                    (distribution_par_classe[classe_rob].loc[:, 0] <= 0.1) & (
                            distribution_par_classe[classe_rob].loc[:, 0] > 0.08), "game"]:
                    line: lines.Line2D = lines.Line2D((x_classe[classe_rob], dates[gm]),
                                                      (
                                                          y_classe[classe_rob],
                                                          age_diegese.loc[age_diegese.game == gm, "score"].iloc[0]),
                                                      lw=1, color='grey', axes=ax, linestyle="dotted")
                    ax.add_line(line)
            plt.scatter(x_classe.values(), y_classe.values(),
                        c="limegreen" if c == 0 else "orange" if c == 1 else "black",
                        s=50)
        ax.tick_params(axis=u'both', which=u'both', length=0)
        plt.yticks([-0.9, 0.35, 1.6], ["Dark Ages", "Renaissance", "Futuriste"], rotation=90)
        plt.xlabel("année de parution")
        plt.ylabel("époque dans la diégèse")
        plt.savefig(os.path.join(path_data, rep, "series_temp_robust", f"time_vs_epoch_graph_{p}.png"))
        plt.close()

    fig, ax = plt.subplots(figsize=(15, 10))
    plt.scatter(dates.values(), [age_diegese.loc[age_diegese.game == gm, "score"].iloc[0] for gm in dates.keys()],
                c="blue")
    for jeu in dates:
        ax.annotate(display_names[jeu],
                    xy=(dates[jeu] + 0.1, age_diegese.loc[age_diegese.game == jeu, "score"].iloc[0] + 0.03))
    x_classe: Dict[str, float] = dict()
    y_classe: Dict[str, float] = dict()
    for classe_rob in distribution_par_classe:
        x_classe[classe_rob] = sum(
            [dates[distribution_par_classe[classe_rob].loc[i, "game"]] * distribution_par_classe[classe_rob].loc[i, 0]
             for i in range(nb_pts_baryctr)]) \
                               / sum([distribution_par_classe[classe_rob].loc[i, 0] for i in range(nb_pts_baryctr)])
        y_classe[classe_rob] = sum([age_diegese.loc[age_diegese.game == distribution_par_classe[classe_rob].loc[
            i, "game"], "score"].iloc[0] * distribution_par_classe[classe_rob].loc[i, 0]
                                    for i in range(3)]) / sum(
            [distribution_par_classe[classe_rob].loc[i, 0] for i in range(nb_pts_baryctr)])
        ax.annotate(classe_rob, xy=(x_classe[classe_rob] + 0.1, y_classe[classe_rob] + 0.03))

    for studio1 in devs:
        for studio3 in [s for s in devs if devs[s] in editors[studio1]
                                           or (bool(set(editors[studio1]) & set(editors[s])) and s > studio1)]:
            line: lines.Line2D = lines.Line2D((dates[studio1], dates[studio3]),
                                              (age_diegese.loc[age_diegese.game == studio1, "score"].iloc[0],
                                               age_diegese.loc[age_diegese.game == studio3, "score"].iloc[0]),
                                              lw=1, color='orange', axes=ax, linestyle="dashed")
            ax.add_line(line)
        for studio2 in [s for s in devs if devs[s] == devs[studio1] and s > studio1]:
            line = lines.Line2D((dates[studio1], dates[studio2]),
                                (age_diegese.loc[age_diegese.game == studio1, "score"].iloc[0],
                                 age_diegese.loc[age_diegese.game == studio2, "score"].iloc[0]),
                                lw=1, color='green', axes=ax)
            ax.add_line(line)

    plt.scatter(x_classe.values(), y_classe.values(), c="red")

    plt.savefig(os.path.join(path_data, rep, "series_temp_robust", f"time_vs_epoch_and_studio.png"))

    fig, ax = plt.subplots(figsize=(15, 10))
    ax.tick_params(axis=u'both', which=u'both', length=0)
    plt.scatter(dates.values(), [y_dev[jeu] for jeu in dates.keys()], c="blue")
    for jeu in dates:
        if jeu in ["Torment Planetscape_EN-FR.xml", "Creation Club (Skyrim)_En-Fr.xml"]:
            ax.annotate(display_names[jeu], xy=(dates[jeu] + 0.15, y_dev[jeu] - 0.16))
        else:
            ax.annotate(display_names[jeu], xy=(dates[jeu] + 0.15, y_dev[jeu] + 0.05))
    mw, mh = 2, 1
    for studio in list(set(devs.values())):
        jeux_du_studio: List[str] = [s for s in devs if devs[s] == studio]
        if len(jeux_du_studio) >= 2:
            minx, maxx = min([dates[jeu] for jeu in jeux_du_studio]), max([dates[jeu] for jeu in jeux_du_studio])
            miny, maxy = min([y_dev[jeu] for jeu in jeux_du_studio]), max([y_dev[jeu] for jeu in jeux_du_studio])
            cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
            wi, he = maxx - minx + mw, mh
            ellipse = Ellipse((cx, cy), width=wi, height=he, fill=False, edgecolor="limegreen", linewidth=2)
            plt.gca().add_patch(ellipse)

    mw, mh = 2.3, 1.3
    for editor in list(set(e for le in editors.values() for e in le)):
        jeux_du_studio = [s for s in editors if editor in editors[s]]
        if len(jeux_du_studio) >= 2:
            minx, maxx = min([dates[jeu] for jeu in jeux_du_studio]), max([dates[jeu] for jeu in jeux_du_studio])
            miny, maxy = min([y_dev[jeu] for jeu in jeux_du_studio]), max([y_dev[jeu] for jeu in jeux_du_studio])
            cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
            if editor == "Bethesda Softworks":
                cy += 0.6
                mh += 0.2
            wi, he = maxx - minx + mw, maxy - miny + mh
            ellipse = Ellipse((cx, cy), width=wi, height=he, fill=False, edgecolor="blue", linewidth=2,
                              linestyle="dashed")
            plt.gca().add_patch(ellipse)

    ellipse: Ellipse = Ellipse((2008.5, 3), width=22, height=4, angle=7, fill=False, edgecolor="red", linewidth=2,
                               linestyle="dotted")
    plt.gca().add_patch(ellipse)
    ellipse: Ellipse = Ellipse((1997.5, 1.45), width=2.5, height=1.6, fill=False, edgecolor="red", linewidth=2,
                               linestyle="dotted")
    plt.gca().add_patch(ellipse)
    plt.yticks([])
    plt.ylim(-1.1, 11.1)
    plt.xlabel("année de parution")
    plt.plot([2000, 2001], [2000, 2001], c="limegreen", label="même studio", linewidth=2)
    plt.plot([2000, 2001], [2000, 2001], c="blue", label="même éditeur", linestyle="dashed", linewidth=2)
    plt.plot([2000, 2001], [2000, 2001], c="red", label="écrivain(s) en commun", linestyle="dotted", linewidth=2)
    plt.legend(loc="center left")
    plt.savefig(os.path.join(path_data, rep, "series_temp_robust", f"studios.png"))
    plt.close()


def dieg_time(rep: str):
    temps_dieg_jeu = pd.DataFrame(columns=list(timewords.keys()))
    for game in [x for x in os.listdir(os.path.join(path_data, rep)) if x != "Battle Front 2_EN-FR.xml"]:
        texte_jeu = open(os.path.join(path_data, rep, game), "r", encoding="utf-8").read()
        temps_dieg_jeu = pd.concat([temps_dieg_jeu, pd.DataFrame({w: texte_jeu.count(w) for w in timewords.keys()},
                                                                 index=[game])], axis=0)
    scores_serie: pd.Series = pd.Series({w: timewords[w] for w in timewords.keys()})
    temps_dieg_jeu.loc[:, "score"] = temps_dieg_jeu.apply(lambda z: z.multiply(scores_serie).sum() / z.sum(), axis=1)
    temps_dieg_jeu.reset_index().rename({"index": "game"}, axis=1).to_csv(os.path.join(path_data, "timer.csv"), sep=";")
