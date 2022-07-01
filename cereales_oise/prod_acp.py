from typing import List

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.preprocessing import StandardScaler
import os
from sklearn.decomposition import PCA
path_data = os.getcwd()
titres = {"CEREALES": "% grains", "INDUS": "% industrial crop", "froment/autres": "wheat/grains",
          "jachere": "% fallow",
          "prairies": "% pastures", "vachesh": "cows/ha", "FROMENTHIVERREND1862": "wheat yield",
          "AVOINEREND1862": "oat yield", "METEILREND1862": "meslin yield", "FROMENTHIVERPRIX": "wheat price"}
villes = ["Formerie", "Songeons", "Grandvilliers", "Maignelay", "Ansauvillers", "Crèvecoeur", "Breteuil",
          "Beauvais", "Méru", "Clermont", "Bresles", "Noailles", "Mouy"]
export_rep = "exports_final_article"


def do_acp() -> None:
    data_acp: pd.DataFrame = pd.read_csv(os.path.join(path_data, "pour_acp_2.csv"), sep=";", encoding="utf-8")
    data_acp = data_acp.dropna(axis=1, how="all")
    features: List[str] = [c for c in data_acp.columns if c not in ["NUMCANTON", "NOMCANTON"]]
    for fi in features:
        data_acp.loc[:, fi] = data_acp.loc[:, fi].astype(float)
    print(data_acp.info())
    XC: np.ndarray[float, float] = data_acp.loc[:, features].values
    print(XC.shape)
    scl = StandardScaler()
    XC: np.ndarray[float, float] = scl.fit_transform(XC)
    print(XC.shape)

    pca = PCA(n_components=4)
    XP: np.ndarray[float, float] = pca.fit_transform(XC)
    print(pca.explained_variance_ratio_)
    print(pca.singular_values_)

    loadings: np.ndarray[float] = pca.components_.T * np.sqrt(pca.explained_variance_)

    plt.figure(figsize=(10, 10))
    ax = plt.subplot(111)
    ax.scatter(XP[:, 0], XP[:, 1], edgecolors="none", marker="o", c="black")
    for i, feature in enumerate(features):
        plt.gca().add_line(plt.Line2D((0, loadings[i, 0] * 3), (0, loadings[i, 1] * 3), color="gray"))
        plt.annotate(xy=(loadings[i, 0] * 3, loadings[i, 1] * 3), s=titres[feature])

    for i, row in data_acp.iterrows():
        if row["NOMCANTON"] in villes:
            plt.annotate(xy=(XP[i, 0]+0.1, XP[i, 1]+0.02), s=row["NOMCANTON"])

    ouest = Ellipse((3, 0.5), width=1, height=2, fill=False, edgecolor="black", linewidth=2, linestyle="dashed")
    plt.gca().add_patch(ouest)
    nord = Ellipse((1.6, 0.1), width=1.5, height=0.8, fill=False, edgecolor="black", linewidth=2, linestyle="dotted")
    plt.gca().add_patch(nord)
    sud = Ellipse((0, -1.6), width=2, height=1, fill=False, edgecolor="black", linewidth=2)
    plt.gca().add_patch(sud)
    plt.xlim(-3.8, 4.3)
    plt.savefig(os.path.join(path_data, export_rep, "acp.png"), dpi=300)
    plt.close()


if __name__ == "__main__":
    do_acp()