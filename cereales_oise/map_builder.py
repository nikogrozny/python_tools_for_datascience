import os
from math import sqrt
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from Circles.circles import circle

path_data = os.getcwd()
path_shp: str = "insert_path_to_shapefiles"

all_cer: List[str] = ["froment", "avoine", "méteil", "seigle", "orge"]
fr2en: Dict[str, str] = {"froment": "wheat", "avoine": "oat", "méteil": "maslin", "seigle": "rye", "orge": "barley"}
villes: List[str] = ["Formeries", "Songeons", "Grandvilliers", "Maignelay", "Ansauvillers", "Crèvecoeur", "Breteuil",
                     "Beauvais", "Méru", "Clermont", "Bresles", "Noailles", "Mouy"]
# col_cer: List[str] = ["yellow", "grey", "orange", "red", "darkblue"]
col_cer: List[str] = ['0.2', '0.4', '0.6', '0.8', '1']
communes_avec_pop: str = os.path.join(path_data, 'communes_with_pop.csv')
export_dir: str = "exports_final_article"


def localisation(donnees_brutes: pd.DataFrame) -> pd.DataFrame:
    geo_table: pd.DataFrame = pd.read_csv(communes_avec_pop, encoding="utf-8", sep=";", low_memory=False)
    donnees_localisees: pd.DataFrame = donnees_brutes.merge(geo_table, how="left", left_on="code insee",
                                                            right_on="code_insee")
    donnees_localisees.loc[donnees_localisees.ville == "Maignelay", "longitude"] = 2 + 31 / 60 + 16 / 3600
    donnees_localisees.loc[donnees_localisees.ville == "Maignelay", "latitude"] = 49 + 33 / 60 + 11 / 3600
    print(donnees_localisees.loc[:, ["ville", "code insee", "longitude"]].head(40))
    return donnees_localisees


def draw_map(loc_data: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(15, 10))
    long_min: float = 1.5
    long_max: float = 3.5
    lat_min: float = 49
    lat_max: float = 49.8
    base_map: Basemap = Basemap(llcrnrlon=long_min, llcrnrlat=lat_min, urcrnrlon=long_max, urcrnrlat=lat_max,
                                resolution='h', projection='tmerc', lat_0=49.4, lon_0=2.5, ax=plt.gca())
    base_map.drawmapboundary()
    base_map.drawrivers(color="#5555ee")
    base_map.readshapefile(os.path.join(path_shp, "FRA_adm", 'FRA_adm3'), 'FRA_adm3')

    routes: pd.DataFrame = pd.read_csv(os.path.join(path_data, "routes.csv"), sep=";", encoding="utf-8")
    for r in routes.route.unique():
        route: pd.DataFrame = routes.loc[routes.route == r, ["latitude", "longitude"]]
        xy: Basemap = base_map(route["longitude"].to_list(), route["latitude"].to_list())
        base_map.plot(x=xy[0], y=xy[1], linewidth=1.5, color='black', marker=None, linestyle='dotted')
    xy_toParis: Tuple[float, float] = base_map(2.32, 49.1)
    xy_toAmiens: Tuple[float, float] = base_map(2.23, 49.75)
    plt.text(xy_toAmiens[0], xy_toAmiens[1], "to Amiens", {"fontstyle": "italic"})
    plt.text(xy_toParis[0], xy_toParis[1], "to Paris", {"fontstyle": "italic"})
    xy_ar1_am: Tuple[float, float] = base_map(2.22, 49.75)
    xy_ar2_am: Tuple[float, float] = base_map(2.22, 49.77)
    plt.annotate(xy=xy_ar1_am, s="", xytext=xy_ar2_am, arrowprops=dict(arrowstyle="<-"))
    xy_ar1_pa: Tuple[float, float] = base_map(2.31, 49.1)
    xy_ar2_pa: Tuple[float, float] = base_map(2.31, 49.12)
    plt.annotate(xy=xy_ar1_pa, s="", xytext=xy_ar2_pa, arrowprops=dict(arrowstyle="->"))

    for i, row in loc_data.iterrows():
        xy: Basemap = base_map(row["longitude"], row["latitude"])
        base_map.plot(x=xy[0], y=xy[1], marker=".", color='m', markersize=8, markeredgewidth=2)
        plt.text(xy[0] + sqrt(row["total"]) * 25, xy[1] + 100, row["ville"])
    fig.add_axes([-0.054, -0.08, 1, 1])

    wedges: List = list()
    for i, row in loc_data.iterrows():
        print(row['longitude'], row['latitude'])
        print(base_map(row['longitude'], row['latitude']))
        xy: Tuple[float, float] = (row['longitude'] * 1.45, row['latitude'] * 2.25)
        wedges, texts = plt.pie(row[all_cer], labels=None, radius=sqrt(row["total"]) / 2500, center=xy, colors=col_cer)

    plt.legend(wedges, [fr2en[c] for c in all_cer], loc="center left")
    plt.savefig(os.path.join(path_data, export_dir, "carte_marches.png"), dpi=300)


def draw_recap_map(df: pd.DataFrame) -> None:
    # TODO : changer le fichier cantons_oise
    # car il corrrespond à un découpage moderne qui est différent de celui de l'ACP
    df.loc[:, "seas"] = df.ville.apply(lambda z: "0.2"
    if z in ["Ansauvillers", "Beauvais", "Breteuil", "Clermont", "Mouy", "Noailles"]
    else "0.8" if z in ["Bresles", "Maignelay", "Méru"]
    else "0.4" if z in ["Songeons"]
    else "0.6")

    fig = plt.figure(figsize=(15, 10))
    fig.add_subplot(111)
    patches: List[Polygon] = list()
    long_min: float = 1.5
    long_max: float = 3.5
    lat_min: float = 49
    lat_max: float = 49.8
    base_map: Basemap = Basemap(llcrnrlon=long_min, llcrnrlat=lat_min, urcrnrlon=long_max, urcrnrlat=lat_max,
                                resolution='h', projection='tmerc', lat_0=49.4, lon_0=2.5, ax=plt.gca())
    base_map.drawmapboundary()
    base_map.drawrivers(color="#5555ee")
    base_map.readshapefile(os.path.join(path_shp, "FRA_adm", 'FRA_adm3'), 'FRA_adm3')

    vaches: pd.DataFrame = pd.read_csv(os.path.join(path_data, "pour_acp_2.csv"), sep=";",
                                       encoding="utf-8").dropna(axis=1, how="all")
    vaches = vaches.loc[(vaches.vachesh > 25) & (vaches.prairies > 25), :]
    cantons_vaches: List[str] = vaches.NUMCANTON.unique()
    print(cantons_vaches)
    cantons: pd.DataFrame = pd.read_excel(os.path.join(path_data, "Liste des communes Oise avec canton.xls"))
    communes: pd.DataFrame = cantons.loc[cantons.ID_CANTON.isin(cantons_vaches)]
    for i, row in communes.iterrows():
        polygon: List[str] = row["wkt_geom"][9:-2].split(",")  # syntaxe dépend beaucoup du fichier en question
        local_map: Basemap = base_map([float(z.split(" ")[0]) for z in polygon],
                                      [float(z.split(" ")[1]) for z in polygon])
        xy_arrondissement: np.ndarray[float, float] = np.array([(local_map[0][i], local_map[1][i])
                                                                for i in range(len(local_map[0]))])
        patches.append(Polygon(xy_arrondissement))
    fig.gca().add_collection(PatchCollection(patches, facecolor='white', edgecolor='0.3', linewidths=0, hatch="//",
                                              zorder=2))

    routes: pd.DataFrame = pd.read_csv(os.path.join(path_data, "routes.csv"), sep=";", encoding="utf-8")
    for r in routes.route.unique():
        route: pd.DataFrame = routes.loc[routes.route == r, ["latitude", "longitude"]]
        xy: Basemap = base_map(route["longitude"].to_list(), route["latitude"].to_list())
        base_map.plot(x=xy[0], y=xy[1], linewidth=1.5, color='black', marker=None, linestyle='dotted')

    plt.rc('text', usetex=True)
    for i, row in df.iterrows():
        xy = base_map(row["longitude"], row["latitude"])
        # if row["ville"] in ["Clermont", "Méru", "Ansauvillers"]:
        #     base_map.plot(x=xy[0], y=xy[1], marker=".", color="black", markersize=sqrt(row["total"]) / 5 + 10,
        #                   markeredgewidth=2)
        base_map.plot(x=xy[0], y=xy[1], marker=".", color=row["seas"], markersize=sqrt(row["total"]) / 5,
                      markeredgewidth=2)
        if row["ville"] not in ["Clermont", "Méru", "Ansauvillers"]:
            plt.text(xy[0] + sqrt(row["total"]) * 15, xy[1] + 100, row["ville"])
        else:
            ville = r"\underline{"+row["ville"]+"}"
            plt.text(xy[0] + sqrt(row["total"]) * 15, xy[1] + 100, ville)

    plt.savefig(os.path.join(path_data, export_dir, "carte_recap.png"), dpi=300)


def draw_distances_map(df: pd.DataFrame) -> None:
    marches_dynamiques = ["Beauvais", "Breteuil", "Clermont", "Formeries", "Grandvilliers", "Songeons", "Crèvecoeur"]
    marches_declin = ["Ansauvillers", "Mouy", "Noailles", "Méru"]
    df.loc[:, "tend"] = df.ville.apply(
        lambda z: "0.2" if z in marches_dynamiques else "0.5" if z in marches_declin else "0.8")

    fig = plt.figure(figsize=(15, 10))
    long_min: float = 1.5
    long_max: float = 3.5
    lat_min: float = 49
    lat_max: float = 49.8
    fig.add_subplot(111, frame_on=False)
    base_map: Basemap = Basemap(llcrnrlon=long_min, llcrnrlat=lat_min, urcrnrlon=long_max, urcrnrlat=lat_max,
                                resolution='h', projection='tmerc', lat_0=49.4, lon_0=2.5, ax=plt.gca())
    base_map.drawmapboundary()
    base_map.drawrivers(color="#5555ee")
    base_map.readshapefile(os.path.join(path_shp, "FRA_adm", 'FRA_adm3'), 'FRA_adm3')

    routes: pd.DataFrame = pd.read_csv(os.path.join(path_data, "routes.csv"), sep=";", encoding="utf-8")
    for r in routes.route.unique():
        route: pd.DataFrame = routes.loc[routes.route == r, ["latitude", "longitude"]]
        xy: Basemap = base_map(route["longitude"].to_list(), route["latitude"].to_list())
        base_map.plot(x=xy[0], y=xy[1], linewidth=1.5, color='black', marker=None, linestyle='dotted')

    for i, row in df.iterrows():
        xy: Basemap = base_map(row["longitude"], row["latitude"])
        print(xy)
        base_map.plot(x=xy[0], y=xy[1], marker=".", color=row["tend"], markersize=sqrt(row["total"]) / 5,
                      markeredgewidth=2)
        plt.text(xy[0] + sqrt(row["total"]) * 12, xy[1] + 100, row["ville"])

        if row["ville"] in ["Beauvais", "Clermont", "Breteuil", "Grandvilliers"]:
            radius: float = 15
            casa: List[circle] = list(circle(base_map, row["longitude"], row["latitude"], radius))
            x, y = [z[0] for z in casa], [z[1] for z in casa]
            base_map.plot(x=x, y=y, color="gray")

    outside: Dict[str, Tuple[float, float]] = {"Montdidier": (49.65, 2.566667), "Rollot": (49.583333, 2.65),
                                               "Beaumont": (49.133333, 2.283333)}

    for ville in outside:
        xy: Basemap = base_map(outside[ville][1], outside[ville][0])
        base_map.plot(x=xy[0], y=xy[1], marker="+", color="black", markersize=10, markeredgewidth=2)
        plt.text(xy[0] + 600, xy[1] + 100, ville)
        radius: float = 15
        casa: List[circle] = list(circle(base_map, outside[ville][1], outside[ville][0], radius))
        x, y = [z[0] for z in casa], [z[1] for z in casa]
        base_map.plot(x=x, y=y, color="gray", linestyle='dashed')

    plt.savefig(os.path.join(path_data, export_dir, "carte_distances.png"), dpi=300)


if __name__ == "__main__":
    donnees_sans_loc: pd.DataFrame = pd.read_csv(os.path.join(path_data, "geo_with_loc.csv"),
                                                 sep=";", encoding="utf-8")
    donnees_avec_loc = localisation(donnees_sans_loc)
    draw_map(donnees_avec_loc)
    draw_recap_map(donnees_avec_loc)
    draw_distances_map(donnees_avec_loc)
