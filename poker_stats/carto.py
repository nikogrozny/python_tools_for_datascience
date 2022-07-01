import os
from typing import Dict, Tuple, List

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from nettoyage import clean_commune

path_data = "insert_path_to_data"
path_geodata = os.getcwd()
communes_avec_pop = os.path.join(path_data, 'temp_files', 'communes_with_pop.csv')
departements_avec_pop = os.path.join(path_data, 'temp_files', 'departements_with_pop.csv')
maps_dir = os.path.join(path_data, "poker_maps")
geojson_dep = os.path.join(path_geodata, "external geodata", "france-geojson", "departements.geojson")


def import_fpt() -> pd.DataFrame:
    df = pd.read_csv(os.path.join(path_data, "poker_data", "FPT_all.csv"), sep="\t", encoding="utf-8")
    df = df.loc[:, ["origine_geo", "residence"]]
    df = clean_commune(df, "residence")
    print(df.info())
    for c in ["origine_geo", "residence"]:
        print(c, list(df.loc[:, c].unique()))
    return df


def localisation(layers: Dict[Tuple[str, str], pd.DataFrame]) -> Dict[Tuple[str, str], pd.DataFrame]:
    geo_table = pd.read_csv(communes_avec_pop, encoding="utf-8", sep=";", low_memory=False).loc[:,
                ["nom_commune_simple", "longitude", "latitude", "population", "numéro_département"]]
    geo_table = geo_table.drop_duplicates(subset=["nom_commune_simple"])
    geo_table.nom_commune_simple = geo_table.nom_commune_simple.apply(str).astype(str)
    geo_layers = dict()
    for layer in layers:
        layers[layer].loc[:, layer[0]] = layers[layer].loc[:, layer[0]].apply(str).astype(str)
        geo_layers[layer] = layers[layer].merge(geo_table, how="left", left_on=layer[0],
                                                right_on="nom_commune_simple").drop("nom_commune_simple", axis=1)
        geo_layers[layer].to_csv(os.path.join(maps_dir, "{}_{}.csv".format(layer[0], layer[1])),
                                 encoding="utf-8", sep=";", index=False)
    return geo_layers


def localisation_dep(layers: Dict[Tuple[str, str], pd.DataFrame]) -> Dict[Tuple[str, str], pd.DataFrame]:
    geo_table = pd.read_csv(departements_avec_pop, encoding="utf-8", sep=";", low_memory=False)
    geo_layers = dict()
    for layer in layers:
        geo_layers[layer] = layers[layer].merge(geo_table, how="left", left_on="numéro_département",
                                                right_on="numero_dep")
        geo_layers[layer].loc[:, "frequence"] = geo_layers[layer].effectif.div(geo_layers[layer].Total)
        geo_layers[layer].to_csv(os.path.join(maps_dir, "{}_{}_dep.csv".format(layer[0], layer[1])),
                                 encoding="utf-8", sep=";", index=False)
    return geo_layers


def population_maps(df: pd.DataFrame, fields: List[str]) -> Dict[Tuple[str, str], pd.DataFrame]:
    layers = dict()
    for field in fields:
        serie = df.loc[~df.loc[:, field].isna() & (df.loc[:, field] != ""), :].groupby(field)[field].count()
        layers[field, "all_players"] = pd.DataFrame({"effectif": serie}).reset_index().sort_values(by="effectif")
    print(layers)
    return layers


def make_maps(df: pd.DataFrame, geo_fields: List[str]) -> Tuple[Dict[Tuple[str, str], pd.DataFrame],
                                                                Dict[Tuple[str, str], pd.DataFrame]]:
    df_layers = population_maps(df, geo_fields)
    geo_layers = localisation(df_layers)

    df_layers_dep = dict()
    for layer in df_layers:
        df_layers_dep[layer] = geo_layers[layer].groupby("numéro_département").sum()
        print(df_layers_dep)
    geo_layers_dep = localisation_dep(df_layers_dep)
    print(geo_layers_dep)
    return geo_layers, geo_layers_dep


def draw_map_dep(layers: Dict[Tuple[str, str], pd.DataFrame]) -> None:
    fig = go.Figure()
    fig.write_image(os.path.join(maps_dir, "yolo.png"))  # quickfix orca
    os.remove(os.path.join(maps_dir, "yolo.png"))
    for layer in layers:
        fig1 = px.choropleth_mapbox(layers[layer], geojson=geojson_dep, locations="nom_dep",
                                    featureidkey="properties.nom", color="effectif", opacity=1,
                                    color_continuous_scale="Reds", range_color=(0, 100), mapbox_style="carto-positron",
                                    zoom=5.8, center={"lat": 46.7, "lon": 1.6},
                                    height=1200, width=1200, labels={"effectif": "effectif"})
        fig2 = px.choropleth_mapbox(layers[layer], geojson=geojson_dep, locations="nom_dep",
                                    featureidkey="properties.nom", color="frequence", opacity=1,
                                    color_continuous_scale="Reds", range_color=(0, 0.0001),
                                    mapbox_style="carto-positron", zoom=5.8, center={"lat": 46.7, "lon": 1.6},
                                    height=1200, width=1200, labels={"effectif": "effectif"})
        fig1.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
        fig1.write_image(os.path.join(maps_dir, "{}_dep.png".format(layer)), scale=4)
        fig2.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
        fig2.write_image(os.path.join(maps_dir, "{}_dep_freq.png".format(layer)), scale=4)


def draw_map(layers: Dict[Tuple[str, str], pd.DataFrame]) -> None:
    fig = go.Figure()
    fig.write_image(os.path.join(maps_dir, "yolo.png"))  # quickfix orca
    os.remove(os.path.join(maps_dir, "yolo.png"))
    fig = go.Figure()
    for layer in layers:
        print(layer)
        fig = px.scatter_mapbox(layers[layer], lat="latitude", lon="longitude",
                                hover_data={"effectif": True, "latitude": False, "longitude": False}, size="effectif",
                                color_discrete_sequence=["red"], zoom=6, height=1200, width=1200,
                                center={"lat": 46.7, "lon": 1.6}, size_max=60, opacity=0.6)
        fig.update_layout(mapbox_style="stamen-toner")
        fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
        fig.write_image(os.path.join(maps_dir, f"{layer[0]}-{layer[1]}.png"), scale=4)


if __name__ == "__main__":
    df_fpt = import_fpt()
    df_res, df_res_dep = make_maps(df_fpt, ["residence"])
    draw_map(df_res)
    draw_map_dep(df_res_dep)
