import os
from typing import List, Tuple, Dict, Callable, Iterable
from matplotlib import pyplot as plt

import pandas as pd

path_data = os.getcwd()
all_cer: List[str] = ["froment", "avoine", "méteil"]
noms_jours: List[str] = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]
year_min, year_max = 1838, 1852
export_rep = "exports_final_article"


def load_data_nord(fichier: str, onglet: str, skip: int, cities: List[str]) -> pd.DataFrame:
    data_echanges_nord = pd.read_excel(os.path.join(path_data, fichier), sheet_name=onglet, skiprows=skip)
    data_echanges_nord = data_echanges_nord.loc[:, [c for c in data_echanges_nord.columns
                                                    if not data_echanges_nord.loc[:, c].dropna(axis=0).empty]]
    data_echanges_nord = data_echanges_nord.rename({"Froment": "Formeries froment", "Méteil": "Formeries méteil",
                                                    "Seigle": "Formeries seigle", "Orge": "Formeries orge",
                                                    "Avoine": "Formeries avoine",
                                                    "Froment.1": "Songeons froment", "Méteil.1": "Songeons méteil",
                                                    "Seigle.1": "Songeons seigle", "Orge.1": "Songeons orge",
                                                    "Avoine.1": "Songeons avoine",
                                                    "Froment.2": "Grandvilliers froment",
                                                    "Méteil.2": "Grandvilliers méteil",
                                                    "Seigle.2": "Grandvilliers seigle", "Orge.2": "Grandvilliers orge",
                                                    "Avoine.2": "Grandvilliers avoine",
                                                    "Froment.3": "Maignelay froment", "Méteil.3": "Maignelay méteil",
                                                    "Seigle.3": "Maignelay seigle", "Orge.3": "Maignelay orge",
                                                    "Avoine.3": "Maignelay avoine",
                                                    "Froment.4": "Ansauvillers froment",
                                                    "Méteil.4": "Ansauvillers méteil",
                                                    "Seigle.4": "Ansauvillers seigle", "Orge.4": "Ansauvillers orge",
                                                    "Avoine.4": "Ansauvillers avoine",
                                                    "Froment.5": "Crèvecoeur froment", "Méteil.5": "Crèvecoeur méteil",
                                                    "Seigle.5": "Crèvecoeur seigle", "Orge.5": "Crèvecoeur orge",
                                                    "Avoine.5": "Crèvecoeur avoine",
                                                    "Froment.6": "Breteuil froment", "Méteil.6": "Breteuil méteil",
                                                    "Seigle.6": "Breteuil seigle", "Orge.6": "Breteuil orge",
                                                    "Avoine.6": "Breteuil avoine",
                                                    }
                                                   , axis=1)
    for ville in cities:
        data_echanges_nord.loc[:, f"{ville} bleds"] = data_echanges_nord.apply(
            lambda z: z[f"{ville} froment"] + z[f"{ville} méteil"], axis=1)
        data_echanges_nord.loc[:, f"{ville} fms"] = data_echanges_nord.apply(
            lambda z: z[f"{ville} bleds"] + z[f"{ville} seigle"], axis=1)
        data_echanges_nord.loc[:, f"{ville} ao"] = data_echanges_nord.apply(
            lambda z: z[f"{ville} avoine"] + z[f"{ville} orge"], axis=1)
    data_echanges_nord.loc[:, "mois"] = data_echanges_nord.Date.apply(lambda x: x.month)
    data_echanges_nord.loc[:, "annee"] = data_echanges_nord.Date.apply(lambda x: x.year - 100)
    print(data_echanges_nord.info())
    print(data_echanges_nord.head())
    return data_echanges_nord


def test_jours(echanges_periode: pd.DataFrame, to_file: str) -> pd.DataFrame:
    echanges_periode: pd.DataFrame = echanges_periode.loc[echanges_periode.annee <= year_max, :]
    nb_jours: Dict[Tuple[int, int, str], int] = \
        {(a, m, j): 0 for j in noms_jours for m in range(1, 13) for a in range(year_min, year_max + 1)}
    jours_par_mois: Callable[[int], int] = \
        lambda z: 29 if z[0] == 2 and z[1] % 4 == 0 else 28 if z[0] == 2 else 30 if z[0] in [4, 6, 9, 11] else 31
    i: int = 0
    date: List[int] = [year_min, 1, 1]
    while date[0] <= year_max:
        i += 1
        if date[2] < jours_par_mois((date[1], date[0])):
            date[2] += 1
        elif date[1] <= 11:
            date[2] = 1
            date[1] += 1
        else:
            date[2], date[1] = 1, 1
            date[0] += 1
        if date[0] <= year_max:
            nb_jours[(date[0], date[1], noms_jours[i % 7])] += 1
        if date[1] == 2 and date[2] == 1:
            print(date[0], noms_jours[i % 7])
    for nj in noms_jours:
        echanges_periode.loc[:, nj] = echanges_periode.apply(lambda z: nb_jours[z["annee"], z["mois"], nj], axis=1)
    echanges_periode.to_csv(os.path.join(path_data, to_file))
    return echanges_periode


def compensation_geo(df: pd.DataFrame, place: str, jours_march: List[Tuple[str, str]], cities: List[str]):
    xtck: List[int] = [(y - year_min) * 12 for y in range(year_min, year_max + 1)]
    for grain in all_cer:
        plt.figure(figsize=(15, 10))
        all_villes: Dict[str, pd.DataFrame] = dict()
        styles = ["-", "--", ":", "-."]
        i = 0
        for (ville, jour) in jours_march:
            all_villes[ville] = df.loc[:, f"{ville} {grain}"].div(df.loc[:, jour])
            plt.plot(df.index, all_villes[ville], label=ville, c="black" if i<4 else "gray", linestyle=styles[i%4])
            i += 1
        plt.legend()
        ymin: int = min([min(all_villes[z]) for z in all_villes])
        ymax: int = max([max(all_villes[z]) for z in all_villes])
        for a in range(year_min, year_max + 1):
            plt.fill_between([(a - year_min) * 12 + 8, (a - year_min) * 12 + 10], [ymin, ymin], [ymax, ymax],
                             color="lightgray", alpha=0.5)
        plt.xticks(xtck, range(year_min, year_max + 1))
        plt.savefig(os.path.join(path_data, export_rep, f"{place}_nb_mark_{grain}.png"), dpi=300)
        plt.close()

        plt.figure(figsize=(15, 10))
        styles = ["-", "--", ":", "-."]
        y_par_marche: Dict[str, pd.DataFrame] = dict()
        y_moy_mob: Dict[str, List[int]] = dict()
        for (ville, jour) in jours_march:
            y_par_marche[ville] = df.loc[:, f"{ville} {grain}"].div(df.loc[:, jour])
            y_moy_mob[ville] = [(sum(y_par_marche[ville][i - 5:i + 6]) + y_par_marche[ville][i - 6] / 2
                                 + y_par_marche[ville][i + 6] / 2) / 12
                                for i in range(6, len(y_par_marche[ville]) - 7)]
        x_moy_mob: Iterable = range(6, len(y_par_marche[cities[0]]) - 7)
        i = 0
        for (ville, jour) in jours_march:
            plt.plot(x_moy_mob, y_moy_mob[ville], label=ville, c="black" if i<4 else "gray", linestyle=styles[i%4])
            i += 1
        ymin: int = min([min(y_par_marche[z]) for z in y_par_marche])
        ymax: int = max([max(y_par_marche[z]) for z in y_par_marche])
        for a in range(year_min, year_max):
            plt.fill_between([(a - year_min) * 12 + 8, (a - year_min) * 12 + 10], [ymin, ymin], [ymax, ymax],
                             color="lightgray", alpha=0.5)
        plt.xticks(xtck, range(year_min, year_max + 1))
        plt.title(f"{place}, {grain} par marché, moyenne mobile")
        plt.legend()
        plt.savefig(os.path.join(path_data, export_rep, f"{place}_mm_nb_mark_{grain}.png"), dpi=300)
        plt.close()


def compensation_cer(df: pd.DataFrame, place: str, jours_march: List[Tuple[str, str]], cities: List[str]):
    xtck: List[int] = [(y - year_min) * 12 for y in range(year_min, year_max + 1)]
    for ville, jour in jours_march:
        plt.figure(figsize=(15, 10))
        all_grains: Dict[str, pd.DataFrame] = dict()
        for i, grain in enumerate(all_cer):
            all_grains[grain] = df.loc[:, f"{ville} {grain}"].div(df.loc[:, jour])
            styles = ["-", "--", ":", "-."]
            plt.plot(df.index, all_grains[grain], label=grain, c="black" if i<4 else "gray", linestyle=styles[i%4])
        plt.legend()
        plt.title(f"{ville}, Céréales par marché")
        ymin: int = min([min(all_grains[z]) for z in all_grains])
        ymax: int = max([max(all_grains[z]) for z in all_grains])
        for a in range(year_min, year_max):
            plt.fill_between([(a - year_min) * 12 + 8, (a - year_min) * 12 + 10], [ymin, ymin], [ymax, ymax],
                             color="lightgray", alpha=0.5)
        plt.xticks(xtck, range(year_min, year_max + 1))
        plt.savefig(os.path.join(path_data, export_rep, f"{place}_nb_mark_{ville}.png"), dpi=300)
        plt.close()

        plt.figure(figsize=(15, 10))
        y_par_marche: Dict[str, pd.DataFrame] = dict()
        y_moy_mob: Dict[str, List[int]] = dict()
        for grain in all_cer:
            y_par_marche[grain] = df.loc[:, f"{ville} {grain}"].div(df.loc[:, jour])
            y_moy_mob[grain] = [(sum(y_par_marche[grain][i - 5:i + 6]) + y_par_marche[grain][i - 6] / 2
                                 + y_par_marche[grain][i + 6] / 2) / 12
                                for i in range(6, len(y_par_marche[grain]) - 7)]
        x_moy_mob = range(6, len(y_par_marche["froment"]) - 7)
        styles = ["-", "--", ":", "-."]
        for i, grain in enumerate(all_cer):
            plt.plot(x_moy_mob, y_moy_mob[grain], label=grain, c="black" if i<4 else "gray", linestyle=styles[i%4])
        plt.legend()
        ymin: int = min([min(y_par_marche[z]) for z in y_par_marche])
        ymax: int = max([max(y_par_marche[z]) for z in y_par_marche])
        for a in range(year_min, year_max):
            plt.fill_between([(a - year_min) * 12 + 8, (a - year_min) * 12 + 10], [ymin, ymin], [ymax, ymax],
                             color="lightgray", alpha=0.5)
        plt.xticks(xtck, range(year_min, year_max + 1))
        plt.title(f"{ville}, Céréales par marché, moyenne mobile")
        plt.savefig(os.path.join(path_data, export_rep, f"{place}_mm_nb_mark_{ville}.png"), dpi=300)
        plt.close()


if __name__ == "__main__":
    villes: List[str] = ["Breteuil", "Formeries", "Maignelay", "Songeons", "Grandvilliers", "Ansauvillers",
                         "Crèvecoeur"]
    jours_marches: List[Tuple[str, str]] = [("Breteuil", "mercredi"), ("Formeries", "mercredi"),
                                            ("Maignelay", "mercredi"),
                                            ("Songeons", "jeudi"), ("Grandvilliers", "lundi"),
                                            ("Ansauvillers", "lundi"),
                                            ("Crèvecoeur", "jeudi")]
    df_arr = load_data_nord("Arrondissement-Clermont-Beauvais_v8.xls", "7 MARCHES NORD DEPT", 8, villes)
    df_test = test_jours(df_arr, "jour_probable_nord.csv")
    compensation_geo(df_test, "Nord", jours_marches, villes)
    compensation_cer(df_test, "Nord", jours_marches, villes)
