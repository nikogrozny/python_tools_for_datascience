import os
from typing import List, Tuple, Dict, Callable
from matplotlib import pyplot as plt
import pandas as pd
from nord_complet import load_data_nord
from sud_complet import load_data_sud

path_data = os.getcwd()
cereales_only: List[str] = ["froment", "avoine", "méteil"]
villes: List[str] = ["Breteuil", "Clermont", "Beauvais", "Songeons", "Grandvilliers"]
villes_nord: List[str] = ["Formeries", "Maignelay", "Ansauvillers", "Crèvecoeur"]
villes_sud: List[str] = ["Méru", "Bresles", "Noailles", "Mouy"]
jours_marches: List[Tuple[str, str]] = [("Breteuil", "mercredi"), ("Clermont", "samedi"), ("Songeons", "samedi"),
                                        ("Songeons", "jeudi"), ("Beauvais", "samedi"), ("Grandvilliers", "lundi")]
year_min, year_max = 1828, 1852
exports_rep = "exports_final_article"


def load_data(fichier_data: str, onglet: str, skip: int) -> pd.DataFrame:
    donnees_echanges: pd.DataFrame = pd.read_excel(os.path.join(path_data, fichier_data),
                                                   sheet_name=onglet, skiprows=skip)
    donnees_echanges = donnees_echanges.loc[:, [c for c in donnees_echanges.columns
                                                if not donnees_echanges.loc[:, c].dropna(axis=0).empty]]
    donnees_echanges = donnees_echanges.rename(
        {"DATE": "Date", "NB marchés": "Breteuil marchés", "NB marchés.1": "Clermont marchés",
         "NB marchés.2": "Beauvais marchés", "NB marchés.3": "Songeons marchés",
         "NB marchés.4": "Grandvilliers marchés", "Méteil Songeons": "Songeons méteil",
         "Froment Songeons": "Songeons froment",
         "Méteil Beauvais": "Beauvais méteil", "Froment Beauvais": "Beauvais froment"},
        axis=1)
    nb_marches: pd.DataFrame = pd.read_excel(os.path.join(path_data, fichier_data), "nb de marchés")
    for ville in villes:
        donnees_echanges.loc[:, f"{ville} bleds"] = donnees_echanges.apply(
            lambda z: z[f"{ville} froment"] + z[f"{ville} méteil"], axis=1)
        donnees_echanges.loc[:, f"{ville} fms"] = donnees_echanges.apply(
            lambda z: z[f"{ville} bleds"] + z[f"{ville} seigle"], axis=1)
        donnees_echanges.loc[:, f"{ville} ao"] = donnees_echanges.apply(
            lambda z: z[f"{ville} avoine"] + z[f"{ville} orge"], axis=1)
        donnees_echanges.loc[:, f"{ville} marchés"] = nb_marches.loc[nb_marches.loc[:, "Marché"] == ville, :].groupby(
            ["Année", "Mois"]).NOMBREDEMARCHES.sum().reset_index().NOMBREDEMARCHES
    donnees_echanges.loc[:, "mois"] = donnees_echanges.Date.apply(lambda x: x.month)
    donnees_echanges.loc[:, "annee"] = donnees_echanges.Date.apply(lambda x: x.year - 100)
    donnees_echanges = donnees_echanges.loc[:,
                       ['annee', 'mois'] + [f"{ville} {champs}" for ville in villes for champs in
                                            cereales_only + ["marchés"]]]
    return donnees_echanges


def test_jours(data_echanges: List[pd.DataFrame]) -> pd.DataFrame:
    noms_jours: List[str] = ["mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche", "lundi"]
    echanges_et_dates = data_echanges[0].merge(data_echanges[1], how="left",
                                               left_on=["mois", "annee"], right_on=["mois", "annee"])
    echanges_et_dates = echanges_et_dates.merge(data_echanges[2], how="left",
                                                left_on=["mois", "annee"], right_on=["mois", "annee"])
    print(echanges_et_dates.info())
    echanges_et_dates = echanges_et_dates.loc[echanges_et_dates.annee < year_max, :]
    nb_jours: Dict[Tuple[int, int, str], int] = {(a, m, j): 0 for j in noms_jours
                                                 for m in range(1, 13) for a in range(year_min, year_max + 1)}
    jours_par_mois: Callable[[int], int] = \
        lambda z: 29 if z[0] == 2 and z[1] % 4 == 0 else 28 if z[0] == 2 else 30 if z[0] in [4, 6, 9, 11] else 31
    i: int = 0
    d: List[int, int, int] = [year_min, 1, 1]
    while d[0] <= year_max:
        i += 1
        if d[2] < jours_par_mois((d[1], d[0])):
            d[2] += 1
        elif d[1] <= 11:
            d[2] = 1
            d[1] += 1
        else:
            d[2], d[1] = 1, 1
            d[0] += 1
        if d[0] <= year_max:
            nb_jours[(d[0], d[1], noms_jours[i % 7])] += 1
        if d[1] == 2 and d[2] == 1:
            print(d[0], noms_jours[i % 7])
    for nj in noms_jours:
        echanges_et_dates.loc[:, nj] = echanges_et_dates.apply(lambda z: nb_jours[z["annee"], z["mois"], nj], axis=1)
    echanges_et_dates.to_csv(os.path.join(path_data, "jour_probable.csv"))

    echanges_debut = echanges_et_dates.loc[echanges_et_dates.annee <= 1845, :]  # série des marchés pas complète
    for ville in villes:
        print(ville)
        distance: Dict[Tuple[str, str], int] = dict()
        if ville != "Beauvais":
            for nj in noms_jours:
                distance[nj] = echanges_debut.apply(lambda z: abs(z[f"{ville} marchés"] - z[nj]), axis=1).sum()
            jours_marches_pb = [nj for nj in noms_jours if distance[nj] == min([distance[j] for j in noms_jours])][0]
            print(jours_marches_pb)
            print(echanges_debut.loc[echanges_et_dates.apply(lambda z: abs(z[f"{ville} marchés"] - z[jours_marches_pb]),
                                                             axis=1) != 0, ["annee", "mois"]].head(40))
        else:
            for j1 in noms_jours:
                for j2 in noms_jours:
                    distance[(j1, j2)] = echanges_debut.apply(lambda z: abs(z[f"{ville} marchés"] - z[j1] - z[j2]),
                                                              axis=1).sum()
            jours_marches_pb = [(j1, j2) for j1 in noms_jours
                                for j2 in noms_jours if distance[(j1, j2)] == min([distance[(j1, j2)]
                                                                                   for j1 in noms_jours for j2 in
                                                                                   noms_jours])][0]
            print(jours_marches_pb)
            print(echanges_debut.loc[echanges_et_dates.apply(
                lambda z: abs(z[f"{ville} marchés"] - z[jours_marches_pb[0]] - z[jours_marches_pb[1]]),
                axis=1) != 0, ["annee", "mois"]].head(40))
    return echanges_et_dates


def quantite_par_semaine(echanges_marche: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    quantites_hebdo: Dict[str, pd.DataFrame] = dict()
    for ville, jour in jours_marches:
        quantites_hebdo[ville] = pd.DataFrame(columns=["annee", "mois"] + [f"{ville} {grain}" for grain in cereales_only])
        for i, row_echange in echanges_marche.iterrows():
            nb_semaines: int = int(row_echange[f"{jour}"])
            for semaine in range(nb_semaines):
                week_row = row_echange[["annee", "mois"] + [f"{ville} {grain}" for grain in cereales_only]]
                for cereale in cereales_only:
                    week_row[f"{ville} {cereale}"] = week_row[f"{ville} {cereale}"] / nb_semaines
                quantites_hebdo[ville] = quantites_hebdo[ville].append(week_row)
        quantites_hebdo[ville].to_csv(os.path.join(path_data, "new_cal", f"{ville}_{jour}.csv"), sep=";",
                                      encoding="utf-8", index=False)
    return quantites_hebdo


def comp_stats(ville: str, jour: str, echanges: pd.DataFrame, echanges_semaine: pd.DataFrame) -> None:
    nb_marches_mois = 4
    nb_marches_mois_alt = 8 if ville == "Beauvais" else 4
    xtck: List[int] = [(y - year_min) * 12 for y in range(year_min, year_max + 2)]
    echanges_par_4semaines = pd.DataFrame(columns=[f"{ville} {gr}" for gr in cereales_only])
    for i in range(echanges_semaine.shape[0] // nb_marches_mois):
        echanges_par_4semaines = \
            echanges_par_4semaines.append(echanges_semaine.iloc[
                                          range(nb_marches_mois * i, nb_marches_mois * i + nb_marches_mois), :
                                          ].sum(axis=0), ignore_index=True)
    for grain in cereales_only:
        plt.figure(figsize=(15, 10))
        plt.plot(echanges_par_4semaines.index, echanges_par_4semaines.loc[:, f"{ville} {grain}"], label="par 4 semaines")
        jours_par_mois: Callable[[int], int] = lambda z: 29 if z[0] == 2 and z[1] % 4 == 0 \
            else 28 if z[0] == 2 else 30 if z[0] in [4, 6, 9, 11] else 31
        plt.plot([sum([jours_par_mois((j % 12 + 1, j // 12)) for j in range(m)]) / 28 for m in echanges.index],
                 echanges.loc[:, f"{ville} {grain}"], label="par mois")

        plt.plot([sum([jours_par_mois((j % 12 + 1, j // 12)) for j in range(m)]) / 28 for m in echanges.index],
                 echanges.loc[:, f"{ville} {grain}"].div(echanges.loc[:, f"{ville} marchés"]) * nb_marches_mois_alt, label="par marché")

        ymin = echanges.loc[:, f"{ville} {grain}"].min()
        ymax = echanges.loc[:, f"{ville} {grain}"].max()
        for a in range(year_min, year_max + 1):
            plt.fill_between([(a - year_min) * 12 + 8, (a - year_min) * 12 + 10], [ymin, ymin], [ymax, ymax],
                             color="yellow",
                             alpha=0.5)
        plt.xticks(xtck, range(year_min, year_max + 2))
        plt.legend()
        plt.savefig(os.path.join(path_data, exports_rep, f"({ville}_{jour}) {grain}.png"))
        plt.close()


def beauvais_markets(echanges: pd.DataFrame):
    for grain in cereales_only:
        plt.figure(figsize=(15, 10))
        plt.plot(echanges.index, echanges.loc[:, f"Beauvais {grain}"].div(echanges.loc[:, f"Beauvais marchés"]) * 2,
                 label="together", c="black", linestyle=":")
        plt.plot(echanges.index, echanges.loc[:, f"Beauvais {grain}"].div(echanges.loc[:, f"mercredi"]),
                 label="wednesday", c="gray", linestyle="-")
        plt.plot(echanges.index, echanges.loc[:, f"Beauvais {grain}"].div(echanges.loc[:, f"samedi"]),
                 label="saturday", c="black", linestyle="-")
        ymin: float = echanges.loc[:, f"Beauvais {grain}"].div(echanges.loc[:, f"samedi"]).min()
        ymax: float = echanges.loc[:, f"Beauvais {grain}"].div(echanges.loc[:, f"samedi"]).max()
        # for a in range(year_min, year_max + 1):
        #     plt.fill_between([(a - year_min) * 12 + 8, (a - year_min) * 12 + 10], [ymin, ymin], [ymax, ymax], color="yellow",
        #                      alpha=0.5)
        xtck: List[int] = [(y - year_min) * 12 for y in range(year_min, year_max + 2)]
        plt.xticks(xtck, range(year_min, year_max + 2))
        plt.legend()
        plt.savefig(os.path.join(path_data, exports_rep, f"Beauvais_nb_mark {grain}.png"), dpi=300)
        plt.close()


def songeons_markets(echanges: pd.DataFrame):
    for grain in cereales_only:
        plt.figure(figsize=(15, 10))
        plt.plot(echanges.index, echanges.loc[:, f"Songeons {grain}"].div(echanges.loc[:, f"Songeons marchés"]),
                 label="officiels")
        plt.plot(echanges.index, echanges.loc[:, f"Songeons {grain}"].div(echanges.loc[:, f"jeudi"]), label="jeudi")
        plt.plot(echanges.index, echanges.loc[:, f"Songeons {grain}"].div(echanges.loc[:, f"samedi"]), label="samedi")
        ymin = echanges.loc[:, f"Songeons {grain}"].div(echanges.loc[:, f"samedi"]).min()
        ymax = echanges.loc[:, f"Songeons {grain}"].div(echanges.loc[:, f"samedi"]).max()
        for a in range(year_min, year_max + 1):
            plt.fill_between([(a - year_min) * 12 + 8, (a - year_min) * 12 + 10], [ymin, ymin], [ymax, ymax],
                             color="yellow",
                             alpha=0.5)
        xtck: List[int] = [(y - year_min) * 12 for y in range(year_min, year_max + 2)]
        plt.xticks(xtck, range(year_min, year_max + 2))
        plt.legend()
        plt.savefig(os.path.join(path_data, exports_rep, f"Songeons_nb_mark {grain}.png"))
        plt.close()


def breteuil_markets(echanges: pd.DataFrame):
    for grain in cereales_only:
        plt.figure(figsize=(15, 10))
        plt.plot(echanges.index, echanges.loc[:, f"Breteuil {grain}"].div(echanges.loc[:, f"Breteuil marchés"]),
                 label="ensemble")
        plt.plot(echanges.index, echanges.loc[:, f"Breteuil {grain}"].div(echanges.loc[:, f"mercredi"]),
                 label="mercredi")
        ymin = echanges.loc[:, f"Breteuil {grain}"].div(echanges.loc[:, f"mercredi"]).min()
        ymax = echanges.loc[:, f"Breteuil {grain}"].div(echanges.loc[:, f"mercredi"]).max()
        for a in range(year_min, year_max + 1):
            plt.fill_between([(a - year_min) * 12 + 8, (a - year_min) * 12 + 10], [ymin, ymin], [ymax, ymax],
                             color="yellow",
                             alpha=0.5)
        xtck: List[int] = [(y - year_min) * 12 for y in range(year_min, year_max + 2)]
        plt.xticks(xtck, range(year_min, year_max + 2))
        plt.legend()
        plt.savefig(os.path.join(path_data, exports_rep, f"Breteuil_nb_mark {grain}.png"))
        plt.close()


if __name__ == "__main__":
    df_arr: pd.DataFrame = load_data("Arrondissement-Clermont-Beauvais_v8.xls", "ARRONDT CLERMONT BEAUVAIS", 1)
    df_nord: pd.DataFrame = \
        load_data_nord("Arrondissement-Clermont-Beauvais_v8.xls", "7 MARCHES NORD DEPT", 8, villes_nord)
    df_nord = df_nord.loc[:, ['annee', 'mois'] + [f"{ville} {champs}" for ville in villes_nord
                                                  for champs in cereales_only]]
    df_sud: pd.DataFrame = \
        load_data_sud("Arrondissement-Clermont-Beauvais_v8.xls", "6 MARCHES SUD DEPT", 8, villes_sud)
    df_sud = df_sud.loc[:, ['annee', 'mois'] + [f"{ville} {champs}" for ville in villes_sud
                                                for champs in cereales_only]]
    df_test: pd.DataFrame = test_jours([df_arr, df_nord, df_sud])
    # df_cal2: pd.DataFrame = quantite_par_semaine(df_test)
    for ville_marche, jour_marche in jours_marches:
        pass
        # data_par_semaine = pd.read_csv(os.path.join(path_data, "new_cal", f"{ville_marche}_{jour_marche}.csv"),
        #                                sep=";", encoding="utf-8")
        # comp_stats(ville_marche, jour_marche, df_test, data_par_semaine)
    beauvais_markets(df_test)
    # breteuil_markets(df_test)
    # songeons_markets(df_test)
