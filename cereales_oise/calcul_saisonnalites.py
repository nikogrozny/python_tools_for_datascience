import os
from typing import List, Dict, Tuple, Iterable
from matplotlib import pyplot as plt
import pylab
import scipy.stats as stats
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from mlinsights.mlmodel import PiecewiseRegressor

path_data = os.getcwd()
all_cer: List[str] = ["froment", "avoine", "méteil"]
fr2en: Dict[str, str] = {"froment": "wheat", "avoine": "oat", "méteil": "maslin", "seigle": "rye", "orge": "barley"}
villes: List[str] = ["Breteuil", "Clermont", "Beauvais", "Songeons", "Grandvilliers"]
villes_small: List[str] = ["Formeries", "Maignelay", "Ansauvillers", "Crèvecoeur", "Méru", "Bresles", "Noailles",
                           "Mouy"]
jours_marches: List[Tuple[str, str]] = [("Breteuil", "mercredi"), ("Clermont", "samedi"), ("Songeons", "jeudi"),
                                        ("Beauvais", "samedi"), ("Grandvilliers", "lundi")]
export_rep = "exports_final_article"


def load_data(fil: str) -> pd.DataFrame:
    donnees_echanges: pd.DataFrame = pd.read_csv(os.path.join(path_data, fil), sep=",", encoding="utf-8", index_col=0)
    donnees_echanges = donnees_echanges.loc[:,
                       [c for c in donnees_echanges.columns if not donnees_echanges.loc[:, c].dropna(axis=0).empty]]
    repartition: Dict[str, List[int]] = dict()
    for ville, jour in jours_marches:
        plt.figure()
        vraies_cereales: List[str] = ["avoine", "froment", "méteil"]
        repartition[ville] = [donnees_echanges.loc[:, f"{ville} {c}"].sum() for c in vraies_cereales]
        plt.bar(vraies_cereales, height=repartition[ville])
        plt.title(f"Répartition {ville}")
        plt.savefig(os.path.join(path_data, export_rep, f"_{ville}_répartition.png"))
        plt.close()
    return donnees_echanges


def decompose_withmm(df: pd.DataFrame, grain: str) -> None:
    print(grain)
    year_min, year_inter, year_max = 1828, 1838, 1852
    x: np.ndarray[int, int] = df.apply(lambda z: z["annee"] + z["mois"] / 12, axis=1).values.reshape(-1, 1)
    plt.figure(figsize=(15, 10))
    y_par_marche: Dict[Tuple[str, str], List[int]] = dict()
    for ville, jour in jours_marches:
        df.loc[:, f"{ville} {jour} {grain}"] = df.apply(lambda z: z[f"{ville} {grain}"] / z[f"{jour}"], axis=1)
        y_par_marche[(ville, jour)] = list(df.loc[:, f"{ville} {jour} {grain}"])
        plt.plot(x, y_par_marche[(ville, jour)], label=f"{ville}_{jour}")
    ymin: int = min([min(y_par_marche[z]) for z in y_par_marche])
    ymax: int = max([max(y_par_marche[z]) for z in y_par_marche])
    plt.ylim(ymin, ymax)
    xtck: Iterable = range(year_min, year_max + 1)
    plt.xticks(xtck)
    for annee in range(year_min, year_max + 1):
        plt.fill_between([annee + 9 / 12, annee + 11 / 12], [ymin, ymin], [ymax, ymax], color="lightgray", alpha=0.5)
    plt.title(f"Raw amount per market day : {grain}")
    plt.legend()
    plt.savefig(os.path.join(path_data, export_rep, f"par_marche_brut_{grain}.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(15, 10))
    y_par_marche: Dict[Tuple[str, str], List[int]] = dict()
    saisonnalite: Dict[Tuple[str, str], List[float]] = dict()
    estimateur_saisonnalite: Dict[Tuple[str, str], List[float]] = dict()
    for ville, jour in jours_marches:
        y_par_marche[(ville, jour)] = list(df.loc[:, f"{ville} {jour} {grain}"])
        nb_annees: int = len(x) // 12
        y_moy_mob: List[float] = [(sum(y_par_marche[(ville, jour)][i - 5:i + 6])
                                   + y_par_marche[(ville, jour)][i - 6] / 2
                                   + y_par_marche[(ville, jour)][i + 6] / 2) / 12
                                  for i in range(6, len(y_par_marche[(ville, jour)]) - 7)]
        saisonnalite[(ville, jour)] = [y_par_marche[(ville, jour)][i + 6] - y_moy_mob[i] for i in
                                       range(len(y_par_marche[(ville, jour)]) - 13)]
        sais_est: List[float] = \
            [sum(saisonnalite[(ville, jour)][i + 12 * a] for a in range(nb_annees - 1)) / (nb_annees - 1)
             for i in range(6)] + [sum(saisonnalite[(ville, jour)][i + 12 * (a - 1)]
                                       for a in range(nb_annees - 1)) / (nb_annees - 1) for i in range(6, 12)]
        sais_est = [x - sum(sais_est) / 12 for x in sais_est]
        print(ville, jour, sais_est)
        estimateur_saisonnalite[(ville, jour)] = sais_est[6:12] + sais_est[:6]
        print(ville, jour, [min(estimateur_saisonnalite[z]) for z in estimateur_saisonnalite])
        plt.plot(x[6:-7], [saisonnalite[(ville, jour)][i] / np.std(y_par_marche[(ville, jour)]) for i in
                           range(len(y_par_marche[(ville, jour)]) - 13)], label=ville)
    ymin: float = min([min(saisonnalite[z] / np.std(y_par_marche[z])) for z in saisonnalite])
    ymax: float = max([max(saisonnalite[z] / np.std(y_par_marche[z])) for z in saisonnalite])
    plt.ylim(ymin, ymax)
    plt.xticks(xtck)
    for annee in range(year_min, year_max + 1):
        plt.fill_between([annee + 9 / 12, annee + 11 / 12], [ymin, ymin], [ymax, ymax], color="lightgray", alpha=0.5)
    plt.legend()
    plt.title(f"Ecart à la moyenne mobile ramenée à l'écart type: {grain}")
    plt.savefig(os.path.join(path_data, export_rep, f"par_marche_e0std1_{grain}.png"), dpi=300)
    plt.close()

    for ville, jour in jours_marches:
        plt.figure(figsize=(15, 10))
        y_moy_mob: List[float] = [(sum(y_par_marche[(ville, jour)][i - 5:i + 6])
                                   + y_par_marche[(ville, jour)][i - 6] / 2
                                   + y_par_marche[(ville, jour)][i + 6] / 2) / 12
                                  for i in range(6, len(y_par_marche[(ville, jour)]) - 7)]
        plt.plot(x, y_par_marche[(ville, jour)])
        plt.plot(x[6:-7], y_moy_mob, linestyle="-", c="black")
        plt.plot(x[6:-7], [estimateur_saisonnalite[(ville, jour)][i % 12] for i in
                           range(6, len(y_par_marche[(ville, jour)]) - 7)], c="black", linestyle="--")
        plt.plot(x[6:-7], [y_par_marche[(ville, jour)][i] - estimateur_saisonnalite[(ville, jour)][i % 12]
                           for i in range(6, len(y_par_marche[(ville, jour)]) - 7)], c="black", linestyle=":")
        print(ville, jour, [min(estimateur_saisonnalite[z]) for z in estimateur_saisonnalite])
        ymin: float = min([min(estimateur_saisonnalite[z]) for z in estimateur_saisonnalite])
        ymax: float = max(y_par_marche[(ville, jour)])
        for annee in range(year_min, year_max):
            plt.fill_between([annee + 9 / 12, annee + 11 / 12], [ymin, ymin], [ymax, ymax], color="lightgrey", alpha=0.5)
        plt.xticks(xtck)
        plt.ylim(ymin, ymax)
        plt.title(f"Moyenne mobile et écart à la saisonnalité {grain} {ville} {jour}")
        plt.savefig(os.path.join(path_data, export_rep, f"mm_ecartsais_{grain}_{ville}_{jour}.png"), dpi=300)
        plt.close()

    for ville, jour in jours_marches:
        residus_local: List[float] = [
            y_par_marche[(ville, jour)][i] - sum(y_par_marche[(ville, jour)]) / len(y_par_marche[(ville, jour)]) -
            estimateur_saisonnalite[(ville, jour)][i % 12]
            for i in range(len(y_par_marche[(ville, jour)]))]

        plt.figure()
        plt.acorr(np.array(residus_local))
        plt.title(f"Autocorrelation of residuals {fr2en[grain]} {ville} ")
        plt.savefig(os.path.join(path_data, export_rep, f"res_acorr_{grain}_{ville}_{jour}.png"), dpi=300)
        plt.close()

        plt.figure()
        plt.hist(np.array(residus_local))
        plt.title(f"Histogram of residuals {fr2en[grain]} {ville} ")
        plt.savefig(os.path.join(path_data, export_rep, f"res_hist_{grain}_{ville}_{jour}.png"), dpi=300)
        plt.close()

        plt.figure()
        plt.scatter(residus_local[:-1], residus_local[1:])
        plt.title(f"Plot residuals n vs n+1 {fr2en[grain]} {ville}")
        plt.savefig(os.path.join(path_data, export_rep, f"res_plotnn1_{grain}_{ville}_{jour}.png"), dpi=300)
        plt.close()

        plt.figure()
        stats.probplot(np.array(residus_local), dist="norm", plot=pylab)
        plt.title(f"QQplot residuals {grain} {ville}")
        plt.savefig(os.path.join(path_data, export_rep, f"mm_qqplot_{grain}_{ville}_{jour}.png"), dpi=300)
        plt.close()


def decompose_withreg(all_echanges: pd.DataFrame, grain: str):
    # -> Tuple[Dict[Tuple[str, str], np.ndarray[float]],
    #                                                                    Dict[Tuple[str, str], List[float]]]:
    print(grain)
    year_min, year_inter, year_max = 1828, 1838, 1852
    y_par_marche: Dict[Tuple[str, str], np.ndarray[float, float]] = dict()
    ypred: Dict[Tuple[str, str], np.ndarray[float]] = dict()
    saisonnalite: Dict[Tuple[str, str], List[float]] = dict()
    estimateur_saisonnalite: Dict[Tuple[str, str], List[float]] = dict()

    for ville, jour in [(v, j) for v, j in jours_marches if (v, grain) != ("Méru", "avoine")]:
        print(ville)
        echanges_sur_periode: pd.DataFrame = all_echanges.loc[all_echanges.annee >= year_inter, :] \
            if ville in villes_small else all_echanges.copy()
        year_min_local: int = year_inter if ville in villes_small else year_min
        x: np.ndarray[int] = \
            echanges_sur_periode.apply(lambda z: z["annee"] + z["mois"] / 12, axis=1).values.reshape(-1, 1)
        xtck: Iterable[int] = range(year_min_local, year_max + 1)

        plt.figure(figsize=(15, 10))
        echanges_sur_periode.loc[:, f"{ville} {jour} {grain}"] \
            = echanges_sur_periode.apply(lambda z: z[f"{ville} {grain}"] / z[f"{jour}"], axis=1)
        y_par_marche[(ville, jour)] = np.array(list(echanges_sur_periode.loc[:, f"{ville} {jour} {grain}"]))
        y_moy_mob: List[float] = \
            [(sum(y_par_marche[(ville, jour)][i - 5:i + 6]) + y_par_marche[(ville, jour)][i - 6] / 2
              + y_par_marche[(ville, jour)][i + 6] / 2) / 12
             for i in range(6, len(y_par_marche[(ville, jour)]) - 7)]

        regression_model: PiecewiseRegressor = \
            PiecewiseRegressor(verbose=True, binner=DecisionTreeRegressor(max_leaf_nodes=4, min_samples_leaf=36))
        regression_model.fit(x, y_par_marche[(ville, jour)])
        ypred[(ville, jour)] = regression_model.predict(x)
        saisonnalite[(ville, jour)] = \
            [y_par_marche[(ville, jour)][i] - ypred[(ville, jour)][i] for i in range(len(y_par_marche[(ville, jour)]))]
        sais_est: List[float] = [sum(saisonnalite[(ville, jour)][i + 12 * a] for a in range(year_max - year_min_local))
                                 / (year_max - year_min_local) for i in range(12)]
        estimateur_saisonnalite[(ville, jour)] = [x - sum(sais_est) / 12 for x in sais_est]

        plt.plot(x, y_par_marche[(ville, jour)], c="black", linestyle="-")
        plt.plot(x[6:-7], y_moy_mob, c="black", linestyle="--")
        plt.plot(x, ypred[(ville, jour)], c="black", linestyle=":")
        plt.xticks(xtck)
        plt.title(f"Moyenne mobile et modèle linéaire {grain} {ville} {jour}")
        ymin: float = min(y_par_marche[(ville, jour)])
        ymax: float = max(y_par_marche[(ville, jour)])
        for a in range(year_min_local, year_max):
            plt.fill_between([a + 9 / 12, a + 11 / 12], [ymin, ymin], [ymax, ymax], color="lightgrey", alpha=0.5)
        plt.ylim(ymin, ymax)
        plt.savefig(os.path.join(path_data, export_rep, f"ml_mm_{grain}_{ville}_{jour}.png"), dpi=300)
        plt.close()

        plt.figure(figsize=(15, 10))
        plt.plot(x, y_par_marche[(ville, jour)], c="black", linestyle="-", label="raw data")
        plt.plot(x, ypred[(ville, jour)], c="black", linestyle="--", label="trend")
        plt.plot(x, [estimateur_saisonnalite[(ville, jour)][i % 12] for i in range(len(x))], label="seasonality",
                 linestyle=":", c="black",)
        plt.xticks(xtck)
        plt.title(f"Modèle linéaire et saisonnalité {grain} {ville} {jour}")
        ymin: float = min(estimateur_saisonnalite[(ville, jour)])
        ymax: float = max(y_par_marche[(ville, jour)])
        for a in range(year_min_local, year_max):
            plt.fill_between([a + 9 / 12, a + 11 / 12], [ymin, ymin], [ymax, ymax], color="lightgray", alpha=0.5)
        plt.ylim(ymin, ymax)
        plt.legend()
        plt.savefig(os.path.join(path_data, export_rep, f"ml_sais_{grain}_{ville}_{jour}.png"), dpi=300)
        plt.close()

        residus: List[float] = [y_par_marche[(ville, jour)][i] - ypred[(ville, jour)][i]
                                - estimateur_saisonnalite[(ville, jour)][i % 12]
                                for i in range(len(y_par_marche[(ville, jour)]))]

        plt.figure()
        plt.acorr(np.array(residus))
        plt.title(f"Autocorrelation of residuals {fr2en[grain]} {ville}")
        plt.savefig(os.path.join(path_data, export_rep, f"ml_res_acorr_{grain}_{ville}_{jour}.png"), dpi=300)
        plt.close()

        plt.figure()
        plt.hist(np.array(residus))
        plt.title(f"Histogram of residuals {fr2en[grain]} {ville}")
        plt.savefig(os.path.join(path_data, export_rep, f"ml_res_hist_{grain}_{ville}_{jour}.png"), dpi=300)
        plt.close()

        plt.figure()
        plt.scatter(residus[:-1], residus[1:])
        plt.title(f"Plot residuals n vs n+1 {fr2en[grain]} {ville}")
        plt.savefig(os.path.join(path_data, export_rep, f"ml_res_plotnn1_{grain}_{ville}_{jour}.png"), dpi=300)
        plt.close()

        plt.figure()
        stats.probplot(np.array(residus), dist="norm", plot=pylab)
        plt.title(f"QQplot residuals {fr2en[grain]} {ville}")
        plt.savefig(os.path.join(path_data, export_rep, f"ml_res_qqplot_{grain}_{ville}_{jour}.png"), dpi=300)
        plt.close()

    return ypred, estimateur_saisonnalite


def froment_nord(all_echanges: pd.DataFrame) -> None:
    year_min, year_max = 1838, 1852
    nord_sauf_grdvlr: List[str] = ["Breteuil", "Songeons", "Formeries", "Ansauvillers", "Crèvecoeur", "Maignelay"]
    echanges_periode: pd.DataFrame = all_echanges.loc[all_echanges.annee >= year_min, :]
    x: np.ndarray[int] = echanges_periode.apply(lambda z: z["annee"] + z["mois"] / 12, axis=1).values.reshape(-1, 1)
    xtck: Iterable[int] = range(year_min, year_max + 1)

    plt.figure(figsize=(15, 10))
    for (ville, jour) in [w for w in jours_marches if w[0] in nord_sauf_grdvlr]:
        echanges_periode.loc[:, f"{ville} froment"] = echanges_periode.apply(
            lambda z: z[f"{ville} froment"] / z[f"{jour}"], axis=1)
    echanges_periode.loc[:, "nord_sauf_gv"] = echanges_periode.loc[:, [f"{vi} froment" for vi in nord_sauf_grdvlr]].sum(
        axis=1)

    y_par_marche: np.ndarray[float] = np.array(list(echanges_periode.loc[:, "nord_sauf_gv"]))
    y_moy_mob: List[float] = [(sum(y_par_marche[i - 5:i + 6]) + y_par_marche[i - 6] / 2 + y_par_marche[i + 6] / 2) / 12
                              for i in range(6, len(y_par_marche) - 7)]

    regression_model: PiecewiseRegressor = PiecewiseRegressor(
        verbose=True, binner=DecisionTreeRegressor(max_leaf_nodes=4, min_samples_leaf=36))
    regression_model.fit(x, y_par_marche)
    y_predit: np.ndarray[float] = regression_model.predict(x)
    saisonnalite: List[float] = [y_par_marche[i] - y_predit[i] for i in range(len(y_par_marche))]
    pre_estim_sais: List[float] = [sum(saisonnalite[i + 12 * a] for a in range(year_max - year_min))
                                   / (year_max - year_min) for i in range(12)]
    estimateur_saisonnalite: List[float] = [x - sum(pre_estim_sais) / 12 for x in pre_estim_sais]

    plt.plot(x, y_par_marche, c="black", linestyle="-")
    plt.plot(x[6:-7], y_moy_mob, c="black", linestyle="--")
    plt.plot(x, y_predit, c="black", linestyle=":")
    plt.xticks(xtck)
    plt.title(f"Moyenne mobile et modèle linéaire froment nord sauf Grandvilliers")
    ymin, ymax = min(y_par_marche), max(y_par_marche)
    for a in range(year_min, year_max):
        plt.fill_between([a + 9 / 12, a + 11 / 12], [ymin, ymin], [ymax, ymax], color="lightgray", alpha=0.5)
    plt.ylim(ymin, ymax)
    plt.savefig(os.path.join(path_data, export_rep, f"ml_mm_froment_nord_sauf_gv.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(15, 10))
    plt.plot(x, y_par_marche, c="black", linestyle="-")
    plt.plot(x, y_predit, c="black", linestyle="--")
    plt.plot(x, [estimateur_saisonnalite[i % 12] for i in range(len(x))], c="black", linestyle=":")
    plt.xticks(xtck)
    plt.title(f"Modèle linéaire et saisonnalité froment nord sauf Grandvilliers")
    ymin: float = min(estimateur_saisonnalite)
    ymax: float = max(y_par_marche)
    for a in range(year_min, year_max):
        plt.fill_between([a + 9 / 12, a + 11 / 12], [ymin, ymin], [ymax, ymax], color="lightgray", alpha=0.5)
    plt.ylim(ymin, ymax)
    plt.savefig(os.path.join(path_data, export_rep, f"ml_sais_froment_nord_sauf_gv.png"), dpi=300)
    plt.close()


def double_saisonnalite(echanges: pd.DataFrame, ville: str, jour: str, grain: str, annee_cesure: int):
    year_min, year_max = 1828, 1852
    ech_avt_cesure: pd.DataFrame = echanges.loc[echanges.annee < annee_cesure, :]
    ech_apr_cesure: pd.DataFrame = echanges.loc[echanges.annee >= annee_cesure, :]
    x_avant: np.ndarray[float] = ech_avt_cesure.apply(lambda z: z["annee"] + z["mois"] / 12, axis=1).values.reshape(-1,
                                                                                                                    1)
    x_apres: np.ndarray[float] = ech_apr_cesure.apply(lambda z: z["annee"] + z["mois"] / 12, axis=1).values.reshape(-1,
                                                                                                                    1)
    xtick_avant: Iterable[int] = range(year_min, annee_cesure)
    xtick_apres: Iterable[int] = range(annee_cesure, year_max + 1)
    y_par_marche: List[np.ndarray[float]] = list()
    saisonnalite: List[List[float]] = list()
    estimateur_saisonnalite: List[List[float]] = list()

    for ix, echanges, x, xtck, year_start, year_end in \
            [(0, ech_avt_cesure, x_avant, xtick_avant, year_min, annee_cesure),
             (1, ech_apr_cesure, x_apres, xtick_apres, annee_cesure, year_max)]:
        plt.figure(figsize=(15, 10))
        echanges.loc[:, f"{ville} {jour} {grain}"] = echanges.apply(lambda z: z[f"{ville} {grain}"] / z[f"{jour}"],
                                                                    axis=1)
        y_par_marche.append(np.array(list(echanges.loc[:, f"{ville} {jour} {grain}"])))
        y_moy_mob = [(sum(y_par_marche[ix][i - 5:i + 6]) + y_par_marche[ix][i - 6] / 2
                      + y_par_marche[ix][i + 6] / 2) / 12
                     for i in range(6, len(y_par_marche[ix]) - 7)]
        regression_model: PiecewiseRegressor = \
            PiecewiseRegressor(verbose=True, binner=DecisionTreeRegressor(max_leaf_nodes=4, min_samples_leaf=36))
        regression_model.fit(x, y_par_marche[ix])
        y_predit: np.ndarray[float] = regression_model.predict(x)
        saisonnalite.append([y_par_marche[ix][i] - y_predit[i] for i in range(len(y_par_marche[ix]))])
        sais_est: List[float] = [sum(saisonnalite[ix][i + 12 * a] for a in range(year_end - year_start))
                                 / (year_end - year_start)
                                 for i in range(12)]
        estimateur_saisonnalite.append([x - sum(sais_est) / 12 for x in sais_est])
        plt.plot(x, y_par_marche[ix], c="black", linestyle="-")
        plt.plot(x[6:-7], y_moy_mob, c="black", linestyle="--")
        plt.plot(x, y_predit, c="black", linestyle=":")
        plt.xticks(xtck)
        plt.title(f"Moyenne mobile et modèle linéaire {grain} {ville} {jour}")
        ymin: float = min(y_par_marche[ix])
        ymax: float = max(y_par_marche[ix])
        for a in range(year_start, year_end):
            plt.fill_between([a + 9 / 12, a + 11 / 12], [ymin, ymin], [ymax, ymax], color="lightgray", alpha=0.5)
        plt.ylim(ymin, ymax)
        plt.savefig(os.path.join(path_data, export_rep, f"div_{ix}_ml_mm_{grain}_{ville}_{jour}.png"), dpi=300)
        plt.close()
        plt.figure(figsize=(15, 10))
        plt.plot(x, y_par_marche[ix], c="black", linestyle="-")
        plt.plot(x, y_predit, c="black", linestyle="--")
        plt.plot(x, [estimateur_saisonnalite[ix][i % 12] for i in range(len(x))], c="black", linestyle=":")
        plt.xticks(xtck)
        plt.title(f"Modèle linéaire et saisonnalité {grain} {ville} {jour}")
        ymin: float = min([estimateur_saisonnalite[ix][i % 12] for i in range(len(x))])
        ymax: float = max(y_par_marche[ix])
        for a in range(year_start, year_end):
            plt.fill_between([a + 9 / 12, a + 11 / 12], [ymin, ymin], [ymax, ymax], color="lightgray", alpha=0.5)
        plt.ylim(ymin, ymax)
        plt.savefig(os.path.join(path_data, export_rep, f"div_{ix}_ml_sais_{grain}_{ville}_{jour}.png"), dpi=300)
        plt.close()
        res_local = [y_par_marche[ix][i] - y_predit[i] - estimateur_saisonnalite[ix][i % 12]
                     for i in range(len(y_par_marche[ix]))]
        plt.figure()
        plt.acorr(np.array(res_local))
        plt.title(f"Autocorrelation of residuals {fr2en[grain]} {ville}")
        plt.savefig(os.path.join(path_data, export_rep, f"div_{ix}_ml_res_acorr_{grain}_{ville}_{jour}.png"), dpi=300)
        plt.close()
        plt.figure()
        plt.hist(np.array(res_local))
        plt.title(f"Histogram of residuals {fr2en[grain]} {ville}")
        plt.savefig(os.path.join(path_data, export_rep, f"div_{ix}_ml_res_hist_{grain}_{ville}_{jour}.png"), dpi=300)
        plt.close()
        plt.figure()
        plt.scatter(res_local[:-1], res_local[1:])
        plt.title(f"Plot residuals n vs n+1 {fr2en[grain]} {ville}")
        plt.savefig(os.path.join(path_data, export_rep, f"div_{ix}_ml_res_plotnn1_{grain}_{ville}_{jour}.png"), dpi=300)
        plt.close()
        plt.figure()
        stats.probplot(np.array(res_local), dist="norm", plot=pylab)
        plt.title(f"QQplot residuals {fr2en[grain]} {ville}")
        plt.savefig(os.path.join(path_data, export_rep, f"div_{ix}_ml_res_qqplot_{grain}_{ville}_{jour}.png"), dpi=300)
        plt.close()

        plt.figure(figsize=(15, 10))
        plt.boxplot([echanges.loc[echanges.mois == m, f"{ville} {grain}"] for m in range(1, 13)], notch=True)
        plt.title(f"Dispersion mensuelle totale {grain} {ville}")
        plt.savefig(os.path.join(path_data, export_rep, f"div_{ix}_plot_brut_{grain}_{ville}.png"), dpi=300)
        plt.close()

        plt.figure(figsize=(15, 10))
        plt.boxplot([echanges.loc[echanges.mois == m, f"{ville} {grain}"].div(echanges.loc[echanges.mois == m, jour])
                     for m in range(1, 13)], notch=True)
        plt.title(f"Dispersion mensuelle par jour de marché {grain} {ville} {jour}")
        plt.savefig(os.path.join(path_data, export_rep, f"div_{ix}_plot_par_marche_{grain}_{ville}_{jour}.png"), dpi=300)
        plt.close()


def ratio_saisonnalite(echanges: pd.DataFrame, modele, saisonnalite_est) -> None:
    for grain in all_cer:
        ratio: Dict[str, List[float]] = dict()
        year_min: int = 1828
        year_max: int = 1852
        lissage: int = 0
        annees: Iterable[int] = range(year_min, year_max + 1)

        for ville in villes:
            colonne_grain: str = f"{ville} {grain}"
            signal: pd.DataFrame = echanges.loc[:, ["annee", colonne_grain]]
            max_annee = [signal.loc[signal.annee.isin([year - lissage, year + lissage]),
                                    colonne_grain].max() for year in annees]
            min_annee = [signal.loc[signal.annee.isin([year - lissage, year + lissage]),
                                    colonne_grain].min() for year in annees]
            moy_annee = [signal.loc[signal.annee.isin([year - lissage, year + lissage]),
                                    colonne_grain].mean() for year in annees]
            ratio[ville] = [(max_annee[i] - min_annee[i]) / moy_annee[i] for i in range(year_max - year_min + 1)]
        plt.figure(figsize=(12, 10))
        styles = ["-", "--", ":", "-."]
        for i, ville in enumerate(villes):
            plt.plot(annees, ratio[ville], label=ville, c="black" if i<4 else "gray", linestyle=styles[i%4])
        plt.legend()
        plt.title(f"Ratio between yearly variations and yearly average ({grain})")
        plt.savefig(os.path.join(path_data, "export11", f"ratio_amplitude_moyenne_{grain}"), dpi=300)

    ratio_saison: Dict[Tuple[str, str], float] = dict()
    for cer in all_cer:
        for ville, jour in jours_marches:
            moyenne_modele: float = modele[cer][(ville, jour)].mean()
            max_saison: float = max(saisonnalite_est[cer][(ville, jour)])
            ratio_saison[(cer, ville)] = max_saison / moyenne_modele
    plt.figure(figsize=(12, 10))
    colors = ["1.", "0.8", "0.6"]
    ix: int = 0
    for ville, jour in jours_marches:
        jx: int = 0
        for cer in all_cer:
            if ix == 0:
                plt.bar(ix + jx * 0.3, ratio_saison[(cer, ville)], 0.3, color=colors[jx], edgecolor="black",
                        label=fr2en[cer])
            else:
                plt.bar(ix + jx * 0.3, ratio_saison[(cer, ville)], 0.3, color=colors[jx], edgecolor="black")
            jx += 1
        ix += 1
    plt.xticks([x + 0.3 for x in range(len(jours_marches))], [ville_jour[0] for ville_jour in jours_marches])
    plt.legend()
    plt.title("Ratio between seasonal variation and global average")
    plt.savefig(os.path.join(path_data, "export11", f"ratio_amplitude_moyenne_tous"), dpi=300)


if __name__ == "__main__":
    df_arr = load_data("jour_probable.csv")
    modele_lineaire = dict()
    estimateur_saison = dict()
    for ce in all_cer:
        # decompose_withmm(df_arr, ce)
        signal_decompose = decompose_withreg(df_arr, ce)
        modele_lineaire[ce], estimateur_saison[ce] = signal_decompose[0], signal_decompose[1]
        # froment_nord(df_arr)
    ratio_saisonnalite(df_arr, modele_lineaire, estimateur_saison)
    # double_saisonnalite(df_arr, "Clermont", "samedi", "méteil", 1845)
