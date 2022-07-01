import os
from typing import List, Tuple
from nord_complet import test_jours, compensation_cer, compensation_geo
import pandas as pd

path_data = os.getcwd()
all_cer: List[str] = ["froment", "avoine", "méteil"]
fr2en = {"froment": "wheat", "avoine": "oat", "méteil": "maslin", "seigle": "rye", "orge": "barley"}
noms_jours: List[str] = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]
year_min, year_max = 1838, 1852


def load_data_sud(fichier: str, onglet: str, skip: int, cities: List[str]) -> pd.DataFrame:
    data_echanges_sud: pd.DataFrame = pd.read_excel(os.path.join(path_data, fichier), sheet_name=onglet, skiprows=skip)
    data_echanges_sud = data_echanges_sud.loc[:,
                        [c for c in data_echanges_sud.columns if not data_echanges_sud.loc[:, c].dropna(axis=0).empty]]
    data_echanges_sud = data_echanges_sud.rename({"Froment": "Beauvais froment", "Méteil": "Beauvais méteil",
                                                  "Seigle": "Beauvais seigle", "Orge": "Beauvais orge",
                                                  "Avoine": "Beauvais avoine",
                                                  "Froment.1": "Méru froment", "Méteil.1": "Méru méteil",
                                                  "Seigle.1": "Méru seigle", "Orge.1": "Méru orge",
                                                  "Avoine.1": "Méru avoine",
                                                  "Froment.2": "Clermont froment", "Méteil.2": "Clermont méteil",
                                                  "Seigle.2": "Clermont seigle", "Orge.2": "Clermont orge",
                                                  "Avoine.2": "Clermont avoine",
                                                  "Froment.3": "Bresles froment", "Méteil.3": "Bresles méteil",
                                                  "Seigle.3": "Bresles seigle", "Orge.3": "Bresles orge",
                                                  "Avoine.3": "Bresles avoine",
                                                  "Froment.4": "Noailles froment", "Méteil.4": "Noailles méteil",
                                                  "Seigle.4": "Noailles seigle", "Orge.4": "Noailles orge",
                                                  "Avoine.4": "Noailles avoine",
                                                  "Froment.5": "Mouy froment", "Méteil.5": "Mouy méteil",
                                                  "Seigle.5": "Mouy seigle", "Orge.5": "Mouy orge",
                                                  "Avoine.5": "Mouy avoine"
                                                  }
                                                 , axis=1)
    for ville in cities:
        data_echanges_sud.loc[:, f"{ville} bleds"] = data_echanges_sud.apply(
            lambda z: z[f"{ville} froment"] + z[f"{ville} méteil"], axis=1)
        data_echanges_sud.loc[:, f"{ville} fms"] = data_echanges_sud.apply(
            lambda z: z[f"{ville} bleds"] + z[f"{ville} seigle"], axis=1)
        data_echanges_sud.loc[:, f"{ville} ao"] = data_echanges_sud.apply(
            lambda z: z[f"{ville} avoine"] + z[f"{ville} orge"], axis=1)
    data_echanges_sud.loc[:, "mois"] = data_echanges_sud.Date.apply(lambda x: x.month)
    data_echanges_sud.loc[:, "annee"] = data_echanges_sud.Date.apply(lambda x: x.year - 100)
    print(data_echanges_sud.info())
    print(data_echanges_sud.head())
    return data_echanges_sud


if __name__ == "__main__":
    villes = ["Beauvais", "Méru", "Clermont", "Bresles", "Noailles", "Mouy"]
    jours_marches: List[Tuple[str, str]] = [("Beauvais", "samedi"), ("Méru", "vendredi"), ("Clermont", "samedi"),
                                            ("Bresles", "jeudi"), ("Noailles", "lundi"), ("Mouy", "samedi")]
    df_arr = load_data_sud("Arrondissement-Clermont-Beauvais_v8.xls", "6 MARCHES SUD DEPT", 8, villes)
    df_test = test_jours(df_arr, "jour_probable_sud.csv")
    compensation_geo(df_test, "Sud", jours_marches, villes)
    compensation_cer(df_test, "Sud", jours_marches, villes)
