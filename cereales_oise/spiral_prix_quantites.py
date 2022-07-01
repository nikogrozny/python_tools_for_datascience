from typing import List, Dict, Tuple
from redecoupage_temps import load_data
from matplotlib import pyplot as plt
import pandas as pd
import os
path_data = os.getcwd()
villes: List[str] = ["Beauvais", "Breteuil", "Clermont", "Songeons", "Grandvilliers"]
cereales: List[str] = ['froment', 'méteil', 'avoine']
year_min: int = 1828
year_max: int = 1852
noms_mois: List[str] = ["JAN", "FEV", "MAR", "AVR", "MAI", "JUN", "JUL", "AOU", "SEP", "OCT", "NOV", "DEC"]

def prix_vs_quantite() -> None:
    prix: Dict[Tuple[str, str], pd.DataFrame] = dict()
    quantites: Dict[Tuple[str, str], pd.DataFrame] = dict()
    quantites_source: pd.DataFrame = load_data("Arrondissement-Clermont-Beauvais_v8.xls",
                                               "ARRONDT CLERMONT BEAUVAIS", 1)
    quantites_source = quantites_source.loc[quantites_source.loc[:, "annee"] <= year_max, :]
    for ville in villes:
        prix_ville = pd.read_excel(os.path.join(path_data, "Copie de RAPPORT DE PRIX.xls"), ville)
        prix_ville = prix_ville.loc[prix_ville.loc[:, "Année"] <= year_max, :]
        for cereale in cereales:
            prix[(ville, cereale)] = prix_ville.loc[:, f"{ville.lower()}{cereale.replace('é', 'e')}px"]
            quantites[(ville, cereale)] = quantites_source.loc[:, f"{ville} {cereale}"]
            plt.figure(figsize=(10, 10))
            for mois in range(12):
                plt.scatter(prix[(ville, cereale)][mois::12], quantites[(ville, cereale)][mois::12],
                            label=noms_mois[mois])
            plt.xlabel("prix")
            plt.ylabel("quantite")
            plt.legend()
            plt.savefig(os.path.join(path_data, "export11", f"prix vs quantite {ville} {cereale}"))


if __name__ == "__main__":
    prix_vs_quantite()