from typing import Dict, Tuple
import pandas as pd
import os
from plotly import graph_objects as go

path_data = os.getcwd()
socio_fields = ["code_sexe", "code_profession", "code_diplome", 'code_emploi', "code_age_1"]
poker_fields = ["code_somme_pni_v1", "code_somme_pnl_v1", "code_nbj_l_v1", "code_somme_p1_v1", "how_decouvert",
                "1partie_1", "code_anciennete_1"]
extra_fields = ["depuis1", "depuis2", "depuis3", "depuis4", "depuis5",
                "depuis6", "depuis7"]
recode = {**{
    "code_somme_pni_v1": {'Moins de 5E': '<5', 'Entre 10 et 20E': '10-20', 'Entre 5 et 10E': '5-10',
                          'Non réponse': 'Non réponse', 'Plus de 50E': '>50', 'Entre 20 et 50E': '20-50',
                          'Aucune': 'Aucune'},
    "code_somme_pnl_v1": {'Moins de 5E': '<5', 'Entre 10 et 20E': '10-20', 'Entre 5 et 10E': '5-10',
                          'Non réponse': 'Non réponse', 'Plus de 50E': '>50', 'Entre 20 et 50E': '20-50',
                          'Aucune': 'Aucune'},
    "code_somme_p1_v1": {'Moins de 5E': '<5', 'Entre 10 et 20E': '10-20', 'Entre 5 et 10E': '5-10',
                         'Non réponse': 'Non réponse', 'Plus de 50E': '>50', 'Entre 20 et 50E': '20-50',
                         'Aucune': 'Aucune'},
    "code_nbj_l_v1": {'2 fois': "2", '3 fois': "3", 'Tous les jours': "7", '4 fois': "4", 'Non réponse': 'Non réponse',
                      'Non': "0", '6 fois': "6", '5 fois': "5", '1 fois': "1"},
    "code_reg_v1": {'Pratique on-line régulière': "reg", 'Pratique on-line quotidienne': "quot",
                    'Pratique on-line très régulière': 'trs reg', '0': '0', 'Pratique on-line irrégulière': 'irr'},
    "code_profession": {'Etudiants, apprentis, stagiaires, sans emploi': "ETU", 'Ouvriers': "OUV",
                        'Professions intermédiaires': "INT", 'Employés': "EMP",
                        'Cadres et professions intellectuelles supérieures': "CAD",
                        "Artisans, commerçants et chef d'entreprises": "ART", '0': 'Non renseigné',
                        'Non classés': 'Non classés',
                        'Agriculteurs exploitants': "AGR"},
    "code_diplome": {'BAC+4 et plus': ">BAC+4", 'BAC+2 ou +3': "BAC+2/3", 'BAC': "BAC", 'BEP, CAP': "BEP/CAP",
                     '0': "Non renseigné", 'Aucun diplôme': 'Aucun'},
    "code_age_1": {'25 à 34 ans': "25-34", '35 à 44 ans': "35-44", '18 à 24 ans': '<24', '0': '0age',
                   '45 à 54 ans': '45-54', '55 à 64 ans': '55-64', '65 ans et plus': ">65"},
    "code_sexe": {'Féminin': "F", 'Masculin': "M", '0': "0sex"},
    "how_decouvert": {'Amis': "deco_amis", 'Internet': 'deco_internet', 'Famille': 'deco_famille',
                      'Emission TV': 'deco_TV', 'Presse': 'deco_presse', '0': 'deco_Non',
                      'Collègues de travail': 'deco_travail', 'Publicité': "deco_pub"},
    "1partie_1": {'Entre amis': "p1_amis", 'Sur Internet': 'p1_internet', 'En famille': 'p1_famille',
                  'Dans une association ou un club': 'p1_asso', 'Dans un casino': 'p1_casino', '0': 'p1_Non',
                  'Entre collègues de travail': 'p1_travail', 'Dans un cercle de jeu': "p1_jeu"}
}, **{f"depuis{i}": {'Entre amis': f"depuis{i}_amis", 'Sur Internet': f"depuis{i}_internet",
                     'En famille': f"depuis{i}_famille", 'Dans une association': f"depuis{i}_asso",
                     'Dans un casino': f"depuis{i}_casino", '0': f"depuis{i}_Non",
                     'Entre collègues de travail': f"depuis{i}_travail", 'Dans un cercle de jeu': f"depuis{i}_jeu"}
      for i in range(1, 7)}}


def import_fpt() -> pd.DataFrame:
    df = pd.read_csv(os.path.join(path_data, "poker_data", "FPT_all.csv"), sep="\t", encoding="utf-8")
    df = df.loc[df.pro_poker == "Amateur", :]
    df = df.loc[:, socio_fields + poker_fields + ["naissance"]]
    for c in socio_fields + poker_fields:
        print(c)
        print(df.loc[:, c].unique())
        if c in recode:
            if c in df.columns:
                df.loc[:, c] = df.loc[:, c].apply(lambda z: recode[c][z])
    return df


def build_visuals(data: pd.DataFrame):
    all_csp = data.loc[:, "code_profession"].unique()
    cash_cat: Dict[str, int] = {"Aucune": 0, "<5": 1, "5-10": 2, "10-20": 3, "20-50": 4, ">50": 5}
    durees: Dict[str, int] = {"1 ans": 1, "2 ans": 2, "3 ans": 3, "4 ans": 4, "5 ans": 5, "6 ans": 6,
                              "6 ans et plus": 7}
    for csp in all_csp:
        print(csp)
        cas_data = data.loc[data.code_profession == csp, :]
        if cas_data.shape[0] > 10:
            starting_categories: Dict[Tuple[str, str, str], int] = {
                (cash_init, duree, cash_final): cas_data.loc[(cas_data.code_anciennete_1 == duree) &
                                                             (cas_data.code_somme_pnl_v1 == cash_final) &
                                                             (cas_data.code_somme_p1_v1 == cash_init), :].shape[0]
                for cash_init in cash_cat for duree in durees for cash_final in cash_cat}
            links = {key: starting_categories[key] for key in starting_categories if starting_categories[key] > 0}
            print(links)
            nodes_in = list(set([(key[0], key[1]) for key in links.keys()]))
            nodes_out = list(set([key[2] for key in links.keys()]))
            labels_in = [key[0] for key in nodes_in]
            labels_out = nodes_out.copy()
            x_in = [(10 - durees[key[1]])/10 for key in nodes_in]
            x_out = [1 for key in nodes_out]
            y_in = [(10 - cash_cat[key[0]])/10 for key in nodes_in]
            y_out = [(10 - cash_cat[key])/10 for key in nodes_out]
            source = [nodes_in.index((key[0], key[1])) for key in links.keys()]
            target = [nodes_out.index(key[2]) + len(nodes_in) for key in links.keys()]
            value = [links[key] for key in links.keys()]
            color = ["white" for key in nodes_in + nodes_out]
            fig = go.Figure(go.Sankey(
                arrangement="snap",
                node={"label": labels_in + labels_out, "x": x_in + x_out, "y": y_in + y_out, 'pad': 10, 'color': color},
                link={"source": source, "target": target, "value": value}))
            fig.update_layout(
                title=csp,
            )
            fig.write_html(os.path.join(path_data, f"sankey_with_time_{csp}.html"))


if __name__ == "__main__":
    data_poker = import_fpt()
    build_visuals(data_poker)
