import os
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

path_data = os.getcwd()
path_dot = 'insert_path_to_dot'

socio_fields = ["code_sexe", 'conjugal1', 'familial2', "code_profession", "code_diplome",  'code_emploi', "code_age_1",
                 "situation_pro1", 'code_agglo_1', 'nb_enfants']

socio_num_fields = []
poker_fields = ["code_somme_pni_v1", "code_somme_pnl_v1", "code_nbj_l_v1"]
poker_fields_test = ["code_nbj_l_v1"]
recode = {
    "code_somme_pni_v1": {'Moins de 5E': '<5', 'Entre 10 et 20E': '10-20', 'Entre 5 et 10E': '5-10',
                          'Non réponse': 'Non réponse', 'Plus de 50E': '>50', 'Entre 20 et 50E': '20-50',
                          'Aucune': 'Aucune'},
    "code_somme_pnl_v1": {'Moins de 5E': '<5', 'Entre 10 et 20E': '10-20', 'Entre 5 et 10E': '5-10',
                          'Non réponse': 'Non réponse', 'Plus de 50E': '>50', 'Entre 20 et 50E': '20-50',
                          'Aucune': 'Aucune'},
    "code_nbj_l_v1": {'2 fois': "2", '3 fois': "3", 'Tous les jours': "7", '4 fois': "4", 'Non réponse': 'Non réponse',
                      'Non': "0", '6 fois': "6", '5 fois': "5", '1 fois': "1"},
    "code_reg_v1": {'Pratique on-line régulière': "reg", 'Pratique on-line quotidienne': "quot",
                    'Pratique on-line très régulière': 'trs reg', '0': '0', 'Pratique on-line irrégulière': 'irr'},
    "code_profession": {'Etudiants, apprentis, stagiaires, sans emploi': "ETU", 'Ouvriers': "OUV",
                        'Professions intermédiaires': "INT", 'Employés': "EMP",
                        'Cadres et professions intellectuelles supérieures': "CAD",
                        "Artisans, commerçants et chef d'entreprises": "ART", '0': '0prof', 'Non classés': 'Non classés',
                        'Agriculteurs exploitants': "AGR"},
    "code_diplome": {'BAC+4 et plus': ">BAC+4", 'BAC+2 ou +3': "BAC+2/3", 'BAC': "BAC", 'BEP, CAP': "BEP/CAP",
                     '0': "0dip", 'Aucun diplôme': 'Sans'},
    "code_age_1": {'25 à 34 ans': "25-34", '35 à 44 ans': "35-44", '18 à 24 ans': '<24', '0': '0age',
                   '45 à 54 ans': '45-54', '55 à 64 ans': '55-64', '65 ans et plus': ">65"},
    "code_sexe": {'Féminin': "F", 'Masculin': "M", '0': "0sex"}

}
titles = {
    "code_profession": "profession", "code_diplome": "diplôme", "code_age_1": "âge", "code_sexe": "genre",
    "code_somme_pni_v1": "mise_internet", "code_somme_pnl_v1": "mise_live", "code_nbj_l_v1": "nombre_jours",
    "code_reg_v1": "régularité", "conjugal1": "sit_conjugale", "familial2": "sit_familiale", "code_emploi": "emploi",
    "situation_pro1": "sit_prof", "code_agglo_1": "taille_agglo", "nb_enfants": "nb_enfants"
}
mods = {
    "code_nbj_l_v1": [str(j) for j in range(8)],
    "code_somme_pnl_v1": ['<5', '5-10', '10-20', '20-50', '>50'],
    "code_somme_pni_v1": ['<5', '5-10', '10-20', '20-50', '>50']
}


def import_fpt() -> pd.DataFrame:
    df = pd.read_csv(os.path.join(path_data, "poker_data", "FPT_all.csv"), sep="\t", encoding="utf-8")
    df = df.loc[:, socio_fields + socio_num_fields + poker_fields]
    print(df.info())
    for c in socio_fields + poker_fields:
        print(c)
        print(df.loc[:, c].unique())
        if c in recode:
            df.loc[:, c] = df.loc[:, c].apply(lambda z: recode[c][z])
    return df


def chi2(df: pd.DataFrame) -> None:
    for yf in poker_fields:
        listy = sorted([y for y in df.loc[:, yf].unique() if y != "Non réponse" and df.loc[df.loc[:, yf] == y, :].shape[0] > 9])
        for xf in socio_fields:
            listx = sorted([x for x in df.loc[:, xf].unique() if x != "Non réponse" and df.loc[df.loc[:, xf] == x, :].shape[0] > 9])
            tab = np.array([[df.loc[(df.loc[:, xf] == x) & (df.loc[:, yf] == y), :].shape[0]
                for x in listx] for y in listy]).transpose()
            cho, pval, d, s = chi2_contingency(tab)
            if pval < 0.1:
                freq = pd.DataFrame(tab, columns=listy, index=listx)
                freq.loc[:, "som"] = freq.apply(lambda z: z.sum(), axis=1)
                for c in listy:
                    freq.loc[:, c] = freq.apply(lambda z: int(z[c]/z["som"]*100), axis=1)
                print("\n", yf, xf)
                print(freq)
                print(pval, cho)


if __name__ == "__main__":
    df = import_fpt()
    chi2(df)