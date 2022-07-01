import os
from random import randint
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC
import matplotlib

path_data = os.getcwd()
path_dot = "insert_path_to_dot"

socio_fields = ["code_sexe", 'conjugal1', 'familial2', "code_profession", "code_diplome", 'code_emploi', "code_age_1",
                'code_agglo_1', 'nb_enfants']
socio_fields = ["code_sexe", "code_profession", "code_diplome", 'code_emploi', "code_age_1"]

socio_num_fields = []
poker_fields = ["code_somme_pni_v1", "code_somme_pnl_v1", "code_nbj_l_v1", "code_somme_p1_v1"]
poker_fields_test = ["code_somme_pnl_v1"]
extra_fields = ["how_decouvert", "1partie_1", "depuis1", "depuis2", "depuis3", "depuis4", "depuis5",
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
      for i in range(1, 7)}
          }

titles = {
    "code_profession": "profession", "code_diplome": "diplôme", "code_age_1": "âge", "code_sexe": "genre",
    "code_somme_pni_v1": "mise_internet", "code_somme_pnl_v1": "mise_live", "code_nbj_l_v1": "nombre_jours",
    "code_reg_v1": "régularité", "conjugal1": "sit_conjugale", "familial2": "sit_familiale", "code_emploi": "emploi",
    "situation_pro1": "sit_prof", "code_agglo_1": "taille_agglo", "nb_enfants": "nb_enfants",
    "code_somme_p1_v1": "mise_partie1"
}
mods = {
    "code_nbj_l_v1": [str(j) for j in range(8)],
    "code_somme_pnl_v1": ['Aucune', '<5', '5-10', '10-20', '20-50', '>50'],
    "code_somme_pni_v1": ['Aucune', '<5', '5-10', '10-20', '20-50', '>50']
}
lmods = {
    "code_nbj_l_v1": ["<=6", "7"],
    "code_somme_pnl_v1": ['<10', '>10'],
    "code_somme_pni_v1": ['<10', '>10']
}


def ordre(n: str) -> int:
    if n == '<5' or n == "1" or n == "irr" or n == "AGR" or n == "BEP/CAP" or n == "<24" or n == "F" or n == "Aucune":
        return 1
    if n == '5-10' or n == '2' or n == "reg" or n == "OUV" or n == "BAC" or n == "25-34" or n == "<10":
        return 2
    if n == '10-20' or n == '3' or n == "trs reg" or n == "EMP" or n == "BAC+2/3" or n == "35-44" or n == "10-50":
        return 3
    if n == '20-50' or n == '4' or n == "quot" or n == "INT" or n == ">BAC+4" or n == "45-54":
        return 4
    if n == '>50' or n == '5' or n == "ART" or n == "55-64":
        return 5
    if n == '6' or n == "CAD" or n == ">65":
        return 6
    if n == '7' or n == "ETU":
        return 7
    return 0


def import_fpt() -> pd.DataFrame:
    df = pd.read_csv(os.path.join(path_data, "poker_data", "FPT_all.csv"), sep="\t", encoding="utf-8")
    ## Test on vire les joueurs pro
    df = df.loc[df.pro_poker == "Amateur", :]
    df = df.loc[:, socio_fields + socio_num_fields + poker_fields + extra_fields + ["naissance"]]
    for c in socio_fields + poker_fields + extra_fields:
        print(c)
        print(df.loc[:, c].unique())
        if c in recode:
            if c in df.columns:
                df.loc[:, c] = df.loc[:, c].apply(lambda z: recode[c][z])
    return df


def etudes_bivariees(df: pd.DataFrame) -> None:
    for i in range(len(socio_fields)):
        for k in range(len(poker_fields)):
            field_x = socio_fields[i]
            field_h = poker_fields[k]
            df.loc[df.loc[:, field_h].isin(["5-10", "<5"]), field_h] = "<10"
            mods_x = sorted([z for z in df.loc[:, field_x].unique() if z not in ['0', "Non réponse", "0age",
                                                                                 "0dip", "Oprof", "Non classés", "Non renseigné"] and
                             df.loc[df.loc[:, field_x] == z, :].shape[0] > 10], key=ordre)
            mods_h = sorted([z for z in df.loc[:, field_h].unique() if z not in ['0', "Non réponse", "Non renseigné",
                                                                                 "Non classés"] and
                             df.loc[df.loc[:, field_h] == z, :].shape[0] > 10], key=ordre)
            fig, ax = plt.subplots(1, len(mods_x), figsize=(20, 15))
            fig.suptitle(f"Distribution de {titles[field_h]} selon x:{titles[field_x]}")
            max_h = max(df.loc[(df.loc[:, field_x] == mods_x[mx]) &
                               (df.loc[:, field_h] == mh)].shape[0] for mh in mods_h for mx in range(len(mods_x)))
            for mx in range(len(mods_x)):
                ax[mx].set_xlabel(mods_x[mx])
                hist = [df.loc[(df.loc[:, field_x] == mods_x[mx]) & (df.loc[:, field_h] == mh)].shape[0] for mh in
                        mods_h]
                print(hist)
                # ax[mx].set_ylim(0, max_h + 1)
                ax[mx].set_xticks(range(len(mods_h)))
                ax[mx].set_xticklabels(mods_h)
                ax[mx].bar(range(len(mods_h)), hist, color="lightgray")
            fig.savefig(os.path.join(path_data, "poker_exports", f"1pc_{titles[field_h]}_{titles[field_x]}.png"))
            plt.close()

        for j in range(i + 1, len(socio_fields)):
            for k in range(len(poker_fields)):
                field_x, field_y = socio_fields[i], socio_fields[j]
                field_h = poker_fields[k]
                mods_y = sorted([z for z in df.loc[:, field_y].unique() if z not in ['0', "Aucune", "Non réponse",
                                                                                     "Non classés"]
                                 and df.loc[df.loc[:, field_y] == z, :].shape[0] > 10], key=ordre, reverse=True)
                mods_h = sorted([z for z in df.loc[:, field_h].unique() if z not in ['0', "Aucune", "Non réponse",
                                                                                     "Non classés"]
                                 and df.loc[df.loc[:, field_h] == z, :].shape[0] > 10], key=ordre)
                print(mods_x, mods_y, mods_h)
                fig, ax = plt.subplots(len(mods_y), len(mods_x), figsize=(20, 15))
                fig.suptitle(f"Distribution de {titles[field_h]} selon x:{titles[field_x]} et y:{titles[field_y]}")
                max_h = max(df.loc[(df.loc[:, field_x] == mods_x[mx]) & (df.loc[:, field_y] == mods_y[my]) &
                                   (df.loc[:, field_h] == mh)].shape[0] for mh in mods_h for mx in range(len(mods_x))
                            for my in range(len(mods_y)))
                for mx in range(len(mods_x)):
                    for my in range(len(mods_y)):
                        ax[my, mx].set_xlabel(mods_x[mx])
                        ax[my, mx].set_ylabel(mods_y[my] if mx == 0 else "")
                        hist = [df.loc[(df.loc[:, field_x] == mods_x[mx]) & (df.loc[:, field_y] == mods_y[my]) &
                                       (df.loc[:, field_h] == mh)].shape[0] for mh in mods_h]
                        print(hist)
                        ax[my, mx].set_ylim(0, max_h + 1)
                        ax[my, mx].set_xticks(range(len(mods_h)))
                        ax[my, mx].set_xticklabels(mods_h)
                        ax[my, mx].bar(range(len(mods_h)), hist)
                fig.savefig(os.path.join(path_data, "poker_exports",
                                         f"{titles[field_h]}_{titles[field_x]}_{titles[field_y]}.png"))
                plt.close()


def predictions(df: pd.DataFrame) -> None:
    for yfield in ["code_somme_pni_v1", "code_somme_pnl_v1"]:
        df_loc = df.loc[df.loc[:, yfield].apply(lambda z: z in mods[yfield])]
        df_loc.loc[df_loc.loc[:, yfield].isin(["5-10", "<5"]), yfield] = "<10"
        df_loc.loc[df_loc.loc[:, yfield].isin(["20-50", ">50"]), yfield] = ">20"
        df_loc.loc[df_loc.code_diplome.isin(["BAC+2/3", ">BAC+4"]), "code_diplome"] = ">BAC"
        df_loc.loc[df_loc.code_diplome.isin(["BEP/CAP", "Aucun", "BAC"]), "code_diplome"] = "<=BAC"
        df_loc.loc[df_loc.naissance <= 1972, "code_age_1"] = ">39"
        df_loc.loc[df_loc.naissance > 1972, "code_age_1"] = "<=39"
        df_loc.loc[df_loc.code_profession.isin(["CAD", "INT", "ART"]), "code_profession"] = "CSP+"
        df_loc.loc[df_loc.code_profession.isin(["OUV", "AGR", "EMP"]), "code_profession"] = "CSP-"
        df_loc.loc[df_loc.loc[:, "1partie_1"].isin(["p1_amis", "p1_famille", "p1_asso", "p1_travail"]),
                   "1partie_1"] = "fam/amis/trav"
        df_loc.loc[df_loc.loc[:, "1partie_1"].isin(["p1_jeu", "p1_casino"]), "1partie_1"] = "casino/cercle"
        df_loc = df_loc.loc[df_loc.code_sexe != '0sex', :]
        df_loc = df_loc.loc[df_loc.code_profession != 'Non classés', :]
        df_loc = df_loc.loc[df_loc.code_profession != 'Non renseigné', :]
        df_loc = df_loc.loc[df_loc.code_profession != 'ETU', :]
        df_loc = df_loc.loc[df_loc.code_diplome != 'Non renseigné', :]
        df_loc = df_loc.loc[df_loc.code_diplome != '0', :]
        df_loc = df_loc.loc[df_loc.code_emploi != '0', :]
        df_loc = df_loc.loc[df_loc.code_age_1 != '0age', :]
        df_loc = df_loc.loc[df_loc.loc[:, "1partie_1"] != 'p1_Non', :]
        X1 = df_loc.loc[:, socio_fields]
        X3 = df_loc.loc[:, socio_fields + ["1partie_1"]]
        Y = df_loc.loc[:, yfield]
        enc = {"base": OneHotEncoder(handle_unknown='ignore', sparse=False),
               "all": OneHotEncoder(handle_unknown='ignore', sparse=False)}
        X_base = enc["base"].fit_transform(X1)
        X_all = enc["all"].fit_transform(X3)

        samples = {"base": X_base, "all": X_all}
        for s in samples:
            X = samples[s]
            features = [c for f in enc[s].categories_ for c in f]

            print("\n*****************\n", yfield, s)
            print(features)
            print(Y.unique())
            n_class = len(Y.unique())
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33)

            svmc = SVC(kernel="linear")
            svmc.fit(X_train, Y_train)
            sc = svmc.score(X_test, Y_test)
            cm = confusion_matrix(Y_test, svmc.predict(X_test))
            print("svcl", sc, 1 / n_class, "\n", cm)
            print(svmc.classes_)

            svmc = SVC(kernel="rbf")
            svmc.fit(X_train, Y_train)
            sc = svmc.score(X_test, Y_test)
            cm = confusion_matrix(Y_test, svmc.predict(X_test))
            print("svcr", sc, 1 / n_class, "\n", cm)

            rlog = LogisticRegression()
            rlog.fit(X_train, Y_train)
            sc = rlog.score(X_test, Y_test)
            cm = confusion_matrix(Y_test, rlog.predict(X_test))
            print("reglog", sc, 1 / n_class, "\n", cm)

            rfc = RandomForestClassifier()
            rfc.fit(X_train, Y_train)
            sc = rfc.score(X_test, Y_test)
            cm = confusion_matrix(Y_test, rfc.predict(X_test))
            print("rfc", sc, 1 / n_class, "\n", cm)
            print(rfc.feature_importances_)


def bimodal(dfg: pd.DataFrame) -> None:
    df = dfg.copy()
    df = df.loc[df.loc[:, "code_somme_pnl_v1"].isin(["Aucune", "<5", "5-10", "10-20", "20-50", ">50"]), :]
    print(df.groupby("code_somme_pnl_v1").count())

    for yfield in ["code_somme_pni_v1", "code_somme_pnl_v1"]:
        print(yfield)
        df_loc = df.loc[df.loc[:, yfield].apply(lambda z: z in mods[yfield])]
        df_loc.loc[df_loc.loc[:, yfield].isin(["Aucune", "5-10", "<5"]), yfield] = "<10"
        df_loc.loc[df_loc.loc[:, yfield].isin(["20-50", "10-20", ">50"]), yfield] = ">10"
        df_loc.loc[df_loc.code_diplome.isin(["BAC+2/3", ">BAC+4"]), "code_diplome"] = ">BAC"
        df_loc.loc[df_loc.code_diplome.isin(["BEP/CAP", "Aucun", "BAC"]), "code_diplome"] = "<=BAC"
        df_loc.loc[df_loc.naissance <= 1972, "code_age_1"] = ">39"
        df_loc.loc[df_loc.naissance > 1972, "code_age_1"] = "<=39"
        df_loc.loc[df_loc.code_profession.isin(["CAD", "INT", "ART"]), "code_profession"] = "CSP+"
        df_loc.loc[df_loc.code_profession.isin(["OUV", "AGR", "EMP"]), "code_profession"] = "CSP-"
        df_loc.loc[df_loc.loc[:, "1partie_1"].isin(["p1_amis", "p1_famille", "p1_asso", "p1_travail"]),
                   "1partie_1"] = "fam/amis/trav"
        df_loc.loc[df_loc.loc[:, "1partie_1"].isin(["p1_jeu", "p1_casino"]), "1partie_1"] = "casino/cercle"
        df_loc.loc[df_loc.loc[:, "1partie_1"].isin(["p1_internet"]), "1partie_1"] = "internet"
        df_loc = df_loc.loc[df_loc.code_sexe != '0sex', :]
        df_loc = df_loc.loc[df_loc.code_profession != 'Non classés', :]
        df_loc = df_loc.loc[df_loc.code_profession != 'Non renseigné', :]
        df_loc = df_loc.loc[df_loc.code_profession != 'ETU', :]
        df_loc = df_loc.loc[df_loc.code_diplome != 'Non renseigné', :]
        df_loc = df_loc.loc[df_loc.code_diplome != '0', :]
        df_loc = df_loc.loc[df_loc.code_age_1 != '0age', :]
        df_loc = df_loc.loc[df_loc.loc[:, "1partie_1"] != 'p1_Non', :]
        ssf = [fi for fi in socio_fields if fi != "code_emploi"]
        X1 = df_loc.loc[:, ssf]
        X3 = df_loc.loc[:, ssf + ["1partie_1"]]
        Y = df_loc.loc[:, yfield]
        print("\n*****************\n", yfield)
        print(df_loc.groupby(yfield).count())
        enc = {"base": OneHotEncoder(handle_unknown='ignore', sparse=False),
               "all": OneHotEncoder(handle_unknown='ignore', sparse=False)}
        X_base = enc["base"].fit_transform(X1)
        X_all = enc["all"].fit_transform(X3)
        svmclf = dict()

        samples = {"base": X_base, "all": X_all}
        for s in samples:
            X = samples[s]
            print(s)

            n_class = len(Y.unique())
            rs = randint(1, 500)
            print(rs)
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=rs)
            roc = dict()

            print("svc, balanced")
            svmc = SVC(kernel="linear", class_weight="balanced", probability=True)
            svmc.fit(X_train, Y_train)
            sc = svmc.score(X_test, Y_test)
            pr = svmc.predict_proba(X_test)[:, 0]
            cm = confusion_matrix(Y_test, svmc.predict(X_test))
            roc["svm_bal"] = roc_curve(Y_test, pr, pos_label="<10")
            svmclf[s] = roc_curve(Y_test, pr, pos_label="<10")
            print(sc, 1 / n_class, "\n", cm)

            print("random forest, balanced")
            rfc = RandomForestClassifier(class_weight="balanced")
            rfc.fit(X_train, Y_train)
            sc = rfc.score(X_test, Y_test)
            pr = rfc.predict_proba(X_test)[:, 0]
            print(list(Y_test)[:10])
            print(pr[:10])
            print(rfc.predict(X_test)[:10])
            cm = confusion_matrix(Y_test, rfc.predict(X_test))
            roc["rdm_for_bal"] = roc_curve(Y_test, pr, pos_label="<10")
            print(sc, 1 / n_class, "\n", cm)

            print("sgd, balanced")
            sgd = SGDClassifier(loss="log", penalty="l2", max_iter=50, class_weight="balanced")
            sgd.fit(X_train, Y_train)
            sc = sgd.score(X_test, Y_test)
            pr = sgd.predict_proba(X_test)[:, 0]
            print(list(Y_test)[:10])
            print(pr[:10])
            print(sgd.predict(X_test)[:10])
            cm = confusion_matrix(Y_test, sgd.predict(X_test))
            roc["sgd_bal"] = roc_curve(Y_test, pr, pos_label="<10")
            print(sc, 1 / n_class, "\n", cm)

            print("ada boost")
            abc = AdaBoostClassifier(n_estimators=100)
            abc.fit(X_train, Y_train)
            sc = abc.score(X_test, Y_test)
            pr = abc.predict_proba(X_test)[:, 0]
            print(list(Y_test)[:10])
            print(pr[:10])
            print(abc.predict(X_test)[:10])
            cm = confusion_matrix(Y_test, abc.predict(X_test))
            roc["abc_bal"] = roc_curve(Y_test, pr, pos_label="<10")
            print(sc, 1 / n_class, "\n", cm)

            plt.figure()
            lw = 2
            for clf in ["svm_bal", "rdm_for_bal", "sgd_bal", "abc_bal"]:
                plt.plot(roc[clf][0], roc[clf][1], lw=lw, label=f"{clf}, AUC={auc(roc[clf][0], roc[clf][1])} ")
            plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver operating characteristic example')
            plt.legend(loc="lower right")
            plt.close()

        plt.figure()
        lw = 2
        for s in samples:
            plt.plot(svmclf[s][0], svmclf[s][1], lw=lw, label=f"{s}, AUC={auc(svmclf[s][0], svmclf[s][1])} ")
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.close()


def stats_adhoc(df) -> None:
    for cat_h in ["code_somme_pni_v1", "code_somme_pnl_v1"]:
        print("\n**************\n", cat_h)
        for cat_x, modset_x in [("code_age_1", ["<24", "25-34"]), ("code_age_1", ["35-44", "45-54", "55-64", ">65"]),
                                ("code_diplome", ["Sans", "BEP/CAP", "BAC"]), ("code_diplome", ["BAC+2/3", ">BAC+4"]),
                                ("code_profession", ["ART", "OUV", "EMP"]), ("code_profession", ["INT", "CAD"]),
                                ("code_sexe", ["M"]), ("code_sexe", ["F"])]:
            print(f"{cat_x}: {' & '.join(modset_x)}")
            tot = df.loc[df.loc[:, cat_x].isin(modset_x) & df.loc[:, cat_h].isin(mods[cat_h]), :].shape[0]
            print([
                f"{mo} : {round(df.loc[df.loc[:, cat_x].isin(modset_x) & (df.loc[:, cat_h] == mo), :].shape[0] * 100 / tot)}%"
                for mo in mods[cat_h]])

        print("\n**")

        for cat_x in socio_fields:
            mods_x = [m for m in df.loc[:, cat_x].unique() if
                      df.loc[(df.loc[:, cat_x] == m) & df.loc[:, cat_h].isin(mods[cat_h]), :].shape[0] > 10]
            for mod_x in mods_x:
                print(f"{cat_x}: {mod_x}")
                tot = df.loc[(df.loc[:, cat_x] == mod_x) & df.loc[:, cat_h].isin(mods[cat_h]), :].shape[0]
                print(
                    [f"{round(df.loc[(df.loc[:, cat_x] == mod_x) & (df.loc[:, cat_h] == mo), :].shape[0] * 100 / tot)}%"
                     for mo in mods[cat_h]])

        print("\n****")

        for (i, cat_x) in enumerate(socio_fields):
            for j in range(i + 1, len(socio_fields)):
                cat_y = socio_fields[j]
                mods_x = [m for m in df.loc[:, cat_x].unique() if
                          df.loc[(df.loc[:, cat_x] == m) & df.loc[:, cat_h].isin(mods[cat_h]), :].shape[0] > 10]
                mods_y = [m for m in df.loc[:, cat_y].unique() if
                          df.loc[(df.loc[:, cat_y] == m) & df.loc[:, cat_h].isin(mods[cat_h]), :].shape[0] > 10]
                for mod_x in [m for m in mods_x if m not in ["M", "Non classé"]]:
                    for mod_y in [m for m in mods_y if m not in ["M", "Non classé"]]:
                        tot = df.loc[(df.loc[:, cat_x] == mod_x) & (df.loc[:, cat_y] == mod_y) &
                                     df.loc[:, cat_h].isin(mods[cat_h]), :].shape[0]
                        if tot > 9:
                            res = [round(df.loc[(df.loc[:, cat_x] == mod_x) & (df.loc[:, cat_y] == mod_y)
                                                & (df.loc[:, cat_h] == mo), :].shape[0] * 100 / tot) for mo in
                                   mods[cat_h]]
                            if (res[0] + res[1]) < 30 or (res[0] + res[1]) > 46 or (res[3] + res[4]) < 35 or (
                                    res[0] + res[1]) > 50:
                                print(f"{cat_x}: {mod_x}, {cat_y}: {mod_y}")
                                print([
                                    f"{round(df.loc[(df.loc[:, cat_x] == mod_x) & (df.loc[:, cat_y] == mod_y) & (df.loc[:, cat_h] == mo), :].shape[0] * 100 / tot)}% "
                                    for mo in mods[cat_h]])


def head_tail(dfa: pd.DataFrame):
    matplotlib.rcParams.update({'font.size': 22})
    for var in ["code_somme_pnl_v1", "code_somme_p1_v1", "code_somme_pni_v1"]:
        dfz = dfa.copy()
        dfz.loc[dfz.loc[:, "code_profession"].isin(["Non renseigné", "ETU"]), "code_profession"] = "autre"
        dfz.loc[dfz.naissance <= 1972, "code_age_1"] = ">39"
        dfz.loc[dfz.naissance > 1972, "code_age_1"] = "<=39"
        dfz = dfz.loc[dfz.code_sexe != '0sex', :]
        dfz = dfz.loc[dfz.code_profession != 'Non classés', :]
        dfz = dfz.loc[dfz.code_profession != 'Non renseigné', :]
        dfz = dfz.loc[dfz.code_profession != 'ETU', :]
        dfz = dfz.loc[dfz.code_diplome != 'Non renseigné', :]
        dfz = dfz.loc[dfz.code_diplome != '0', :]
        dfz = dfz.loc[dfz.code_age_1 != '0age', :]
        dfz = dfz.loc[dfz.loc[:, "1partie_1"] != 'p1_Non', :]
        print(var)
        top = dfz.loc[dfz.loc[:, var] == ">50", :]
        free = dfz.loc[dfz.loc[:, var] == "Aucune", :]
        print("top players")
        for field_h in socio_fields:
            mods_h = sorted([z for z in dfz.loc[:, field_h].unique() if z not in ["Non réponse", "Non classés"]],
                            key=ordre)
            print(mods_h)

            ntop, n, nfree = top.shape[0], dfz.shape[0], free.shape[0]
            plt.figure(figsize=(20, 15))
            plt.title(f"Top vs Free players : {titles[field_h]}")
            hist_top = [top.loc[(top.loc[:, field_h] == mh)].shape[0] for mh in mods_h]
            hist_all = [dfz.loc[(dfz.loc[:, field_h] == mh)].shape[0] for mh in mods_h]
            hist_free = [free.loc[(free.loc[:, field_h] == mh)].shape[0] for mh in mods_h]

            print(mods_h)
            print([z/sum(hist_top) for z in hist_top])
            print([z/sum(hist_all) for z in hist_all])
            print([z/sum(hist_free) for z in hist_free])

            plt.xticks(range(len(mods_h)), mods_h)
            plt.yticks([i/10 for i in range(10)], [f"{i*10} %" for i in range(10)])
            plt.bar([x + 0.2 for x in range(len(mods_h))], [h / ntop for h in hist_top], width=0.25,
                    label="mises supérieures à 50 euros", color="white", hatch="//", edgecolor="black")
            plt.bar([x for x in range(len(mods_h))], [h / n for h in hist_all], width=0.25,
                    label="mises intermédiaires", color="white", edgecolor="black")
            plt.bar([x - 0.2 for x in range(len(mods_h))], [h / nfree for h in hist_free], width=0.25,
                    label="parties gratuites", color="gray", edgecolor="black")
            plt.legend()
            plt.ylabel("Pourcentage de l'effectif")
            plt.xlabel(titles[field_h])
            plt.savefig(os.path.join(path_data, "poker_exports", f"top {var} {titles[field_h]}.png"))
            plt.close()

            if field_h == "code_diplome":
                top_i1 = top.loc[top.loc[:, "1partie_1"] == "p1_internet", :]
                top_l1 = top.loc[top.loc[:, "1partie_1"] != "p1_internet", :]
                free_i1 = free.loc[free.loc[:, "1partie_1"] == "p1_internet", :]
                free_l1 = free.loc[free.loc[:, "1partie_1"] != "p1_internet", :]
                dfz_i1 = dfz.loc[dfz.loc[:, "1partie_1"] == "p1_internet", :]
                dfz_l1 = dfz.loc[dfz.loc[:, "1partie_1"] != "p1_internet", :]

                ntopi, ni, nfreei = top_i1.shape[0], dfz_i1.shape[0], free_i1.shape[0]
                ntopl, nl, nfreel = top_l1.shape[0], dfz_l1.shape[0], free_l1.shape[0]

                plt.figure(figsize=(20, 15))
                plt.title(f"Top vs Free players : {titles[field_h]}")
                hist_top = [top_i1.loc[(top_i1.loc[:, field_h] == mh)].shape[0] for mh in mods_h]
                hist_all = [dfz_i1.loc[(dfz_i1.loc[:, field_h] == mh)].shape[0] for mh in mods_h]
                hist_free = [free_i1.loc[(free_i1.loc[:, field_h] == mh)].shape[0] for mh in mods_h]

                plt.xticks(range(len(mods_h)), mods_h)
                plt.yticks([i / 10 for i in range(10)], [f"{i * 10} %" for i in range(10)])
                plt.bar([x + 0.2 for x in range(len(mods_h))], [h / ntopi for h in hist_top], width=0.25,
                        label="mises supérieures à 50 euros", color="white", hatch="//", edgecolor="black")
                plt.bar([x for x in range(len(mods_h))], [h / ni for h in hist_all], width=0.25,
                        label="mises intermédiaires", color="white", edgecolor="black")
                plt.bar([x - 0.2 for x in range(len(mods_h))], [h / nfreei for h in hist_free], width=0.25,
                        label="parties gratuites", color="gray", edgecolor="black")
                plt.legend()
                plt.ylabel("Pourcentage de l'effectif")
                plt.xlabel(titles[field_h])
                plt.savefig(os.path.join(path_data, "poker_exports", f"top {var} {titles[field_h]} start internet.png"))
                plt.close()

                plt.figure(figsize=(20, 15))
                plt.title(f"Top vs Free players : {titles[field_h]}")
                hist_top = [top_l1.loc[(top_l1.loc[:, field_h] == mh)].shape[0] for mh in mods_h]
                hist_all = [dfz_l1.loc[(dfz_l1.loc[:, field_h] == mh)].shape[0] for mh in mods_h]
                hist_free = [free_l1.loc[(free_l1.loc[:, field_h] == mh)].shape[0] for mh in mods_h]

                plt.xticks(range(len(mods_h)), mods_h)
                plt.yticks([i / 10 for i in range(10)], [f"{i * 10} %" for i in range(10)])
                plt.bar([x + 0.2 for x in range(len(mods_h))], [h / ntopl for h in hist_top], width=0.25,
                        label="mises supérieures à 50 euros", color="white", hatch="//", edgecolor="black")
                plt.bar([x for x in range(len(mods_h))], [h / nl for h in hist_all], width=0.25,
                        label="mises intermédiaires", color="white", edgecolor="black")
                plt.bar([x - 0.2 for x in range(len(mods_h))], [h / nfreel for h in hist_free], width=0.25,
                        label="parties gratuites", color="gray", edgecolor="black")
                plt.legend()
                plt.ylabel("Pourcentage de l'effectif")
                plt.xlabel(titles[field_h])
                plt.savefig(os.path.join(path_data, "poker_exports", f"top {var} {titles[field_h]} start live.png"))
                plt.close()

        # va = ["code_sexe", "code_profession", "code_diplome", "code_age_1", "code_somme_pni_v1", "code_somme_pnl_v1"]
        # for fi in va:
        #     dfz = dfz.loc[~dfz.loc[:, fi].isin(["Non réponse", "0", 0]), va]
        # enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
        # XC = enc.fit_transform(dfz)
        # print(XC.shape)
        #
        # pca = PCA(n_components=6)
        # XP = pca.fit_transform(XC)
        # print(pca.explained_variance_ratio_)
        # print(pca.singular_values_)
        #
        # loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
        # features = [f"{va[int(f.split('_')[0][1:])]}_{f.split('_')[1]}" for f in enc.get_feature_names()]
        # print(enc.get_feature_names())
        # print(features)
        #
        # fig = px.scatter(XP, x=0, y=1, color=dfz.loc[:, var])
        # for i, feature in enumerate(features):
        #     if (loadings[i, 0] ** 2 + loadings[i, 1] ** 2) > 0.01:
        #         fig.add_shape(type='line', x0=0, y0=0, x1=loadings[i, 0] * 3, y1=loadings[i, 1] * 3)
        #         fig.add_annotation(x=loadings[i, 0] * 3, y=loadings[i, 1] * 3, ax=0, ay=0, xanchor="center",
        #                            yanchor="bottom", text=feature)
        # fig.show()
        #
        # fig = px.scatter(XP, x=2, y=3, color=dfz.loc[:, var])
        # for i, feature in enumerate(features):
        #     if (loadings[i, 2] ** 2 + loadings[i, 3] ** 2) > 0.01:
        #         fig.add_shape(type='line', x0=0, y0=0, x1=loadings[i, 2] * 3, y1=loadings[i, 3] * 3)
        #         fig.add_annotation(x=loadings[i, 2] * 3, y=loadings[i, 3] * 3, ax=0, ay=0, xanchor="center",
        #                            yanchor="bottom", text=feature)
        # fig.show()


def pop_spec(dfa: pd.DataFrame) -> None:

    feat_titles = {
        "code_age_1_<24": "<24 ans", "code_age_1_25-34": "25-34 ans", "code_age_1_35-44": "35-44 ans",
        "code_age_1_45-54": "45-54 ans", "code_age_1_55-64": "55-64 ans", "code_age_1_>65": ">65 ans",
        "code_age_1_<34>": "<34 ans", "code_age_1_>35": ">35 ans",
        "code_profession_ETU": "ETU",
        "code_profession_CAD": "CAD", "code_profession_AGR": "AGR", "code_profession_INT": "INT",
        "code_profession_OUV": "OUV", "code_profession_EMP": "EMP",
        "code_profession_Non renseigné": "prof NR", "code_diplome_>BAC+4": ">BAC+4", "code_diplome_BAC+2/3": "BAC+2/3",
        "code_diplome_BAC": "BAC", "code_diplome_BEP/CAP": "BEP/CAP", "code_diplome_Non renseigné": "dipl NR",
        "code_diplome_>BAC+2": ">BAC+2",
        "code_sexe_F": "F", "code_sexe_M": "M",
        "code_somme_pnl_v1_<5": "live <5", "code_somme_pnl_v1_Aucune": "live 0", "code_somme_pnl_v1_5-10": "live 5-10",
        "code_somme_pnl_v1_10-20": "live 10-20", "code_somme_pnl_v1_20-50": "livee 20-50",
        "code_somme_pnl_v1_>50": "live > 50", "code_somme_pnl_v1_5-50": "live 5-50",
        "code_somme_pni_v1_<5": "internet <5", "code_somme_pni_v1_Aucune": "internet 0",
        "code_somme_pni_v1_5-10": "internet 5-10",
        "code_somme_pni_v1_10-20": "internet 10-20", "code_somme_pni_v1_20-50": "internet 20-50",
        "code_somme_pni_v1_>50": "internet > 50", "code_somme_pni_v1_5-50": "internet 5-50"
    }
    if True:
        ### Regroupement optionnel
        dfa.loc[dfa.code_age_1.isin(["35-44", "45-54", "55-64", ">65"]), "code_age_1"] = ">35"
        dfa.loc[dfa.code_somme_pnl_v1.isin(["<5", "5-10"]), "code_somme_pnl_v1"] = "<10"
        dfa.loc[dfa.code_somme_pnl_v1.isin(["10-20", "20-50"]), "code_somme_pnl_v1"] = "10-50"
        dfa.loc[dfa.code_somme_pni_v1.isin(["<5", "5-10"]), "code_somme_pni_v1"] = "<10"
        dfa.loc[dfa.code_somme_pni_v1.isin(["10-20", "20-50"]), "code_somme_pni_v1"] = "10-50"
        dfa.loc[dfa.code_diplome.isin(["BAC+2/3", ">BAC+4"]), "code_diplome"] = ">BAC+2"
        dfa.loc[dfa.code_profession.isin(["ETU"]), "code_profession"] = "Non renseigné"
        ###
        for var in [("code_emploi", "Chômeur, sans emploi"), ("code_profession", "CAD")]:
            print(var)
            dfz = dfa.loc[dfa.loc[:, var[0]] == var[1], :]
            va = ["code_sexe", "code_diplome", "code_age_1", "code_profession", "code_somme_pni_v1",
                  "code_somme_pnl_v1"]
            for fi in va:
                dfz = dfz.loc[~dfz.loc[:, fi].isin(["Non réponse", "0", 0]), va]
            enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
            XC = enc.fit_transform(dfz)
            print(XC.shape)

            pca = PCA(n_components=4)
            XP = pca.fit_transform(XC)
            print(pca.explained_variance_ratio_)
            print(pca.singular_values_)

            loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
            features = [f"{va[int(f.split('_')[0][1:])]}_{f.split('_')[1]}" for f in enc.get_feature_names()]
            print(enc.get_feature_names())
            print(features)

            plt.figure(figsize=(10, 10))
            ax = plt.subplot(111)
            for target in [("code_somme_pni_v1", 70), ("code_somme_pnl_v1", 30)]:
                ax.scatter(XP[:, 0], XP[:, 1], c=dfz.loc[:, target[0]].apply(lambda z: str(ordre(z)/6)),
                           edgecolors="none", marker="o", s=target[1])
            for i, feature in enumerate(features):
                if (loadings[i, 0] ** 2 + loadings[i, 1] ** 2) > 0.01 and feature in feat_titles:
                    plt.gca().add_line(plt.Line2D((0, loadings[i, 0] * 3), (0, loadings[i, 1] * 3), color="gray"))
                    plt.annotate(xy=(loadings[i, 0] * 3, loadings[i, 1] * 3), text=feat_titles[feature])
            for value in sorted(dfz.code_somme_pnl_v1.unique(), key=ordre):
                ax.scatter(0, 0, c=str(ordre(value)/6), label=value, edgecolors='none')

            ax.legend()
            plt.savefig(os.path.join(path_data, "poker_exports", f"acp_{var[0]}_12.png"))

            plt.figure(figsize=(10, 10))
            ax = plt.subplot(111)
            for target in [("code_somme_pni_v1", 70), ("code_somme_pnl_v1", 30)]:
                ax.scatter(XP[:, 2], XP[:, 3], c=dfz.loc[:, target[0]].apply(lambda z: str(ordre(z)/6)),
                           edgecolors="none", marker="o", s=target[1])
            for i, feature in enumerate(features):
                if (loadings[i, 0] ** 2 + loadings[i, 1] ** 2) > 0.01 and feature in feat_titles:
                    plt.gca().add_line(plt.Line2D((0, loadings[i, 2] * 3), (0, loadings[i, 3] * 3), color="gray"))
                    plt.annotate(xy=(loadings[i, 2] * 3, loadings[i, 3] * 3), text=feat_titles[feature])
            for value in sorted(dfz.code_somme_pnl_v1.unique(), key=ordre):
                ax.scatter(0, 0, c=str(ordre(value) / 6), label=value, edgecolors='none')
            ax.legend()
            plt.savefig(os.path.join(path_data, "poker_exports", f"acp_{var[0]}_34.png"))

            plt.close()


def evo_mise_sans_bac(df: pd.DataFrame) -> None:
    df = df.loc[df.code_diplome.isin(["Aucun", "BEP/CAP"]), :]
    for typ in ["code_somme_pnl_v1", "code_somme_pni_v1"]:
        df.loc[:, "contexte1"] = df.loc[:, "1partie_1"]
        # df.loc[df.loc[:, "1partie_1"].isin(["p1_asso", "p1_casino"]), "contexte1"] = "p1_casino_asso_"
        # df.loc[df.loc[:, "1partie_1"].isin(["p1_amis", "p1_travail"]), "contexte1"] = "p1_amis_travail"
        df.loc[df.loc[:, "1partie_1"].isin(["p1_amis", "p1_travail", "p1_asso", "p1_casino"]), "contexte1"] = "p1_live"
        df.loc[:, "somme1"] = df.loc[:, "code_somme_p1_v1"]
        df.loc[:, "sommeN"] = df.loc[:, typ]
        df.loc[df.loc[:, "code_somme_p1_v1"].isin(["5-10", "<5"]), "somme1"] = "<10"
        df.loc[df.loc[:, "code_somme_p1_v1"].isin(["20-50", "10-20"]), "somme1"] = "10-50"
        df.loc[df.loc[:, typ].isin(["5-10", "<5"]), "sommeN"] = "<10"
        df.loc[df.loc[:, typ].isin(["20-50", "10-20"]), "sommeN"] = "10-50"
        mises = [m for m in df.somme1.unique() if m != "Non réponse"]
        df = df.loc[df.contexte1 != "p1_Non", :]
        contextes = df.loc[:, "contexte1"].unique()
        for c in contextes:
            print(c, typ)
            dfloc = df.loc[df.loc[:, "contexte1"] == c]
            print(dfloc.shape[0])
            flows, labels1, labels2 = list(), list(), list()
            for i1, m1 in enumerate(mises):
                labels1.append(f"init: {m1}")
            for i2, m2 in enumerate(mises):
                labels2.append(f"final: {m2}")
            print(labels1)
            for i1, m1 in enumerate(mises):
                for i2, m2 in enumerate(mises):
                    flo = dfloc.loc[(dfloc.somme1 == m1) & (dfloc.sommeN == m2), :].shape[0]
                    if flo > 0:
                        flows.append((i1, i2+len(labels1), flo))
                        print(m1, m2, flo)
            print([f"{labels1[f[0]]}->{labels2[f[1]-len(labels1)]}:{f[2]}" for f in flows])
            # fig = go.Figure(data=[go.Sankey(
            #     node=dict(
            #         pad=15,
            #         thickness=20,
            #         line=dict(color="black", width=0.5),
            #         label=labels1+labels2,
            #         color="white"
            #     ),
            #     link=dict(
            #         source=[i[0] for i in flows],
            #         target=[i[1] for i in flows],
            #         value=[i[2] for i in flows]
            #     ))])
            #
            # fig.update_layout(title_text=f"Sankey Diagram {c} {typ}", font_size=10)
            # fig.show()
            # fig.write_image(os.path.join(path_data, "poker_exports", f"sankey_full_{c}_{typ}.png"))


if __name__ == "__main__":
    df_fpt = import_fpt()
    # etudes_bivariees(df_fpt)
    predictions(df_fpt)
    # bimodal(df_fpt)
    # stats_adhoc(df_fpt)
    # head_tail(df_fpt)
    # pop_spec(df_fpt)
    # evo_mise_sans_bac(df_fpt)