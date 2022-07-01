import os
import re
from typing import TextIO, Dict, List
import pandas as pd
from matplotlib import pyplot as plt
from PyPDF2 import PdfFileReader, PdfFileWriter
from french_lefff_lemmatizer.french_lefff_lemmatizer import FrenchLefffLemmatizer as FLF
from nltk import FreqDist, NaiveBayesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, roc_curve, auc
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_graphviz
from tika import parser
import numpy as np

path_data: str = os.getcwd()
all_sources: List[str] = ["europresse 1997 2010", "europresse 2011 2021"]
journaux: List[str] = ["Le Progrès", "La Voix du Nord", "Le Télégramme", "L'Est Républicain", "La Montagne",
                       "La Dépêche du Midi", "La République du Centre", "L'Indépendant"]
modalites = {"grèv", "manifest", "pétition", "rassembl", "occupation", "blocage", "bloquer", "occuper", "barrage",
             "diffus", "tract", "saccage", "vandalis"}
acteurs = {"étudiant", "chômeu", "ouvrier", "riverain", "locataire", "agricult", "salarié", "migrant", "réfugié",
           "parent", "lycéen", "exploitant", "retraité", "enseignant", "fonctionnaire", "professeur"}
groupements = {"syndicat", "association", "collectif"}
enjeux = {"salaire", "35 h", "35h", "retraite", "sécu", "augmentation", "conditions", "travail", "licenciement",
          "plan social"}
keywords = modalites.union(acteurs, groupements, enjeux)
modalites = ["Manifestation", "Grève", "Pétition papier", "Pétition en ligne", "Irruption locaux ou réunion",
             "Occupation de locaux", "Blocages routes marchandises OpEscargots voitures", "boycott",
             "Jet de projectiles en groupes", "Saccage locaux marchandises",
             "Grève de la faim", "Blocage lieu ou outils d'activité"]


def split_pdf(pdf_rep: str, pages_rep: str):
    for pdf_file in os.listdir(os.path.join(path_data, pdf_rep)):
        print(pdf_file)
        inputpdf = PdfFileReader(open(os.path.join(path_data, pdf_rep, pdf_file), "rb"))
        for i in range(inputpdf.numPages):
            print(i)
            output = PdfFileWriter()
            output.addPage(inputpdf.getPage(i))
            with open(os.path.join(path_data, pages_rep, f"{pdf_file}_{i}s.pdf"), "wb") as outputStream:
                output.write(outputStream)


def pdf2txt(pages_rep: str, text_rep: str):
    for pdf_file in os.listdir(os.path.join(path_data, pages_rep)):
        print(pdf_file)
        raw: Dict[str] = parser.from_file(os.path.join(path_data, pages_rep, pdf_file))
        txt_file: TextIO = open(os.path.join(path_data, text_rep, pdf_file[:-3] + "txt"), "w", encoding="utf-8")
        txt_file.write(raw["content"])
        txt_file.close()


def post_traitement(text_rep: str, db_rep: str):
    ordered_list_file: List[str] = os.listdir(os.path.join(path_data, text_rep))
    ordered_list_file.sort(key=lambda z: (int(z[:2]) if z[:2].isnumeric() else int(z[12:14])) * 1000
                                         + int(z.split("pdf_")[1].split("s.")[0]))
    nom_export: str = ""
    for i, txt_file in enumerate(ordered_list_file):
        txt_file_io: TextIO = open(os.path.join(path_data, text_rep, txt_file), "r", encoding="utf-8")
        texte_brut: str = txt_file_io.read()
        txt_file_io.close()
        texte_brut = re.sub(r"\n+", "\n", texte_brut)
        if "©" in texte_brut:
            texte_brut = texte_brut.split("©")[0]
        dates: List[str] = re.findall("[A-Z]?[a-z]+[dh][ie]\s[0-9]{,2}\s[A-Z]?[a-zéû]+\s[12][09][0-9]{2}", texte_brut)
        real_date: List[str] = [date for date in dates if date[-9:] != "obre 2021"]
        if real_date:
            journal = re.findall("|".join(journaux), texte_brut)
            formatted_journal: str = journal[0].replace("'", "_").replace(" ", "_") if journal else "inconnu"
            formatted_date: str = f"{real_date[0][-4:]}_{real_date[0][:-5]}"
            nom_export: str = f"{formatted_date}_{formatted_journal}_{i}.txt"
            file_write: TextIO = open(os.path.join(path_data, db_rep, nom_export), "w", encoding="utf-8")
            file_write.write(texte_brut)
            file_write.close()
        else:
            if "2021_jeudi 30 septembre" in nom_export and dates:
                journal: List[str] = re.findall("|".join(journaux), texte_brut)
                formatted_journal: str = journal[0].replace("'", "_").replace(" ", "_") if journal else "inconnu"
                formatted_date: str = f"{dates[0][-4:]}_{dates[0][:-5]}"
                nom_export_new: str = f"{formatted_date}_{formatted_journal}_{i}.txt"
                file_write: TextIO = open(os.path.join(path_data, db_rep, nom_export_new), "w", encoding="utf-8")
                file_write.write(texte_brut)
                file_write.close()
            else:
                file_write: TextIO = open(os.path.join(path_data, db_rep, nom_export), "a", encoding="utf-8")
                file_write.write(texte_brut)
                file_write.close()


def build_table(files: str):
    list_files: List[str] = os.listdir(os.path.join(path_data, files))
    metadata = pd.DataFrame({"nom_fichier": list_files})
    metadata.loc[:, "annee"] = metadata.loc[:, "nom_fichier"].apply(lambda z: z.split("_")[0])
    metadata.loc[:, "date"] = metadata.loc[:, "nom_fichier"].apply(lambda z: z.split("_")[1])
    metadata.loc[:, "info"] = metadata.loc[:, "nom_fichier"].apply(lambda z: " ".join(z.split("_")[2:]).split(".")[0])
    metadata.loc[:, "mois"] = metadata.loc[:, "date"].apply(lambda z: z.split(" ")[2])
    metadata.loc[:, "jour"] = metadata.loc[:, "date"].apply(lambda z: z.split(" ")[1])
    metadata.loc[:, "journal"] = metadata.loc[:, "info"].apply(lambda z: " ".join(z.split(" ")[:-1]))
    metadata.loc[:, "keywords"] = np.nan
    metadata.loc[:, "lien fichier"] = metadata.loc[:, "nom_fichier"]
    metadata = metadata.set_index("nom_fichier")
    metadata = metadata.drop(["date", "info"], axis=1)
    for file in list_files:
        file_io: TextIO = open(os.path.join(path_data, files, file), encoding="utf-8")
        content: str = file_io.read()
        file_io.close()
        raw_content = " ".join(re.findall(r"[a-zéèàùçäëïöüâêîôû]+", content.lower()))
        kw_here: List[str] = [word for word in keywords if word in raw_content]
        print(kw_here)
        metadata.loc[file, "keywords"] = f"{kw_here}"

        if int(file[:4]) < 2011:
            try:
                metadata.loc[file, "header"] = " ".join(content.split("mots\n")[1].split("\n")[:4]).replace("\n", "")
            except IndexError as ie:
                print(file)
        else:
            if re.search(r"mots, p\. .*", content):
                metadata.loc[file, "header"] = " ".join(re.split(r"mots, p\. .*",
                                                                 content)[1].split("\n")[:4]).replace("\n", "")
            else:
                try:
                    metadata.loc[file, "header"] = " ".join(re.split(r"mots",
                                                                     content)[1].split("\n")[:4]).replace("\n", "")
                except IndexError as ie:
                    print(file)

    metadata.to_csv(os.path.join(path_data, "table_recap.csv"), sep=";", encoding="utf-8", index=None)


def base_stats():
    metadata: pd.DataFrame = pd.read_csv(os.path.join(path_data, "table_recap.csv"), sep=";", encoding="utf-8")
    plt.figure(figsize=(15, 10))
    an_min, an_max = 1997, 2022
    x = range(an_min, an_max)
    somme = [0] * (an_max - an_min)
    print(metadata.journal.unique())
    for journal in metadata.journal.unique():
        print(journal)
        y = [metadata.loc[(metadata.journal == journal) & (metadata.annee == annee)].shape[0] for annee in x]
        print(y)
        plt.bar(x, height=y, bottom=somme, label=journal)
        somme = [somme[annee] + y[annee] for annee in range(an_max - an_min)]
    plt.legend()
    plt.savefig(os.path.join(path_data, "recap_all.png"), dpi=300)
    plt.close()


def lemmatize_data():
    metadata: pd.DataFrame = pd.read_csv(os.path.join(path_data, "table_recap.csv"), sep=";", encoding="utf-8")
    lemmatizer = FLF()
    for i, row in metadata.iterrows():
        print(row["lien fichier"])
        fichier_in: TextIO = open(os.path.join(path_data, "file_db", row["lien fichier"]), "r", encoding="utf-8")
        texte: str = re.sub("-\n", "", fichier_in.read().lower())
        texte = " ".join(re.findall(r"[a-zéèàùçâêûîôäëüïö35]+", texte))
        fichier_in.close()
        fichier_out: TextIO = open(os.path.join(path_data, "lem_db", row["lien fichier"]), "w", encoding="utf-8")
        fichier_out.write(" ".join(lemmatizer.lemmatize(token) for token in texte.split()))
        fichier_out.close()


def aggregate_lems():
    metadata: pd.DataFrame = pd.read_csv(os.path.join(path_data, "table_recap.csv"), sep=";", encoding="utf-8")
    metadata.loc[:, "texte"] = metadata.loc[:, "lien fichier"].apply(get_texte)
    metadata.to_csv(os.path.join(path_data, "table_recap_big.csv"), sep=";", encoding="utf-8")


def get_texte(adr: str) -> str:
    fichier_in: TextIO = open(os.path.join(path_data, "lem_db", adr), "r", encoding="utf-8")
    texte = fichier_in.read()
    fichier_in.close()
    return texte


def clean_keywords():
    df_keywords: pd.DataFrame = pd.read_csv(os.path.join(path_data, "keywords.csv"), sep=";", encoding="utf-8",
                                            header=None).rename({0: "categorie", 1: "expression"}, axis=1)
    print(df_keywords.info())
    lemmatizer = FLF()
    metadata: pd.DataFrame = pd.read_csv(os.path.join(path_data, "table_recap_big.csv"), sep=";", encoding="utf-8")
    metadata2: pd.DataFrame = metadata.copy()
    print(metadata.info())
    all_expressions = list()
    for i, row in df_keywords.iterrows():
        print(row["categorie"])
        disjonction: List[str] = row["expression"].split("|")
        conjonctions: List[List[str]] = [clause.split("&") for clause in disjonction]
        conjonctions = [[" ".join([lemmatizer.lemmatize(token)
                                   for token in subclause.replace('"', '').lower().split(" ")])
                         for subclause in clause] for clause in conjonctions]
        print(conjonctions)
        all_expressions += [c for c in conjonctions]
        metadata.loc[:, row["categorie"]] = metadata.loc[:, "texte"].apply(
            lambda z: find_expression(z, conjonctions))
    metadata = metadata.drop(["Unnamed: 0", "keywords", "lien fichier", "header", "texte"], axis=1)
    metadata.to_csv(os.path.join(path_data, "categories.csv"), sep=";", encoding="utf-8")
    liste_all_expressions: List[str] = [e for liste in all_expressions for e in liste]
    for expr in liste_all_expressions:
        metadata2.loc[:, expr] = metadata2.loc[:, "texte"].apply(lambda z: expr in set(z.split()))
    metadata2.to_csv(os.path.join(path_data, "categories2.csv"), sep=";", encoding="utf-8")


def find_expression(texte: str, expr: List[List[str]]):
    return any([all([trouver_cl(subclause, texte) for subclause in clause]) for clause in expr])


def trouver_cl(subclause: str, texte: str):
    return subclause[1:] not in texte if subclause[0] == "^" else subclause in texte


def correlations():
    actions: pd.DataFrame = pd.read_csv(os.path.join(path_data, "categories.csv"), sep=";", encoding="utf-8",
                                        index_col=0)
    correls: pd.DataFrame = actions.drop(["annee", "jour"], axis=1).corr()
    for i, row in correls.iterrows():
        print(i)
        print(row.sort_values(ascending=False)[1:5])


def find_modalite(row: pd.Series) -> str:
    return "&".join([clef for clef in modalites if row[clef]])


def build_classes():
    actions: pd.DataFrame = pd.read_csv(os.path.join(path_data, "categories.csv"), sep=";", encoding="utf-8",
                                        index_col=0)
    actions.loc[:, "local commune département"] = actions.apply(
        lambda z: z["local commune département"] and not z["immeuble cit‚ quartier"], axis=1)
    actions.loc[:, "rentrée scolaire"] = actions.apply(lambda z: z["mois"] == "septembre" and int(z["jour"]) < 16,
                                                       axis=1)
    for modalite in ["Pétition papier"]:
        mod_actions = actions.loc[actions.loc[:, modalite]==True, :]
        print(modalite, mod_actions.shape[0])
        compte = pd.Series({c: mod_actions.loc[mod_actions.loc[:, c], :].shape[0]
                               / actions.loc[actions.loc[:, c], :].shape[0]
                            for c in [cx for cx in mod_actions.columns[9:]
                                      if actions.loc[actions.loc[:, cx], :].shape[0] > 0]})
        print(compte.sort_values().tail(20))

    complete: pd.DataFrame = pd.read_csv(os.path.join(path_data, "table_recap_big.csv"), sep=";", encoding="utf-8",
                                         index_col=0).loc[:, ["lien fichier", "header"]]

    print(actions.columns.to_list())
    actions.loc[:, "modalite"] = actions.apply(find_modalite, axis=1)
    actions = actions.loc[actions.modalite != "", :]
    print(actions.loc[actions.loc[:, "Grève de la faim"], :].index.to_list())
    main_mods = actions.groupby("modalite").modalite.count().sort_values(ascending=False).index[1:12]

    plt.figure(figsize=(15, 10))
    an_min, an_max = 1997, 2022
    x = range(an_min, an_max)
    somme = [0] * (an_max - an_min)
    for modalite in main_mods:
        y = [actions.loc[(actions.modalite == modalite) & (actions.annee == annee)].shape[0] for annee in x]
        plt.bar(x, height=y, bottom=somme, label=modalite)
        somme = [somme[annee] + y[annee] for annee in range(an_max - an_min)]
    y_no: List[int] = [actions.loc[(actions.modalite == "") & (actions.annee == annee)].shape[0] for annee in x]
    plt.bar(x, height=y_no, bottom=somme, label="inconnu", color="lightgray")
    plt.legend()
    plt.savefig(os.path.join(path_data, "modalites_onlypositive.png"))
    plt.close()

    side_cols = ["jour", "mois", "annee", "journal", "modalite"]
    taux_conversion: pd.DataFrame = pd.DataFrame(columns=modalites)
    for incentive in [col for col in actions.columns
                      if col not in side_cols and actions.loc[(actions.loc[:, col]), :].shape[0] > 0]:
        x = pd.DataFrame(
            {result: f"{actions.loc[(actions.loc[:, incentive]) & (actions.loc[:, result]), :].shape[0]} / "
                     f"{actions.loc[(actions.loc[:, incentive]), :].shape[0]}"
             for result in taux_conversion.columns}, index=[incentive])
        print(x)
        taux_conversion = taux_conversion.append(x)
        print(taux_conversion)
    taux_conversion.to_csv(os.path.join(path_data, "conversion_effectifs.csv"), sep=";", encoding="utf-8")

    for loc in [actions, actions.loc[actions.annee < 2001, :], actions.loc[actions.annee >= 2001, :]]:
        scol = loc.loc[actions.loc[:, "scolaire"], :]
        print("all", scol.loc[scol.loc[:, "Occupation de locaux"]].shape[0], "/", scol.shape[0],
              scol.loc[scol.loc[:, "Occupation de locaux"]].shape[0] / scol.shape[0])
        add_pool = ["parents", "fermetures de classe", "rentrée scolaire"]
        for i, add in enumerate(add_pool):
            scol2 = scol.loc[scol.loc[:, add], :]
            print(add, scol2.loc[scol.loc[:, "Occupation de locaux"]].shape[0], "/", scol2.shape[0],
                  scol2.loc[scol.loc[:, "Occupation de locaux"]].shape[0] / scol2.shape[0])
            for j, add2 in enumerate(add_pool):
                if j > i:
                    scol3 = scol2.loc[scol2.loc[:, add2], :]
                    print(add, add2, scol3.loc[scol.loc[:, "Occupation de locaux"]].shape[0], "/", scol3.shape[0],
                          scol3.loc[scol.loc[:, "Occupation de locaux"]].shape[0]/scol3.shape[0])
        scol_all = scol.loc[scol.loc[:, add_pool[0]] & scol.loc[:, add_pool[1]] & scol.loc[:, add_pool[2]], :]
        print("all", scol_all.loc[scol.loc[:, "Occupation de locaux"]].shape[0], "/", scol_all.shape[0],
              scol_all.loc[scol.loc[:, "Occupation de locaux"]].shape[0]/scol_all.shape[0])

    add_pool = ["habitants et locataires", "nuisance locale et environnement", "immeuble cit‚ quartier",
                "local commune département"]
    for modal in modalites:
        print("******", "\n", modal)
        for i, add in enumerate(add_pool):
            scol2 = actions.loc[actions.loc[:, add], :]
            print(add, scol2.loc[scol2.loc[:, modal]].shape[0], "/", scol2.shape[0],
                  scol2.loc[scol2.loc[:, modal]].shape[0] / scol2.shape[0] if scol2.shape[0] > 0 else "N/A")
            for j, add2 in enumerate(add_pool):
                if j > i:
                    scol3 = scol2.loc[scol2.loc[:, add2], :]
                    print(add, "+", add2, scol3.loc[scol3.loc[:, modal]].shape[0], "/", scol3.shape[0],
                          scol3.loc[scol3.loc[:, modal]].shape[0] /scol3.shape[0] if scol3.shape[0] > 0 else "N/A")
                    for k, add3 in enumerate(add_pool):
                        if k > j:
                            scol4 = scol3.loc[scol3.loc[:, add3], :]
                            print(add, "+", add2, "+", add3, scol4.loc[scol4.loc[:, modal]].shape[0], "/",
                                  scol4.shape[0], scol4.loc[scol4.loc[:, modal]].shape[0] / scol4.shape[0] if scol4.shape[0] > 0 else "N/A")

    actions2 = actions.loc[actions.annee > 2010, :]
    manifs_imm = actions2.loc[actions.loc[:, "immeuble cit‚ quartier"] & actions.loc[:, "Manifestation"],
                              ["jour", "mois", "annee", "journal"]].merge(complete, how="inner", left_index=True,
                                                                          right_index=True)
    manifs_imm.loc[:, "header"] = manifs_imm.loc[:, "lien fichier"].apply(
        lambda z: open(os.path.join(path_data, "lem_db", z), "r", encoding="utf-8").read())
    manifs_imm.to_csv(os.path.join(path_data, "manifs_quartier.csv"), sep=";", encoding="utf-8")
    petitions_imm = actions2.loc[actions.loc[:, "immeuble cit‚ quartier"] & actions.loc[:, "Pétition papier"],
                                 ["jour", "mois", "annee", "journal"]].merge(complete, how="inner", left_index=True,
                                                                             right_index=True)
    petitions_imm.loc[:, "header"] = petitions_imm.loc[:, "lien fichier"].apply(
        lambda z: open(os.path.join(path_data, "lem_db", z), "r", encoding="utf-8").read())
    petitions_imm.to_csv(os.path.join(path_data, "petition_quartier.csv"), sep=";", encoding="utf-8")

    train: pd.DataFrame = actions.iloc[[i for i in range(actions.shape[0]) if i % 3 < 2], :]
    test: pd.DataFrame = actions.iloc[[i for i in range(actions.shape[0]) if i % 3 == 2], :]
    X_train: pd.DataFrame = train.loc[:, [c for c in test.columns if c not in modalites + side_cols]]
    X_test: pd.DataFrame = test.loc[:, [c for c in test.columns if c not in modalites + side_cols]]
    for modalite in modalites:
        print(modalite)
        Y_train: pd.Series = train.loc[:, modalite]
        Y_test: pd.Series = test.loc[:, modalite]
        classifier = RandomForestClassifier(class_weight="balanced")
        classifier.fit(X_train, Y_train)
        plt.figure(figsize=(15, 10))
        important_features: List[int] = [i for i in range(len(classifier.feature_importances_))
                                         if classifier.feature_importances_[i] > 0.025]
        plt.barh(X_train.columns[important_features], classifier.feature_importances_[important_features])
        plt.savefig(os.path.join(path_data, f"featimp_{modalite}.png"))
        plt.close()
        Y_pred = classifier.predict(X_test)
        acc_score = accuracy_score(Y_test, Y_pred)
        pre_score = precision_score(Y_test, Y_pred)
        rec_score = recall_score(Y_test, Y_pred)
        conf_mat = confusion_matrix(Y_test, Y_pred)
        fpr, tpr, _ = roc_curve(Y_test, classifier.predict_proba(X_test)[:, 1], pos_label=1)
        roc_auc = auc(fpr, tpr)
        print(f"Acc {acc_score}", f"Prec {pre_score}", f"Rec {rec_score}", f"AUC {roc_auc}")
        print(conf_mat)

        plt.figure()
        plt.plot(fpr, tpr)
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        plt.savefig(os.path.join(path_data, f"roc_{modalite}.png"))
        plt.close()

        tree = DecisionTreeClassifier(max_depth=5)
        cl_tree = tree.fit(X_train, Y_train)
        plot_tree(cl_tree)
        dot_data = export_graphviz(cl_tree, out_file=os.path.join(path_data, f"tree_{modalite}"),
                                   feature_names=[c for c in test.columns if c not in modalites + side_cols])


def supervise():
    texte: pd.Series = pd.read_csv(os.path.join(path_data, "categories2.csv"), sep=";", encoding="utf-8",
                                        index_col=0).loc[:, ["texte"]]
    petition: pd.Series = pd.read_csv(os.path.join(path_data, "categories.csv"), sep=";", encoding="utf-8",
                                        index_col=0).loc[:, ["Pétition papier"]]
    df: pd.DataFrame = pd.concat([texte, petition], axis=1)
    lmax = 2000
    print("building dictionnary")
    texte1 = " ".join(df.loc[df.loc[:, "Pétition papier"]==True, "texte"].tolist()).split(" ")
    texte0 = " ".join(df.loc[df.loc[:, "Pétition papier"]==False, "texte"].tolist()).split(" ")
    fdist1 = FreqDist(texte1).most_common(lmax)
    fdist0 = FreqDist(texte0).most_common(lmax)
    words = set([x[0] for x in fdist1]).union(set([x[0] for x in fdist0]))
    words = [w for w in words if len(w) > 3]
    df.loc[:, [f"has_{word}" for word in words]] = df.texte.apply(
        lambda z: pd.Series([word in z for word in words], index=[f"has_{word}" for word in words]))
    print(df.head())
    classifierNB = NaiveBayesClassifier.train([(row[[f"has_{w}" for w in words]], row["Pétition papier"])
                                               for i, row in df.iterrows()])
    print(classifierNB.show_most_informative_features(30))


def classes_all_voc():
    actions: pd.DataFrame = pd.read_csv(os.path.join(path_data, "yolo2.csv"), sep=";", encoding="utf-8",
                                        index_col=0)
    print(actions.columns.to_list())
    print(actions.info())
    side = ["Unnamed: 0.1", "annee", "mois", "jour", "journal", "keywords", "lien fichier", "header", "texte"]
    actions = actions.loc[:, [c for c in actions.columns if c not in side]]
    print(actions.columns.to_list())
    categories: pd.DataFrame = pd.read_csv(os.path.join(path_data, "categories.csv"), sep=";", encoding="utf-8",
                                        index_col=0)
    print(categories.columns.to_list())
    print(categories.info())
    actions = pd.concat([actions, categories.loc[:, modalites]], axis=1)
    print(actions.info())

    train: pd.DataFrame = actions.iloc[[i for i in range(actions.shape[0]) if i % 3 < 2], :]
    test: pd.DataFrame = actions.iloc[[i for i in range(actions.shape[0]) if i % 3 == 2], :]
    X_train: pd.DataFrame = train.loc[:, [c for c in test.columns if c not in modalites]]
    X_test: pd.DataFrame = test.loc[:, [c for c in test.columns if c not in modalites]]
    for modalite in modalites:
        print(modalite)
        Y_train: pd.Series = train.loc[:, modalite]
        Y_test: pd.Series = test.loc[:, modalite]
        classifier = RandomForestClassifier(class_weight="balanced")
        classifier.fit(X_train, Y_train)
        plt.figure(figsize=(15, 10))
        important_features: List[int] = [i for i in range(len(classifier.feature_importances_))
                                         if classifier.feature_importances_[i] > 0.025]
        plt.barh(X_train.columns[important_features], classifier.feature_importances_[important_features])
        plt.savefig(os.path.join(path_data, f"featimp_{modalite}.png"))
        plt.close()
        Y_pred = classifier.predict(X_test)
        acc_score = accuracy_score(Y_test, Y_pred)
        pre_score = precision_score(Y_test, Y_pred)
        rec_score = recall_score(Y_test, Y_pred)
        conf_mat = confusion_matrix(Y_test, Y_pred)
        fpr, tpr, _ = roc_curve(Y_test, classifier.predict_proba(X_test)[:, 1], pos_label=1)
        roc_auc = auc(fpr, tpr)
        print(f"Acc {acc_score}", f"Prec {pre_score}", f"Rec {rec_score}", f"AUC {roc_auc}")
        print(conf_mat)

        plt.figure()
        plt.plot(fpr, tpr)
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        plt.savefig(os.path.join(path_data, f"roc_{modalite}.png"))
        plt.close()

        tree = DecisionTreeClassifier(max_depth=5)
        cl_tree = tree.fit(X_train, Y_train)
        plot_tree(cl_tree)
        dot_data = export_graphviz(cl_tree, out_file=os.path.join(path_data, f"tree_{modalite}"),
                                   feature_names=[c for c in test.columns if c not in modalites])
        plt.show()


if __name__ == "__main__":
    # for source in all_sources:
    # split_pdf(source, "pages")
    # pdf2txt("pages", "ocr")
    # post_traitement("ocr", "file_db")
    # build_table("file_db")
    # base_stats()
    # lemmatize_data()
    # aggregate_lems()
    # clean_keywords()
    # correlations()
    # build_classes()
    classes_all_voc()
    # supervise()
