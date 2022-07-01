from math import sqrt
from statistics import mean
from typing import Union, List, Set, Dict, TextIO

from matplotlib import pyplot as plt
import pandas as pd
import os
import re
import numpy as np
from docx import Document
from nltk import NaiveBayesClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import spacy

nlp = spacy.load('fr_core_news_md')

path_data = os.getcwd()
stopwords = open(os.path.join(path_data, "streaming_bruit", "french_stopwords.txt"),
                 "r", encoding="utf-8").read().split("\n")
junk = ['ve', 'months', 'they', 'for', 'subscribed', 'at', 'Tier']

# plt.rcParams['text.usetex'] = True


def similarites(texte1: str, texte2: str) -> Set[str]:
    return bigrammes(texte1).intersection(bigrammes(texte2))


def count_chars(texte: Union[str, float]) -> int:
    if not isinstance(texte, str):
        return 0
    else:
        return len(" ".join(re.findall(r"[a-zéèàùâêîôûäïëöüç]+", texte.lower())))


def bigrammes(texte: str) -> Set[str]:
    if not isinstance(texte, str):
        return set()
    else:
        words = re.findall(r"[a-zéèàùâêîôûäïëöüç]{2,}", texte.lower())
        return set([words[i] + " " + words[i + 1] for i in range(len(words) - 1)])


def tables_from_docs():
    for adr in [a for a in os.listdir(os.path.join(path_data, "streaming_bruit", "transcriptions")) if
                a[-4:] == "docx"]:
        doc: Document = Document(os.path.join(path_data, "streaming_bruit", "transcriptions", adr))
        trans_table: pd.DataFrame = pd.DataFrame()
        for row in doc.tables[0].rows[2:]:
            trans_table = trans_table.append(pd.Series([c.text for c in row.cells]), ignore_index=True)
        trans_table.to_csv(os.path.join(path_data, "streaming_bruit", "transcriptions", f"{adr[:-4]}csv"), sep=";",
                           encoding="utf-8")


def min2sec(date: Union[str, float]) -> str:
    if "min" in str(date):
        return str(int(re.findall(r"[0-9]+", re.findall(r"[0-9]+\s?min", date)[0])[0]) * 60)
    elif "," in str(date):
        return f"{date.split(',')[0].strip()}.{date.split(',')[0].strip()}"
    else:
        return date


def bagify(texte: str) -> str:
    texte = texte.replace("{", "£").replace("}", "¤")
    texte = re.sub(r"£[^¤]+¤", "", texte)
    bag_of_words: List[str] = re.findall("[a-zâêûîôäëüïöùàçéèo\-]+", texte.lower())
    bag_of_words = [w for w in bag_of_words if w.isalpha() and 1 < len(w) < 22 and w not in stopwords + junk]
    lems = nlp(" ".join(bag_of_words))
    return lems.text


def stats_croisees():
    voc_transcription: List[str] = list()
    com_transcription: List[str] = list()
    for i, adr in enumerate([a for a in os.listdir(os.path.join(path_data, "streaming_bruit", "transcriptions")) if
                             a[-3:] == "csv"]):
        date: str = re.findall(r"[0-9]+-[0-9]+-[0-9]+", adr)[0]
        print(date)
        matching_com = [a for a in os.listdir(os.path.join(path_data, "streaming_bruit", "dataframes_comments")) if
                        date in a]
        transcription: pd.DataFrame = pd.read_csv(os.path.join(path_data, "streaming_bruit", "transcriptions", adr),
                                                  sep=";", encoding='utf-8', index_col=0)
        transcription = transcription.rename({"0": "timing_tr", "1": "timing_com", "2": "actions", "3": "auteur",
                                              "4": "texte"}, axis=1)
        transcription.loc[:, "timing_tr"] = transcription.loc[:, "timing_tr"].apply(min2sec).apply(float)
        transcription.loc[:, "timing_com"] = transcription.loc[:, "timing_com"].apply(min2sec).apply(float)
        transcription.loc[transcription.timing_tr.isna(), "timing_tr"] = \
            transcription.loc[transcription.timing_tr.isna(), "timing_com"]
        start_timing = transcription.timing_tr.to_list()[0]
        end_timing = transcription.timing_tr.to_list()[-1]

        commentaires = pd.read_csv(os.path.join(path_data, "streaming_bruit", "dataframes_comments", matching_com[0]),
                                   sep=";", encoding="utf-8")
        commentaires = commentaires.loc[
            (commentaires.timing > start_timing - 5) & (commentaires.timing < end_timing + 20)]

        transcription = transcription.loc[~transcription.texte.isna(), :]
        commentaires = commentaires.loc[~commentaires.texte.isna(), :]
        voc_transcription.append(bagify(" ".join(transcription.texte.to_list())))
        com_transcription.append(bagify(" ".join(commentaires.texte.to_list())))
        print(len(voc_transcription[-1]), len(com_transcription[-1]))
    nb_pics = len(voc_transcription)
    vectorizer = TfidfVectorizer(max_df=0.5, max_features=5000, min_df=2, input="content")
    X = vectorizer.fit_transform(voc_transcription + com_transcription).toarray()
    features = vectorizer.get_feature_names_out()
    print(X.shape)
    results: pd.DataFrame = pd.DataFrame(np.transpose(X), features)

    topwords: TextIO = open(os.path.join(path_data, "tfidf_pics.csv"), "w", encoding="utf-8")
    for i, adr in enumerate([a for a in os.listdir(os.path.join(path_data, "streaming_bruit", "transcriptions")) if
                             a[-3:] == "csv"]):
        date: str = re.findall(r"[0-9]+-[0-9]+-[0-9]+", adr)[0]
        topwords.write(f"\n{date};stream;" + ";".join(features[np.argsort(X[i, :])[-1:-21:-1]]))
        topwords.write(f"\n;;" + ";".join([str(k) for k in np.sort(X[i, :])[-1:-21:-1]]))
        topwords.write(f"\n{date};chat;" + ";".join(features[np.argsort(X[i + nb_pics, :])[-1:-21:-1]]))
        topwords.write(f"\n;;" + ";".join([str(k) for k in np.sort(X[i + nb_pics, :])[-1:-21:-1]]))
        common_freq = [min(X[i, w], X[i + nb_pics, w]) for w in range(len(features))]
        topwords.write(f"\n{date};both;" + ";".join(features[np.argsort(common_freq)[-1:-21:-1]]))
        topwords.write(f"\n;;" + ";".join([str(k) for k in np.sort(common_freq)[-1:-21:-1]]))
    topwords.close()
    all_textes: List[str] = list()
    for adr in [a for a in os.listdir(os.path.join(path_data, "streaming_bruit", "dataframes_comments"))
                if a[-3:] == "csv"]:
        df: pd.DataFrame = pd.read_csv(os.path.join(path_data, "streaming_bruit", "dataframes_comments", adr),
                                       sep=";", encoding="utf-8")
        all_textes.append(" ".join(set(df.loc[~df.texte.isna(), "texte"].to_list())).lower())
    vect2 = CountVectorizer()
    cvf = pd.DataFrame(np.transpose(vect2.fit_transform(all_textes).toarray()), index=vect2.get_feature_names_out())
    cvf.to_csv(os.path.join(path_data, "frequences_all.csv"), sep=";", encoding="utf-8")
    results.to_csv(os.path.join(path_data, "specificite_voc_pics.csv"), sep=";", encoding="utf-8")


def find_noise():
    specifite: pd.DataFrame = pd.read_csv(os.path.join(path_data, "specificite_voc_pics.csv"), sep=";",
                                          encoding="utf-8", index_col=0)
    nb_streams = specifite.shape[1] // 2
    specifite = specifite.rename({c: f"stream_{int(c)}" if int(c) < nb_streams
    else f"com_{int(c) - nb_streams}" for c in specifite.columns}, axis=1)
    comptage_global: pd.DataFrame = pd.read_csv(os.path.join(path_data, "frequences_all.csv"), sep=";",
                                                encoding="utf-8", index_col=0)
    comptage_global = comptage_global.rename({c: f"doc_{c}" for c in comptage_global.columns}, axis=1)
    comptage_global = comptage_global.reset_index().rename({"index": "mot"}, axis=1)
    comptage_global = comptage_global.loc[~comptage_global.mot.isna(), :]
    comptage_global = comptage_global.loc[comptage_global.mot.apply(lambda z: str(z).isalpha() and len(z) > 2), :]
    comptage_global = comptage_global.loc[~comptage_global.mot.isin(stopwords + junk), :]
    comptage_global.loc[:, "_total_"] = comptage_global.sum(axis=1)
    comptage_global = comptage_global.loc[comptage_global.loc[:, "_total_"] > 50, :].drop("_total_", axis=1) \
        .set_index("mot")
    comptage_global.loc["_total_", :] = comptage_global.sum(axis=0)
    comptage_global = comptage_global.div(comptage_global.loc["_total_", :]).drop("_total_", axis=0)
    comptage_global.loc[:, "variance"] = comptage_global.var(axis=1)
    comptage_global.loc[:, "moyenne"] = comptage_global.mean(axis=1)
    comptage_global.loc[:, "var_rel"] = comptage_global.variance.apply(sqrt).div(comptage_global.moyenne)
    print(specifite.shape, comptage_global.shape)
    gen_table = comptage_global.loc[:, ["variance", "moyenne", "var_rel"]].merge(specifite, how="outer",
                                                                                 left_index=True, right_index=True)
    print(gen_table.info())
    gen_table.loc[:, "is_echo"] = gen_table.apply(echo_score, axis=1)
    gen_table.loc[:, "is_vacarme"] = gen_table.apply(vacarme_score, axis=1)
    gen_table.to_csv(os.path.join(path_data, "spec_vs_freq.csv"), sep=";", encoding="utf-8")


def echo_score(row: pd.Series) -> bool:
    return any([row[f"stream_{c}"] > 0.005 and row[f"com_{c}"] > 0.005 for c in range((len(row) - 3) // 2)])


def vacarme_score(row: pd.Series) -> bool:
    return any([row[f"com_{c}"] > 0.01 for c in range((len(row) - 3) // 2)])


def build_threads():
    voc_transcription: List[str] = list()
    com_transcription: List[str] = list()
    for i, adr in enumerate([a for a in os.listdir(os.path.join(path_data, "streaming_bruit", "transcriptions")) if
                             a[-3:] == "csv"]):
        date: str = re.findall(r"[0-9]+-[0-9]+-[0-9]+", adr)[0]
        print(date)
        matching_com = [a for a in os.listdir(os.path.join(path_data, "streaming_bruit", "dataframes_comments")) if
                        date in a]
        transcription: pd.DataFrame = pd.read_csv(os.path.join(path_data, "streaming_bruit", "transcriptions", adr),
                                                  sep=";", encoding='utf-8', index_col=0)
        transcription = transcription.rename({"0": "timing_tr", "1": "timing_com", "2": "actions", "3": "auteur",
                                              "4": "texte"}, axis=1)
        transcription.loc[:, "nb_chars"] = transcription.loc[:, "texte"].apply(count_chars)
        transcription.loc[:, "texte"] = transcription.loc[:, "texte"].apply(
            lambda z: " ".join(re.findall(r"[a-zéèàùäëïöüçâêîôû]+", str(z).lower())))
        transcription.loc[:, "timing_tr"] = transcription.loc[:, "timing_tr"].apply(min2sec).apply(float)
        transcription.loc[:, "timing_com"] = transcription.loc[:, "timing_com"].apply(min2sec).apply(float)
        transcription.loc[transcription.timing_tr.isna(), "timing_tr"] = \
            transcription.loc[transcription.timing_tr.isna(), "timing_com"]
        start_timing: float = transcription.timing_tr.to_list()[0]
        end_timing: float = transcription.timing_tr.to_list()[-1]
        ms_per_char: float = (end_timing - start_timing) / sum(transcription.nb_chars.to_list())
        timings: List = [start_timing]
        for j, row in transcription.iterrows():
            timings.append(timings[-1] + row["nb_chars"] * ms_per_char)
        transcription.loc[:, "est_timing"] = pd.Series(timings[:-1])
        print(transcription.est_timing.head())

        arrows: pd.DataFrame = pd.DataFrame(
            columns=["timing_stream", "texte_stream", "timing_comment", "texte_comment"])

        commentaires: pd.DataFrame = pd.read_csv(os.path.join(path_data, "streaming_bruit", "dataframes_comments",
                                                              matching_com[0]), sep=";", encoding="utf-8")
        commentaires = commentaires.loc[
            (commentaires.timing > start_timing - 5) & (commentaires.timing < end_timing + 20)]

        transcription = transcription.loc[~transcription.texte.isna(), :]
        commentaires = commentaires.loc[~commentaires.texte.isna(), :]
        commentaires.loc[:, "texte"] = commentaires.loc[:, "texte"].apply(
            lambda z: " ".join(re.findall(r"[a-zéèàùäëïöüçâêîôû]+", str(z).lower())))
        voc_transcription.append(bagify(" ".join(transcription.texte.to_list())))
        com_transcription.append(bagify(" ".join(commentaires.texte.to_list())))
        print(len(voc_transcription[-1]), len(com_transcription[-1]))

        for j, row in transcription.iterrows():
            window_gap: int = 30
            print("timing", row["est_timing"], row["texte"])
            fenetre: pd.DataFrame = commentaires.loc[(commentaires.timing > row["est_timing"] - 1) &
                                                     (commentaires.timing < row["est_timing"] + window_gap), :]
            fenetre.loc[:, "texte"] = fenetre.loc[:, "texte"].apply(lambda z: str(z).lower())
            print("fenêtre", fenetre.head())
            fenetre = fenetre.loc[fenetre.texte.apply(lambda z: len(similarites(z, row["texte"]))) > 0, :]
            print("similarités", fenetre.head())
            if fenetre.shape[0] > 0:
                print("contenu", row["texte"])
                for k, sentence in fenetre.iterrows():
                    arrows = arrows.append(
                        pd.Series({"timing_stream": row["est_timing"], "texte_stream": row["texte"],
                                   "timing_comment": sentence["timing"], "texte_comment": sentence["texte"]}),
                        ignore_index=True)
        arrows.to_csv(os.path.join(path_data, f"similitudes_exemple_{i}.csv"), sep=";", encoding="utf-8")

        plt.figure(figsize=(80, 200))
        timing_current: int = 0
        for j, row in transcription.iterrows():
            mots: List[str] = row["texte"].split() if isinstance(row["texte"], str) else [""]
            blocs: List[str] = [" ".join(mots[k * 8:(k + 1) * 8]) for k in range(len(mots) // 8 + 1)]
            for nb, bloc in enumerate(blocs):
                y = min(-row["est_timing"] - 2 - nb * 0.8, timing_current - 0.8)
                plt.annotate(bloc, (-0.21, y),
                             color="red" if "n me dit" in str(row["texte"]) else "black")
                timing_current = y
        timing_current = 0
        decalage: float = 0
        for j, row in commentaires.iterrows():
            if row["timing"] - timing_current > 0.8:
                timing_current = row["timing"]
                decalage = 0
            plt.annotate(row["texte"], (decalage, -row["timing"]))
            decalage += (len(str(row["texte"])) + 1) * 0.004
        for j, row in arrows.iterrows():
            plt.plot([-0.15, 0.05], [-row["timing_stream"] - 2, -row["timing_comment"]], color="blue")
            plt.fill([-0.005, -0.005, 1, 1], [-row["timing_comment"] - 0.4, -row["timing_comment"] + 0.8,
                                              -row["timing_comment"] + 0.8, -row["timing_comment"] - 0.4], "lightblue")
            mots: List[str] = row["texte_stream"].split() if isinstance(row["texte_stream"], str) else [""]
            blocs: List[str] = [" ".join(mots[k * 8:(k + 1) * 8]) for k in range(len(mots) // 8 + 1)]
            for nb, bloc in enumerate(blocs):
                plt.fill([-0.22, -0.22, -0.04, -0.04], [-row["timing_stream"] - 0.4 - (nb + 2),
                                                        -row["timing_stream"] + 0.8 - (nb + 2),
                                                        -row["timing_stream"] + 0.8 - (nb + 2),
                                                        -row["timing_stream"] - 0.4 - (nb + 2)], "lightyellow")
        plt.ylim(-transcription.loc[transcription.shape[0] - 1, "timing_tr"] - 1,
                 - transcription.loc[0, "timing_tr"])
        plt.xlim(-0.23, 3)
        plt.axis("off")
        plt.savefig(os.path.join(path_data, f"exemple_threads_{i}.svg"))
        plt.close()


def pics_vocabulary(rep_from: str):
    path_peaks = os.path.join(path_data, "streaming_bruit", rep_from)
    comments: Dict[str, pd.DataFrame] = dict()
    for fichier in os.listdir(path_peaks):
        comments[fichier] = pd.read_csv(os.path.join(path_peaks, fichier), sep=";", encoding="utf-8")
    all_comments: pd.DataFrame = pd.concat(comments, axis=0)
    all_comments = all_comments.loc[~all_comments.texte.isna(), :]
    compteur = CountVectorizer()
    vocabulaire: np.ndarray = compteur.fit_transform(all_comments.texte.apply(
        lambda z: " ".join(re.findall(r"[:a-zéèàùâêîôûäïëöüç]{2,}", str(z).lower()))))
    voc_compteur = pd.DataFrame(np.sum(vocabulaire, axis=0)).transpose().rename({0: "nombre"}, axis=1)
    print(voc_compteur.info())
    voc_compteur.loc[:, "mot"] = compteur.get_feature_names()
    voc_compteur = voc_compteur.loc[(voc_compteur.nombre > 1) & ~voc_compteur.mot.isin(stopwords), :]

    voc_compteur = voc_compteur.sort_values(by="nombre", ascending=False).set_index("mot")
    voc_compteur.to_csv(os.path.join(path_data, f"vocabulaire_{rep_from}_.csv"), sep=";", encoding="utf-8")


def decomposition_signal(all_comments: str, common_words: str):
    time_steps: int = 120
    scores: pd.DataFrame = pd.read_csv(os.path.join(path_data, "spec_vs_freq.csv"), sep=";", encoding="utf-8")
    vocabulaire: Set[str] = set(scores.loc[:, "Unnamed: 0"].to_list())
    scores = scores.set_index("Unnamed: 0")
    print(scores.info())
    for comments_file_adr in \
            [fi for fi in os.listdir(os.path.join(path_data, "streaming_bruit", all_comments)) if "10-23" in fi]:
        print(comments_file_adr)
        comments_df: pd.DataFrame = pd.read_csv(
            os.path.join(path_data, "streaming_bruit", all_comments, comments_file_adr),
            sep=";", encoding="utf-8")
        timings = comments_df.timing.to_list()
        frequence_coms: List[int] = [0] * int(max(timings) / time_steps + 1)
        frequence_coms_echo: List[int] = [0] * int(max(timings) / time_steps + 1)
        frequence_coms_bruit: List[int] = [0] * int(max(timings) / time_steps + 1)
        frequence_coms_vacarme: List[int] = [0] * int(max(timings) / time_steps + 1)
        for i, row in comments_df.iterrows():
            sentence: str = bagify(str(row["texte"]))
            nb_echo: int = 0
            nb_vacarme: int = 0
            nb_bruit: int = 0
            nb_autres: int = 0
            flag_content: bool = False
            for word in sentence.split():
                if word in vocabulaire:
                    if len(word) > 2:
                        if scores.loc[word, "var_rel"] < 1.5:
                            nb_bruit += 1
                            flag_content = True
                        if scores.loc[word, "is_echo"]:
                            nb_echo += 1
                        if scores.loc[word, "is_vacarme"]:
                            nb_vacarme += 1
                        if scores.loc[word, "var_rel"] >= 1.5 and not scores.loc[word, "is_echo"] \
                                and not scores.loc[word, "is_vacarme"]:
                            nb_autres += 1
            if len(set(sentence.split())) > 8:
                if nb_echo >= 1:
                    what_type = "écho"
                else:
                    what_type = "bruit"
            elif len(sentence.split()) > 5 and len(set(sentence.split())) <= 2:
                what_type = "vacarme"
            elif len(sentence.split()) > 3 and len(set(sentence.split())) == 1:
                what_type = "vacarme"
            elif nb_echo >= 1:
                what_type = "écho"
            elif nb_vacarme >= 1:
                what_type = "vacarme"
            elif nb_bruit >= 1:
                what_type = "bruit"
            else:
                what_type = "inconnu"
            if what_type == "écho":
                frequence_coms_echo[int(row["timing"] / time_steps)] += 1
            elif what_type == "vacarme":
                frequence_coms_vacarme[int(row["timing"] / time_steps)] += 1
            elif what_type == "bruit":
                frequence_coms_bruit[int(row["timing"] / time_steps)] += 1
            else:
                frequence_coms[int(row["timing"] / time_steps)] += 1
            print(sentence, what_type)
        print(comments_file_adr, sum(frequence_coms_echo), sum(frequence_coms_vacarme), sum(frequence_coms_bruit),
              sum(frequence_coms))

        norm: int = mean(frequence_coms)
        plt.figure(figsize=(15, 10))
        plt.plot(range(len(frequence_coms_echo)), frequence_coms_echo, label='écho', color="green")
        plt.plot(range(len(frequence_coms_bruit)), frequence_coms_bruit, label='bruit', color="orange")
        plt.plot(range(len(frequence_coms_vacarme)), frequence_coms_vacarme, label='vacarme', color="red")
        plt.plot(range(len(frequence_coms)), frequence_coms, label='non identifié', color="lightgrey")
        plt.xlabel(f"Temps (intervalles de {time_steps}s.)")
        plt.ylabel("Nombre de commentaires")
        plt.legend()
        plt.savefig(
            os.path.join(path_data, "streaming_bruit", "graphes", f"decomposition_{comments_file_adr[:10]}.png"))
        plt.close()


def classifications():
    labeled_comments: pd.DataFrame = pd.read_csv(os.path.join(path_data, "9-3-21_v2.csv"), encoding="utf-8", sep=";")
    labeled_comments = labeled_comments.loc[~labeled_comments.Y.isna(), :]
    labeled_comments.loc[labeled_comments.Y == "bruit", "Y"] = "vacarme"
    print(labeled_comments.info())
    train_set = labeled_comments.iloc[[i for i in range(labeled_comments.shape[0]) if i % 3 == 0 or i % 3 == 1], :]
    test_set = labeled_comments.iloc[[i for i in range(labeled_comments.shape[0]) if i % 3 == 2], :]
    words = re.findall(r"[a-zéèàäëïöüâêîôûçù]{2,}", " ".join(labeled_comments.loc[:, "text"].to_list()).lower())
    print(words)
    classifierNB = NaiveBayesClassifier.train([(extract_features(row["text"], words), row["Y"])
                                               for i, row in train_set.iterrows()])
    print(classifierNB.show_most_informative_features(20))
    test_result = pd.DataFrame({"text": test_set.loc[:, "text"], "Y_real": test_set.loc[:, "Y"],
                                "Y_pred": classifierNB.classify_many([(extract_features(row["text"], words))
                                                                 for i, row in test_set.iterrows()])})

    scores: pd.DataFrame = pd.read_csv(os.path.join(path_data, "spec_vs_freq.csv"), sep=";", encoding="utf-8")
    vocabulaire: Set[str] = set(scores.loc[:, "Unnamed: 0"].to_list())
    scores = scores.set_index("Unnamed: 0")
    test_result.loc[:, "Y_mano"] = np.nan
    for i, row in test_result.iterrows():
        sentence: str = bagify(str(row["text"]))
        nb_echo: int = 0
        nb_vacarme: int = 0
        for word in sentence.split():
            if word in vocabulaire:
                if len(word) > 2:
                    if scores.loc[word, "var_rel"] < 1.5:
                        nb_vacarme += 1
                    if scores.loc[word, "is_echo"]:
                        nb_echo += 1
                    if scores.loc[word, "is_vacarme"]:
                        nb_vacarme += 1
        if len(set(sentence.split())) > 8:
            if nb_echo >= 1:
                what_type = "écho"
            else:
                what_type = "vacarme"
        elif len(sentence.split()) > 5 and len(set(sentence.split())) <= 2:
            what_type = "vacarme"
        elif len(sentence.split()) > 3 and len(set(sentence.split())) == 1:
            what_type = "vacarme"
        elif nb_echo >= 1:
            what_type = "écho"
        elif nb_vacarme >= 1:
            what_type = "vacarme"
        else:
            what_type = "inconnu"
        test_result.loc[i, "Y_mano"] = what_type
    classes = ["vacarme", "echo"]
    test_result.to_csv(os.path.join(path_data, "test_détection.csv"), sep=";", encoding="utf-8")
    for cl1 in classes:
        for cl2 in classes:
            print(cl1, cl2, test_result.loc[(test_result.Y_real == cl1) & (test_result.Y_pred == cl2), :].shape[0])


def extract_features(texte: str, words) -> Dict[str, int]:
    return {w: texte.count(w) for w in words}


def comptage_simple():
    nombre_comments: int = 0
    nombre_mots: int = 0
    for comments_file_adr in os.listdir(os.path.join(path_data, "streaming_bruit", "dataframes_comments")):
        df = pd.read_csv(os.path.join(path_data, "streaming_bruit", "dataframes_comments", comments_file_adr), sep=";",
                         encoding="utf-8")
        nombre_comments += df.shape[0]
        nombre_mots += df.loc[:,  "texte"].apply(lambda z: len(str(z).split(" "))).sum()
    print(f"{nombre_comments} lignes, {nombre_mots} mots")


if __name__ == "__main__":
    # tables_from_docs()
    # stats_croisees()
    # find_noise()
    # build_threads()
    # pics_vocabulary("dataframes_comments")
    # pics_vocabulary("common_words")
    # decomposition_signal("dataframes_comments", "common_words")
    classifications()
    # comptage_simple()
