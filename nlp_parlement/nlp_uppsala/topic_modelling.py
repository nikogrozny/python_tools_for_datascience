import os
from itertools import combinations
from typing import Tuple, List, Dict, Any, TextIO
import pandas as pd
import numpy as np
import spacy

nlp = spacy.load('fr_core_news_sm')
from nltk import FreqDist, NaiveBayesClassifier
from matplotlib import pyplot as plt
import re
import plotly.io as pio
import plotly.graph_objects as go
from nltk.corpus import stopwords
from nltk import word_tokenize
from french_lefff_lemmatizer.french_lefff_lemmatizer import FrenchLefffLemmatizer as FLF
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel, LdaModel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pickle

lemmatizer = FLF()

path_data = os.getcwd()
path_to_model = os.path.join(path_data, "model_ML")
stop_words = set(list(open(os.path.join(path_data, "french_stopwords.txt"), "r", encoding="utf-8").read().split("\n")))


def table_ocr_1880():
    print("récupération des fichiers")
    fichiers = os.listdir(os.path.join(path_data, "blocs"))
    data = pd.DataFrame(columns=["date", "text"])
    for name in fichiers:
        print(name)
        if name.split("-")[0][:3] == '188':
            try:
                file: TextIO = open(os.path.join(path_data, "blocs", name), "r", encoding="utf-8")
                text: str = file.read()
                file.close()
                data = data.append({'date': name.split('.')[0], "text": text}, ignore_index=True)
            except Exception:
                pass
    data.to_csv(os.path.join(path_data, "textes_1880s.csv"))
    return data


def build_corpus(df: pd.DataFrame):
    print("nettoyage du texte")
    data: List = []
    for i in range(df.shape[0]):
        print(f"{i}/{df.shape[0]}")
        text = " ".join(re.findall("[A-Za-zâêûîôäëüïöùàçéèÉ\-\.]+", df.text[i]))
        text = re.sub("([a-z])- ", r"\1", text)
        text = re.sub("\-", " ", text)
        text = re.sub("[M]+\. ([A-Z]+[a-zâêûîôäëüïöùàçéè]+(?:\s[A-Z]+[a-zâêûîôäëüïöùàçéè]+)?)", " ", text)
        text = re.sub("\.", " ", text)
        bag_of_words: List[str] = word_tokenize(text.lower(), language="french")
        bag_of_words = [w for w in bag_of_words if 1 <= len(w) < 22]
        bag_of_words = [lemmatizer.lemmatize(w).lower() for w in bag_of_words if w not in stop_words]
        data.append(bag_of_words)
    df.loc[:, "bag_of_words"] = data
    df.to_csv(os.path.join(path_data, "textes_1880s.csv"))
    return df


def count_vectorizer(df: pd.DataFrame, p: int):
    print("compte des coccurences")
    data = [" ".join(w) for w in df.bag_of_words]
    vectorizer = CountVectorizer(max_features=p)
    X = vectorizer.fit_transform(data)
    word_frequency_matrix = pd.DataFrame(data=X.toarray(), index=df.date, columns=vectorizer.get_feature_names())
    word_frequency_matrix = word_frequency_matrix.sort_index()
    word_frequency_matrix.to_csv(os.path.join(path_data, "word_frequency_80.csv"),
                                 sep=";", encoding="utf-8", index=False)
    return word_frequency_matrix


def lemmatization(texts, allowed_postags):
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out


def sent_to_words(sentences: List[str]) -> List[str]:
    for sentence in sentences:
        yield gensim.utils.simple_preprocess(str(sentence), deacc=True)


def build_model(word_frequency_matrix: pd.DataFrame, nb_topics: int):
    words_per_topic: int = 20
    clefs: List[str] = list(word_frequency_matrix.columns)
    blocs: List[str] = list(word_frequency_matrix.index)
    print("Topic modeling")
    lda = LatentDirichletAllocation(n_components=nb_topics)
    topic_to_text = lda.fit_transform(word_frequency_matrix.values)
    pkl_filename = os.path.join(path_to_model, f"lda_model_blocs_{nb_topics}.pkl")
    with open(pkl_filename, 'wb') as file:
        pickle.dump(lda, file)
    topics: pd.DataFrame = pd.DataFrame({f"Topic{i}": [clefs[w] for w in top.argsort()[-words_per_topic:]]
                                         for i, top in enumerate(lda.components_)})
    table_topics_to_texts: pd.DataFrame = pd.DataFrame(np.vectorize(lambda z: f"{z:.3f}")(topic_to_text),
                                                       columns=range(nb_topics), index=blocs)
    topics.to_excel(os.path.join(path_data, f"topics_{nb_topics}.xlsx"), encoding="utf-8", index=False)
    table_topics_to_texts.to_excel(os.path.join(path_data, f"corpus_topics_{nb_topics}.xlsx"), encoding="utf-8",
                                   index=True)


def calculate_coherence(w2v_model, term_rankings):
    overall_coherence = 0.0
    for topic_index in range(len(term_rankings)):
        pair_scores = []
        for pair in combinations(term_rankings[topic_index], 2):
            try:
                pair_scores.append(w2v_model.wv.similarity(pair[0], pair[1]))
            except KeyError as e:
                print(e)
        topic_score = sum(pair_scores) / len(pair_scores)
        overall_coherence += topic_score
    return overall_coherence / len(term_rankings)


def get_descriptor(all_terms, H, topic_index, top):
    top_indices = np.argsort(H[topic_index, :])[::-1]
    top_terms = []
    for term_index in top_indices[0:top]:
        top_terms.append(all_terms[term_index])
    return top_terms


class TokenGenerator:
    def __init__(self, documents):
        self.documents = documents
        for i in range(len(documents)):
            self.documents[i] = self.documents[i].lower()
        self.tokenizer = re.compile(r"(?u)\b\w\w+\b")

    def __iter__(self):
        for doc in self.documents:
            tokens = []
            for tok in self.tokenizer.findall(doc):
                if len(tok) >= 2:
                    tokens.append(tok)
            yield tokens


def word2vec(raw_documents):
    docgen = TokenGenerator(raw_documents)
    w2v_model = gensim.models.Word2Vec(docgen, min_count=20, sg=1)
    print(len(w2v_model.wv.key_to_index))
    w2v_model.save("w2v-model.bin")


def load_model(word_frequency: pd.DataFrame, nb_topics):
    print(f"évaluation cohérence k={nb_topics}")
    clefs: List[str] = list(word_frequency.columns)
    blocs: List[str] = list(word_frequency.index)
    pkl_filename = os.path.join(path_to_model, f"lda_model_blocs_{nb_topics}.pkl")
    with open(pkl_filename, 'rb') as file:
        lda = pickle.load(file)
    words_per_topic: int = 20
    topic_to_text = lda.transform(word_frequency.values)
    topics: pd.DataFrame = pd.DataFrame({f"Topic{i}": [clefs[w] for w in top.argsort()[-words_per_topic:]]
                                         for i, top in enumerate(lda.components_)})
    table_topics_to_texts: pd.DataFrame = pd.DataFrame(np.vectorize(lambda z: f"{z:.3f}")(topic_to_text),
                                                       columns=range(nb_topics), index=blocs)
    term_rankings = [get_descriptor(clefs, lda.components_, topic_index, 10) for topic_index in range(nb_topics)]
    print(term_rankings)
    w2v_model = gensim.models.Word2Vec.load("w2v-model.bin")
    coherence = calculate_coherence(w2v_model, term_rankings)
    print(f"cohérence : {coherence}")
    return topic_to_text, topics, table_topics_to_texts


def get_parameter(adr: str):
    pkl_filename = os.path.join(path_to_model, adr)
    with open(pkl_filename, 'rb') as file:
        lda: LatentDirichletAllocation = pickle.load(file)
    print(lda.get_params())


if __name__ == "__main__":
    # df = table_ocr_1880()
    # df = build_corpus(df)
    # df = pd.read_csv(os.path.join(path_data, "textes_1880s.csv"))
    # word_frequency = count_vectorizer(df, 6000)
    df_textes: pd.Series = pd.read_csv(os.path.join(path_data, "textes_1880s.csv")).loc[:, "text"]
    df_freq: pd.DataFrame = pd.read_csv(os.path.join(path_data, "word_frequency_80.csv"), sep=";", encoding="utf-8",
                                        index_col=0)
    # for n_topics in list(range(3, 10, 1)) + list(range(10, 60, 5)):
    #     build_model(df_freq, n_topics)
    word2vec(df_textes.to_list())
    for n_topics in list(range(3, 10, 1)) + list(range(10, 60, 5)):
        text_topics, topics, table_text_topics = load_model(df_freq, n_topics)
