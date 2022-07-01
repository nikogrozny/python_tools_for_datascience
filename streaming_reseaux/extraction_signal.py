import os
from typing import List
import pandas as pd

path_data = os.path.join(os.getcwd(), "data")
path_exports = os.path.join(os.getcwd(), "exports")
list_data_dir: List[str] = ["pixel_war", "corpus_lausanne"]


def extraire_commentaires() -> None:
    for data_dir in list_data_dir:
        for stream_file_adr in os.listdir(os.path.join(path_data, data_dir)):
            if stream_file_adr[-3:] == "txt":
                auteur: str = stream_file_adr.split(" - ")[0]
                titre: str = " - ".join(stream_file_adr.split(" - ")[1:])[:-11]
                print(titre)
                date: str = auteur.split("]")[0][1:]
                auteur = auteur.split("]")[1]
                try:
                    comments: pd.DataFrame = pd.read_csv(os.path.join(path_data, data_dir, stream_file_adr),
                                                         encoding="utf-8", sep="UTC]", engine="python", header=None)
                    comments.loc[:, "date"] = comments.loc[:, 0].apply(lambda z: z.split()[0][1:].strip())
                    comments.loc[:, "time"] = comments.loc[:, 0].apply(lambda z: z.split()[1].strip())
                    comments.drop(0, axis=1, inplace=True)
                    comments.loc[:, "author"] = comments.loc[:, 1].apply(lambda z: z.split(":")[0].strip())
                    comments.loc[:, "message"] = comments.loc[:, 1].apply(lambda z: ":".join(z.split(":")[1:]).strip())
                    comments.drop(1, axis=1, inplace=True)
                    comments.loc[:, "streamer"] = auteur.strip()
                    print(comments.head(2))
                    comments.to_csv(
                        os.path.join(path_data, "dataframe_comments", f"{data_dir}__{auteur}__{date}__{titre}.csv"),
                        sep=";", encoding="utf-8", index=False)
                except pd.errors.EmptyDataError:
                    pass


def texte_par_auteur():
    total: int = len(os.listdir(os.path.join(path_data, "dataframe_comments")))
    for data_dir in list_data_dir:
        auteurs = sorted(list(set([adr.split("__")[1].strip()
                                   for adr in os.listdir(os.path.join(path_data, "dataframe_comments")) if
                                   data_dir in adr])))
        for auteur in auteurs:
            print(auteur)
            subset: List[str] = [adr for adr in os.listdir(os.path.join(path_data, "dataframe_comments"))
                                 if data_dir in adr and adr.split("__")[1].strip() == auteur]
            dataframe: pd.DataFrame = pd.concat([pd.read_csv(os.path.join(path_data, "dataframe_comments", adr),
                                                            sep=";", encoding="utf-8") for adr in subset], axis=0)
            dataframe = dataframe.loc[:, ["date", "message"]].dropna()
            dataframe = dataframe.groupby(['date'])['message'].apply(lambda x: ' '.join(x)).reset_index()
            dataframe.loc[:, "streamer"] = auteur
            dataframe.loc[:, "event"] = data_dir
            dataframe.to_csv(os.path.join(path_data, "par_auteur", f"{data_dir}_{auteur}.csv"), sep=";",
                             encoding="utf-8", index=False)
            total -= len(subset)
    print(total)


def followers_par_auteur():
    total: int = len(os.listdir(os.path.join(path_data, "dataframe_comments")))
    for data_dir in list_data_dir:
        auteurs = sorted(list(set([adr.split("__")[1].strip()
                                   for adr in os.listdir(os.path.join(path_data, "dataframe_comments")) if
                                   data_dir in adr])))
        for auteur in auteurs:
            print(auteur)
            subset: List[str] = [adr for adr in os.listdir(os.path.join(path_data, "dataframe_comments"))
                                 if data_dir in adr and adr.split("__")[1].strip() == auteur]
            dataframe: pd.DataFrame = pd.concat([pd.read_csv(os.path.join(path_data, "dataframe_comments", adr),
                                                            sep=";", encoding="utf-8") for adr in subset], axis=0)
            dataframe = dataframe.loc[:, ["date", "author"]].dropna()
            dataframe = dataframe.groupby(['date'])['author'].apply(lambda x: ' '.join(x)).reset_index()
            dataframe.loc[:, "author"] = dataframe.loc[:, "author"].apply(
                lambda z: " ".join(sorted(list(set(z.split())))))
            dataframe.loc[:, "streamer"] = auteur
            dataframe.loc[:, "event"] = data_dir
            dataframe.to_csv(os.path.join(path_data, "par_auteur_vw", f"{data_dir}_{auteur}.csv"), sep=";",
                             encoding="utf-8", index=False)
            total -= len(subset)
    print(total)


if __name__ == "__main__":
    # extraire_commentaires()
    # texte_par_auteur()
    followers_par_auteur()
