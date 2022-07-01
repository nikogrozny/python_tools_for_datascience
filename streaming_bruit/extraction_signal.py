import os
from statistics import mean
from typing import TextIO, List, Dict, Tuple
import re

import pandas as pd
from matplotlib import pyplot as plt

path_source = "write_here_path_to_streams"
path_data = os.getcwd()


def extraire_commentaires() -> Dict[str, pd.DataFrame]:
    coms_with_time: Dict[str, pd.DataFrame] = dict()
    for stream_file_adr in os.listdir(os.path.join(path_source, "streams")):
        print(stream_file_adr)
        if stream_file_adr[-4:] == "json":
            stream_file: TextIO = open(os.path.join(path_source, "streams", stream_file_adr), encoding="utf-8")
            stream_text = stream_file.read().replace("[", "µ").replace("]", "§")
            stream_file.close()
            infos_dates: List[str] = re.findall("\"content_offset_seconds\":[0-9.]+", stream_text)
            timings: List[float] = [float(z.split(":")[1]) for z in infos_dates]
            infos_content: List[str] = re.findall("\"fragments\":µ[^§]+§", stream_text)
            text_fields: List[List[str]] = [re.findall("\"text\":\"[^\"]+\"", frag) for frag in infos_content]
            text_only: List[str] = [" ".join([piece.split(":")[1][1:-1] for piece in frag]) for frag in text_fields]
            print(len(timings) - len(text_only))
            timings = timings[:len(text_only)]
            coms_with_time[stream_file_adr[:-5]] = pd.DataFrame({"timing": timings, "texte": text_only})
            coms_with_time[stream_file_adr[:-5]].to_csv(os.path.join(path_data, "dataframes_comments",
                                                                     f"{stream_file_adr[1:9]}.csv"), sep=";",
                                                        encoding="utf-8", index=False)
    return coms_with_time


def detect_peaks() -> None:
    time_steps: int = 120
    for comments_file_adr in os.listdir(os.path.join(path_data, "dataframes_comments")):
        print(comments_file_adr)
        comments_df: pd.DataFrame = pd.read_csv(os.path.join(path_data, "dataframes_comments", comments_file_adr),
                                                sep=";", encoding="utf-8")
        timings = comments_df.timing.to_list()
        frequence_coms: List[int] = [0] * int(max(timings) / time_steps + 1)
        for i, timez in enumerate(timings):
            frequence_coms[int(timez / time_steps)] += 1

        norm: int = mean(frequence_coms)
        is_peak: List[bool] = [False] + [frequence_coms[p-1]+frequence_coms[p]+frequence_coms[p+1] > 5*norm
                               for p in range(1, len(frequence_coms)-1)] + [False]
        peaks_df: pd.DataFrame = comments_df.loc[comments_df.timing.apply(lambda t: is_peak[int(t/time_steps)]), :]
        peaks_df.to_csv(os.path.join(path_data, "peaks", comments_file_adr), sep=";",
                           encoding="utf-8", index=False)

        plt.figure(figsize=(15, 10))
        plt.bar(range(len(frequence_coms)), frequence_coms, color=["blue" if z else "orange" for z in is_peak])
        plt.xlabel(f"Temps (intervalles de {time_steps}s.)")
        plt.ylabel("Nombre de commentaires")
        plt.savefig(os.path.join(path_data, "graphes", f"coms_{comments_file_adr[:10]}.png"))
        plt.close()


if __name__ == "__main__":
    all_comments: Dict[str, pd.DataFrame] = extraire_commentaires()
    detect_peaks()
