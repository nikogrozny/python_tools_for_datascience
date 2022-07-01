from topic_models import topic_models, identify_topics
from classification_topics import topics_games_graph, topics_cha, topics_proxy
from temporalite_classes_topics import graphe_topics, topics_with_time
from topic_time_graphs import graphe_contraint, dieg_time
from som_acp import do_acp, so_map

dest = "topics_par_periode"

if __name__ == "__main__":
    topic_models("textes_jv_blocks", dest)
    identify_topics(dest)
    tgg, tgw = topics_games_graph(dest)
    topics_proxy(dest)
    macha = topics_cha(dest)
    graphe_topics(dest)
    topics_with_time(macha, tgg, tgw, dest)
    dieg_time("textes_jv_clean")
    graphe_contraint(dest)
    so_map(dest)
    do_acp()