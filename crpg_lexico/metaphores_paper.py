import os
from typing import List, TextIO, Set, Tuple
from matplotlib import pyplot as plt
from matplotlib import rc
import pandas as pd
import re

path_data = os.getcwd()
dic_eng = open(os.path.join(path_data, "metaphores", "french_words.txt"), "r", encoding="utf-8")
french_words = set(dic_eng.read().split("\n"))
dic_eng.close()
st_fr = open(os.path.join(path_data, "metaphores", "french_stopwords.txt"), "r", encoding="utf-8")
stopwords = st_fr.read().split("\n")
st_fr.close()
stopwords = set([w.strip() for w in stopwords])
export_rep = "exports_final_article"
shortnames = ["Witcher 3", "Witcher 2", "Wastelands 2", "Ultima 9", "Ultima 8", "Ultima 7",
              "Tides of Num.", "Planescape", "TES Skyrim", "TES Oblivion", "TES Morrowind",
              "Pillars", "Fallaout NV", "Fallout 4", "Fallout 3", "Fallout 2", "Fallout 1",
              "Divinity", "Dk Dungeon", "CC Skyrim", "CC Fallout", "Baldur 1", "Baldur 2"]

candidats = [["alignement"],
             ["classe"],
             ["dommages", "dégâts"],
             ["points de vie", "pv", "vitalité", "santé"],
             ["aventure", "chapitre", "quête", "tour", "campagne", "progression", "round"]]
diegetique = [["balayage", "empreintes", "satellite", "planètes", "maillage", "harmonieux", "plate forme", "biais",
               "besoin", "phil", "supposée", "prévu", "capsule", "technologie", "accepterai", "combien", "inutile",
               "administrons", "prochain", "ultime", "sciences", "cognitif", "dérangeant", "réputations", "épaule",
               "étoile", "correct", "ombre", "maisons", "plan", "plans", "crocs", "flux", "transiteur", "ployé",
               "oom", "cosmique", "astronomique", "céleste", "voir", "transmuteur", "équilibre", "forces", "kilomètres",
               "rochers", "éviers", "recevrez", "capsules", "donnera", "prendre", "santé", "pensées", "heure",
               "corps", "voulez", "donnera", "indices", "objet", "dois", "animaux", "élégance", "frégate", "populaires",
               "ressent", "retrouver", "venez", "parlé", "admets", "voyager", "cadeau", "frissons", "dispersion",
               "procédure", "robots", "épitaphe", "hein", "ça", "camarades", "jeune", "shows", "données", "reste",
               "relation", "basses", "aurait", "distinguée", "dessert", "veux", "camarades", "intimidée",
               "bureaucratique", "titan", "factotum", "structure", "inégalité", "système", "réunions", "dangers",
               "séparation", "mystérieux", "première", "civile", "tenue", "mec", "bouquin", "sandwich", "certaine",
               "chié", "angeles", "recommande", "az", "mourez", "pompon", "kikimorrhes", "organisées", "temeria",
               "finie", "hommes", "allons", "vouloir", "souillures", "murs", "couteau", "enseigné", "dos", "remarqué",
               "procéder"],
              ["inférieure", "dirigeante", "établissement", "pense", "supérieures", "suppose", "distinction",
               "air", "sociale", "jour", "inégalitaire", "quelle", "première", "rang", "premier", "devrait",
               "prie", "seconde", "cartes", "évènements", "sais", "toi", "tu", "avoir", "boxe", "chica", "sup",
               "deuxième", "vraiment", "marin", "amiral", "réacteur", "phylum", "troisième", "subphylum", "salle",
               "rapport", "père", "accord", "très", "vivent", "mensonge", "aller", "chemin", "copains",
               "matins", "occupe", "ville", "viens", "avez", "aurez", "déborde", "ouvrière", 'trop', "défaut", "salle",
               "propre", "condition", "noble", "défavorisées", "ressemblance", "incinérations", "terminal", "virus",
               "sophistiquée", "histoire", "avenir", "jette", "typique", "chérie", "dope", "abri", "faire", "heures",
               "perturbez", "dérangez", "minimum", "parler", "rail", "carrington", "toujours", "ici", "sythétiques",
               "robot", "enfants", "commonwealth", "couleur", "professeur", "catégorie", "espionage", "alerte",
               "superviseur", "dîner", "vachement", "rendu", "gênez", "manques", "watson", "poussin", "sentir",
               "monde", "lycée", "bouffons", "marchande", "sociales", "moyenne", "basse", "moyennes", "dirigeantes",
               "assiste", "académie", "taxonomie", "tyrannie", "aboli", "pote", "cowboys", "sentirez", "carrément",
               "plutôt", "divertissement", "bronx", "ophiri", "erreur", "inférieures", "location", "représentants",
               "personne", "maintenant", "look", "chose", "tête", "traîner", "gustavo", "maquillage", "freddie",
               "apôtre", "commençons", "excursion", "escorter", "racaille", "jalousie", "camarade", "jolie", "mme",
               "sortir", "soldat", "entrepôt", "décharge", "sortir", "incarnée", "folle", "parler", "voler", "pendant",
               "lourd", "crever", "coursier", "panne", "incorrecte", "synthétiques", "jouer", "strip", "tox", "affaire",
               "raffinement", "haute", "époque", "dois", "animaux", "élégance", "frégate", "populaires",
               "ressent", "retrouver", "venez", "parlé", "admets", "voyager", "cadeau", "frissons", "dispersion",
               "procédure", "robots", "épitaphe", "hein", "ça", "camarades", "jeune", "shows", "données", "reste",
               "relation", "basses", "aurait", "distinguée", "dessert", "veux", "camarades", "intimidée",
               "bureaucratique", "titan", "factotum", "structure", "inégalité", "système", "réunions", "dangers",
               "séparation", "mystérieux", "première", "civile", "tenue", "mec", "bouquin", "sandwich", "certaine",
               "chié", "angeles", "recommande", "az", "mourez", "pompon", "kikimorrhes", "organisées", "temeria",
               "finie", "hommes"],
              ["collatéraux", "cérébraux", "assurer", "frais", "provoquer", "histoire", "évaluer", "réparer",
               "utilisateur", "irrémédiables", "irréparables", "catastrophiques", "matériels", "réparant", "aujourd",
               "changer", "bobine", "sacrés", "inimaginables", "structure", "prime", "fiasco", "date", "intentions",
               "essuyer", "sectes", "conséquents", "subi", "évaluation", "estimation", "causé", "aimerait", "soldés",
               "repoussée", "significatifs", "sévères", "repoussés", "annonce", "étendue", "ferait", "ampleur",
               "provoquaient", "conséquents", "peux", "seraient", "irréversibles", "paierai", "boucher", "penche",
               "rendre", "compensation", "récents"],
              ["jeunesse", "joie", "ardeur", "sang", "traverse", "âme", "peu", "promenade", "verset", "excellente",
               "campement", "maintenant", "enquérir", "garantir", "dieux", "saine", "permettez", "vaudrait",
               "gens", "bonne", "promenade", "toast", "restez", "verre", "réagiriez", 'trinquer', "pleine", "boire",
               "améliorer", "sacrifice", "excellente", "bord", "travail", "veux", "souciez", "problèmes", "parfaite",
               "voilà", "buvons", "semble", "chope", "améliorée", "plein", "mauvaise", "aptes", "spécimen",
               "examen", "cardiaque", "esclave", "esclaves", "goût", "ville", "résidents", "don", "connais", "plats",
               "doc", "pleine", "veillez", "assurer", "propres", "pauvreté", "hygiène", "berceau", "marcher", "partir",
               "préoccupiez", "bilan", "dépenses", "radicaux", "publique", "problème", "vieux", "comportement",
               "couverture", "risque", "stress", "inquiéter", "alors", "refaire", "chien", "problèmes", "bonheur",
               "heureux", "dangereux", "garder", "intervalles", "péril", "inquiet", "mauvaise", "parfaite", 'travail',
               'recouvrer', 'pleine', "bonne", "joie", "maintien", "boirai", "déteriore", "aurait", "saper", "raison",
               "bière", "affaires", "ancêtres", "veiller", "chaleur", "abandonne", "inquiète", "regarde",
               "préjudiciable",
               "parie", "ligue", "priorité", "bouffer", "image", "filles", "pissenlits", "productivité", "chimiques",
               "promenades", "crache", "assurée", "tenez", "mauvais", "nocifs", "discute", "parler", "fragile",
               "imagine",
               "terres", "montagnes", "trinquera", "consacré", "fragile", "raisons", "coïncidence", "choses", "maison",
               "enquiert", "médiocre", "meilleure", "merci", "mercy", "britannia", "boisson", "arbres", "inquiets"
                                                                                                        "voeux",
               "gratitude", "politiques", "père", "pied", "buvez", "boit", "boiront", "messieurs", "chasse",
               "animal", "souverain", "tournée", "souhaite", "paix", "bouteille", "vignes", "chantant", "trinqueront",
               "financière", "roi", "douter", "fer", "disons", "hydromel", "nuire", "ahhhh", "noble", 'trinquons',
               "graves", "forcément", "bilans", "besoin", "vint"],
              ["rappelle", "sujets", "rapporte", "méfier", "autel", "sacré", "verset", "écrire", "loi", "inepties",
               "existence", "estomac", "simple", "voix", "vengeance", "américaine", "risques", "travaille", "rédige",
               'sortent', 'promenade', 'lancé', 'sacrée', 'gloire',
               'marché', 'goût', 'partons', 'siècles', 'risquée', 'impressionnantes',
               'rêvait', 'honneurs', 'congé', 'devons', 'côtés', 'années', 'compagnon', 'nouvelles',
               'cours', 'fortune', 'épopées', 'partir', 'occupe', 'aimez', 'confort', 'démangé',
               'amies', 'mâle', 'aujourd', 'durant', 'opportune', 'stupide', 'cafe', 'réalité',
               'toutes', 'cachés', 'vivre', 'amitié', 'nouvelle', 'aimeriez', 'parles', 'nautiques',
               'partait', 'pillards', 'danger', 'mâles', 'film', 'tuées', 'passé', 'merveilles',
               'collègue', 'rêver', 'maître', 'écoutez', 'occupée', 'vendeur', 'lançons', 'silver',
               'hasardeuse', 'voici', 'espérais', 'lanceraient', 'nuka', 'safari', 'attend', 'accompagne',
               'épreuves', 'malheureuse', 'prochaines', 'valait', 'veniez', 'soldée', 'trésors', 'connaissances',
               'fonctionnaires', 'suggéré', 'navire', 'glorieuses', 'combats', 'ami', 'défi', 'planifier',
               'raconter', 'archipel', 'ouvrages', 'épique', 'espérait', 'nombreuses', 'lot', 'voulais',
               'nouvelle', 'chants', 'rêve', 'compagnons', 'connu', 'désert', 'relatant', 'dérobée',
               'souviens', 'balafré', 'raconter', 'vivre', 'intéressante', 'colin', 'soif', 'conduiront',
               'aventureux', 'regarde', 'sourit', 'héros', 'mois', 'appelé', 'enfin', 'hubert',
               'partier', 'caverne', 'jour', 'respertoriés', 'glanés', 'nage', 'amis', 'finale',
               'vivre', 'parler', 'inspirent', 'meilleures', 'raison', 'prit', 'âges', 'curieuse',
               'passionnés', 'attendent', 'tombés', 'gloire', 'raconte', 'albinos', 'héroïques', 'effarantes',
               'antre', 'affût', 'nombreuses', 'triomphe', 'armurier', 'vécu', 'ésotérique', 'clos',
               'frères', 'artillerie', 'dire', 'confrérie', 'mémoires', 'âme', 'préparations', 'responsable',
               'oeuvre', 'anvil', 'page', 'livre', 'sanglant', 'personnelle', 'étendrai', 'sanglant',
               'régime', 'aborderai', 'nature', 'fêtes', 'serais', 'devons', 'seconder', 'longtemps',
               'exposez', 'continuons', 'poursuivons', 'honorable', 'cagoulés', 'arrête', 'souhaite', 'vouerai',
               'habiles', 'réfléchissez', 'quincy', 'netchs', 'obstinée', 'sillonner', 'devrait', 'dallis',
               'divinité', 'raconte', 'éternelle', 'ingrédients', 'délices', 'poursuivit', 'injustice', 'accomplissez',
               'justifie', 'bravons', 'connotation', 'accumulation', 'progresses', 'anciens', 'abouti', 'déjà',
               'données', 'ruines', 'connaissances', 'amenée', 'marqua', 'souffrance', 'culture', 'quelque',
               'vétérans', 'brûlées', 'semblables', 'saintes', 'satisfaite', 'expliquer', 'espoir', 'accomplissez',
               'lancés', 'fatale', 'quêteur', 'culte', 'pathétique', 'nérévarine', 'tamriel', 'gloire',
               'proies', 'formuler', 'wabbajack', 'croisé', 'aurait', 'casque', 'bénissent', 'stupide',
               'commandeur', 'messire', 'fantômes', 'semblables', 'boueuses', 'histoires', 'tiber', 'empereur',
               'interdites', 'services', 'cesses', 'plans', 'renseignements', 'justice', 'rêve', 'érudits',
               'sacrifices', 'poursuis', 'aider', 'accomplir', 'excitante', 'caravansérail', 'prouver', 'poursuivons',
               'désireux', 'effrénée', 'désabusé', 'flux', 'minax', 'disparaissent', 'afentures', 'héroik',
               'bains', 'utilité', 'incline', 'lord', 'prophète', 'accélérer', 'consumé', 'vertus',
               'importantes', 'vent', 'noble', 'grotte', 'testeur', 'bannière', 'pressé', 'ingrédients',
               'organisations', 'égouts', 'cheval', 'savoir', 'déshonorante', 'stupide', 'martyr', 'ancestral',
               'ciri', 'stock', 'marigots', 'yennefer', 'protais', 'gloire', 'trahir', 'détour',
               'athkala', 'cossue', 'abandonnée', 'tit', 'submergées', 'père ', 'gong', 'jumelles',
               'duc', 'dit', 'nord', 'continuez', 'bâtards', 'ouvrirai', 'infecté', 'prenez',
               'sceau', 'procéder', 'espece', 'compris', 'faux', 'pourrais', 'dominaient', 'menacées',
               'quartiers', 'cloches', 'séjour', 'récompense', 'durlag', 'prendre', 'signaux', 'relais',
               'origine', 'mystère', 'occupants', 'mystérieuse', 'demi', 'calamiteux', 'orgueil', 'demi',
               'luth', 'venu', 'réussi', 'haute', 'plaisanterie', 'maître', 'rictus', 'siège',
               'chatouille', 'échoué', 'marioles', 'jardin', "reins", "manège", "sac", "juste", "torr",
               "cave", "ville", "suicida", "miria", "joli", "touristes", "démarre", "épidémie",
               "joue", "écart", "flambeau", "croyant", "prydwen", "père", "connais", "désolées",
               "embuscade", "fusion", 'reprenne', 'retourne', 'présenter', 'gratuit', 'illuminée',
               "maintenant", "débile", "garde", "service", "détache", "pièces", "plaque", "enfantin",
               "obéira", "dégénéré", "osrya", "lierre", "périra", "patrouiller", "camp", "guet", "roue",
               'soigner', "boutique", "bloqués", "ouest", "rejoindre", "repassant", "suffit", 'ouest',
               "cloche", "petit", "fait", "sol", "résidents", "sale", "sales", "jongler", "boulot", "egret",
               "opérateurs", "gomorrah", "immobiles", "fois", "notre", "jouez", "gauche", "abattant", "chargeaient",
               "figure", "livrées", "retourné", "chant", "retour", "dominés", "dresse", "contentez", "viendra",
               "coin", "revenu", "telvanni", "branora", "hein", "nôtres", "débarrasser", "merci", "écarlate",
               "tel", "telvannis", "devrai", "soie", "bientôt", "tournant", "yeux", "muraille", "dorée", "château",
               "effondrées", "détruite", "pied", "faire", "vestibule", "traverser", "vint", "sud", "cristal",
               "palais", "pendu", "couloirs", "endroits", "casier", "donnez", "âge", "attaquèrent", "descendit",
               "rosée", "lorkhan", "cellule", "enneigée", "donjon", "blanc", "archer", "vivant", "main", "hauteur",
               "voulez", "joué", "abri", "destinée", "volikhar", "morvayn", "ruine", "censé", "remède", "hériterai",
               "loin", "contorsion", "ancree", "legions", "plan", "entropie", "maison", "quelques", "platine",
               "mords", "racines", "placés", "souhaitez", "simplet", "éblouissent", "clignote", "vole", "évapore",
               "défenses", "verrous", "sommet", "intérieur", "pylones", "ombres", "éternel", "murs", "porte",
               "dragon", "goule", "radio", "rivière", "panneau", "récupéré", "towers", "remplies", "confiance",
               "cuir", "disque", "highpool", "draves", "repliez", "esquivant", "sorceleur", "éliminée", "vallée",
               "vieille", "tarots", "rond", "enfermé", "géant", "depuis", "maudite", "transporter", "volé", "isolée",
               "gentil", "ensevelir", "fallait", "intrus", "enchères", "truffée", "investie", "apparue", "bâtie",
               "elfe", "vides", "dépouille", "capitaine", "confinée", "fonctionneront", "festival", "voyageurs",
               "anchorage", "soldat", "relations", "président", "représailles", "dénigrement", "clair", "chique",
               "croisée", "proximité", "considération", "vallonnée", "militaires", "bonté", "empire", "gens", "mines",
               "renifle", "venue", "électorale", "tenebrae", "soutenir", "colonisation", "moines", "paysans",
               "faisait", "hôpital", "reine", "inquiète", "dragon", "quantum", "gêner", "christine", "ralentit",
               "ralentit", "boue", "reprenaient", "entravaient", "entravée", "lente", "estompent", "escadrons",
               "vannes"]
              ]
# [],
# []]
metaphores = [["flux"],
              ["jumelée", "progresser", "compétences", "expérience", "jumelées", "jumelant", "clerc", "druide",
               "voleur", "personnage", "personnages", "parametres"],
              [],
              ["dégât", "dégâts", "barre", "régénérer", "fortifié", "régénérez", "degré", "points"],
              ["active", "échec", "durée", "portée", "sauvegarde", "sauvegardée"]]


def bag_of_words() -> None:
    print("Extraction des textes")
    for adr in os.listdir(os.path.join(path_data, "textes_jv")):
        print(adr)
        f = open(os.path.join(path_data, "textes_jv", adr), "r", encoding="utf-8")
        original_game_content: str = f.read()
        f.close()
        blocks: List[str] = re.findall("<TRADUIT>[^<]+</TRADUIT>", original_game_content)
        words_all: List[str] = blocks.copy()
        words_fr: List[str] = blocks.copy()
        for i, block in enumerate(blocks):
            bag_of_all_words = block.replace("<TRADUIT>", "").replace("</TRADUIT>", "")
            words: List[str] = re.findall(r"[a-zéèëêààäâîïôöùûüç]+", bag_of_all_words.lower())
            words_all[i] = " ".join(words)
            words_fr[i] = " ".join([w for w in words if w in french_words])
        bag_of_french_words = "\n".join(words_fr)
        bag_of_all_words = "\n".join(words_all)

        f = open(os.path.join(path_data, "textes_fr", adr), "w", encoding="utf-8")
        f.write(bag_of_all_words)
        f.close()
        f = open(os.path.join(path_data, "textes_fr_bags", adr), "w", encoding="utf-8")
        f.write(bag_of_french_words)
        f.close()
    print("Extraction terminée")


def around(sentence: str, word: str, radius: int) -> List[str]:
    words: List[str] = sentence.split()
    neighborhood: List[str] = list()
    for i in range(len(words)):
        if re.search(f"{word}s?", words[i]):
            neighborhood.append(" ".join(words[max(0, i - radius):min(len(words), i + radius)]))
    return neighborhood


def contexte(mots_cibles: List[List[str]], indices_diegese: List[List[str]], certitude_meta: List[List[str]]) -> None:
    print("Extraction du contexte")
    noms_jeux: List[str] = [j[:-4] for j in os.listdir(os.path.join(path_data, "textes_fr"))]
    comptage_true: pd.DataFrame = pd.DataFrame(0, index=noms_jeux, columns=['-'.join(li) for li in mots_cibles])
    comptage_detailed: pd.DataFrame = pd.DataFrame(0, index=noms_jeux, columns=[x for li in mots_cibles for x in li])
    comptage_fake: pd.DataFrame = pd.DataFrame(0, index=noms_jeux, columns=['-'.join(li) for li in mots_cibles])
    for i, liste_mots in enumerate(mots_cibles):
        nom_liste_mots = '-'.join(liste_mots)
        # quand les listes de mots sont longues on ne fait qu'un échantillon de 10% :
        jump = 10 if len(liste_mots) > 2 else 1
        file_all: TextIO = open(os.path.join(path_data, "metaphores", "contextes", f"{nom_liste_mots}.txt"),
                                "w", encoding="utf-8")
        file_discriminated: TextIO = open(os.path.join(path_data, "metaphores", "contextes",
                                                       f"discr_{nom_liste_mots}.txt"), "w", encoding="utf-8")
        for mot in liste_mots:
            print(f"\nExtraction des contextes - {mot}")
            for adr in os.listdir(os.path.join(path_data, "textes_fr")):
                file_all.write(f"*** {adr}\n\n")
                file_discriminated.write(f"*** {adr}\n\n")
                file_data: TextIO = open(os.path.join(path_data, "textes_fr", adr), "r", encoding="utf-8")
                n: int = 0
                for line in file_data:
                    if re.search(f"[^a-zéèëêààäâîïôöùûüç]{mot}s?[^a-zéèëêààäâîïôöùûüç]", line):
                        if n % jump == 0:
                            contexte_local: List[str] = around(line, mot, 5)
                            file_all.write(line)
                            file_all.write("\n")
                            if set(indices_diegese[i]) & set(" ".join(contexte_local).split(" ")) \
                                    and not (set(certitude_meta[i]) & set(" ".join(contexte_local).split(" "))):
                                comptage_fake.loc[adr[:-4], nom_liste_mots] += 1
                                file_discriminated.write("\n".join(contexte_local))
                            else:
                                comptage_true.loc[adr[:-4], nom_liste_mots] += 1
                                comptage_detailed.loc[adr[:-4], mot] += 1
                                file_discriminated.write("\n".join(contexte_local).upper())
                            file_discriminated.write("\n")
                        n += 1
                file_data.close()
        file_all.close()
        file_discriminated.close()

    print(comptage_true.head())
    print(comptage_fake.head())
    comptage_true.to_csv(os.path.join(path_data, "metaphores", "comptage_brut.csv"), encoding="utf-8", sep=";")
    comptage_detailed.to_csv(os.path.join(path_data, "metaphores", "comptage_detailed.csv"), encoding="utf-8", sep=";")
    comptage_fake.to_csv(os.path.join(path_data, "metaphores", "comptage_fake.csv"), encoding="utf-8", sep=";")
    print("Extraction terminée")


def frequences() -> None:
    colmap = ["blue", "lightsteelblue", "lime", "olive", "gold", "orange", "red", "fuchsia"]
    print("Calcul des fréquences")
    rc('font', **{'size': 16})
    comptage_brut: pd.DataFrame = pd.read_csv(os.path.join(path_data, "metaphores", "comptage_brut.csv"),
                                              encoding="utf-8", sep=";", index_col=0)
    tailles_fichiers_jeu = [os.stat(os.path.join(path_data, "textes_fr", f"{jeu}.xml")).st_size
                            for jeu in comptage_brut.index]
    comptage_brut.loc[:, "siz"] = pd.Series(tailles_fichiers_jeu, index=comptage_brut.index)
    comptage_brut = comptage_brut.apply(lambda z: z.div(z.siz).multiply(10000), axis=1)
    comptage_fake: pd.DataFrame = pd.read_csv(os.path.join(path_data, "metaphores", "comptage_fake.csv"),
                                              encoding="utf-8", sep=";", index_col=0)
    comptage_fake.loc[:, "siz"] = pd.Series(tailles_fichiers_jeu, index=comptage_fake.index)
    comptage_fake = comptage_fake.apply(lambda z: z.div(z.siz).multiply(10000), axis=1)
    comptage_detailed: pd.DataFrame = pd.read_csv(os.path.join(path_data, "metaphores", "comptage_detailed.csv"),
                                                  encoding="utf-8", sep=";", index_col=0)
    comptage_detailed.loc[:, "siz"] = pd.Series(tailles_fichiers_jeu, index=comptage_fake.index)
    comptage_detailed = comptage_detailed.apply(lambda z: z.div(z.siz).multiply(10000), axis=1)
    for i, mots_cibles in enumerate([x for x in comptage_brut.columns if x != "siz"]):
        plt.figure(figsize=(15, 10))
        for j, mot in enumerate(mots_cibles.split("-")):
            if j == 0:
                plt.barh([z[:-6] for z in comptage_detailed.index], comptage_detailed.loc[:, mot], color=colmap[j],
                        edgecolor="black", label=mot)
            else:
                plt.barh([z[:-6] for z in comptage_detailed.index], comptage_detailed.loc[:, mot],
                        color=colmap[j],
                        edgecolor="black",
                        left=[sum(comptage_detailed.loc[z, mots_cibles.split("-")[k]] for k in range(j)) for z in
                                comptage_detailed.index], label=mot)
        plt.barh([z[:-6] for z in comptage_brut.index], comptage_fake.loc[:, mots_cibles], color="white",
                left=comptage_brut.loc[:, mots_cibles], edgecolor="black",
                label="non métaphorique")
        plt.title(f"Fréquence d'utilisation de {mots_cibles}")
        plt.yticks([z[:-6] for z in comptage_detailed.index], shortnames[::-1])
        plt.legend()
        plt.savefig(os.path.join(path_data, "metaphores", export_rep, f"freq_{mots_cibles}.png"), dpi=300)

        plt.figure(figsize=(15, 10))
        for j, mot in enumerate(mots_cibles.split("-")):
            if j == 0:
                plt.barh([z[:-6] for z in comptage_detailed.index], comptage_detailed.loc[:, mot], color="0.4",
                        edgecolor="black", label=mot)
            else:
                plt.barh([z[:-6] for z in comptage_detailed.index], comptage_detailed.loc[:, mot],
                        color=f"{min(0.9, j / 10 + 0.4)}",
                        edgecolor="black",
                        left=[sum(comptage_detailed.loc[z, mots_cibles.split("-")[k]] for k in range(j)) for z in
                                comptage_detailed.index], label=mot)
        plt.title(f"Fréquence d'utilisation de {mots_cibles}")
        plt.yticks([z[:-6] for z in comptage_detailed.index], shortnames[::-1])
        plt.legend()
        plt.savefig(os.path.join(path_data, "metaphores", export_rep, f"freq_{mots_cibles}_metonly.png"), dpi=300)
    print("Calcul terminé")


def cooccurences(mots_cible: List[List[str]], indices_diegese: List[List[str]],
                 certitudes_meta: List[List[str]]) -> None:
    taille_contexte = 5
    tous_mots_cibles: List[str] = [x for li in mots_cible for x in li]
    table_cooccurences: pd.DataFrame = pd.DataFrame(0, index=tous_mots_cibles, columns=tous_mots_cibles)
    for i, liste_mots in enumerate(mots_cible):
        for mot_cible in liste_mots:
            print(f"\nExtraction des cooccurences - {mot_cible}")
            for adr in os.listdir(os.path.join(path_data, "textes_fr")):
                sourcefile: TextIO = open(os.path.join(path_data, "textes_fr", adr), "r", encoding="utf-8")
                for line in sourcefile:
                    if re.search(f"[^a-zéèëêààäâîïôöùûüç]{mot_cible}s?[^a-zéèëêààäâîïôöùûüç]", line):
                        contexte_local: List[str] = around(line, mot_cible, taille_contexte)
                        if set(indices_diegese[i]) & set(" ".join(contexte_local).split(" ")) \
                                and not (set(certitudes_meta[i]) & set(" ".join(contexte_local).split(" "))):
                            pass
                        else:
                            ctx_broad: Set[str] = set(" ".join(around(line, mot_cible, 10)).split(" ")) - stopwords
                            for mot_candidat in ctx_broad:
                                if mot_candidat not in table_cooccurences.index:
                                    table_cooccurences.loc[mot_candidat, :] = pd.Series(0,
                                                                                        index=tous_mots_cibles.copy())
                                table_cooccurences.loc[mot_candidat, mot_cible] += 1
                sourcefile.close()

    for mot_cible in tous_mots_cibles:
        table_coocs_locale: pd.Series = table_cooccurences.loc[:, mot_cible].sort_values(ascending=False)
        print(table_coocs_locale.head())
        table_coocs_locale.to_csv(os.path.join(path_data, "metaphores", f"coocs_{mot_cible}.csv"),
                                  encoding="utf-8", sep=";")


def graphe_coocs(cand: List[List[str]]) -> None:
    tous_mots_cibles: List[str] = [x for li in cand for x in li]
    graphe_file: TextIO = open(os.path.join(path_data, "metaphores", "coocs_graph.dot"), "w")
    extra_stopwords: List[str] = ["br", "gt", "lt", "end", "start"]
    edges: Set[Tuple[str, str]] = set()
    for mot_cible in tous_mots_cibles:
        main_cooccurences: List[str] = pd.read_csv(os.path.join(path_data, "metaphores", f"coocs_{mot_cible}.csv"),
                                              encoding="utf-8", sep=";", header=None).iloc[1:15, 0].tolist()
        for cooccurence in main_cooccurences:
            edges.add((mot_cible, cooccurence))
    all_coocurents: List[str] = [e[1] for e in edges]
    connecteurs: Set[str] = {c for c in all_coocurents if all_coocurents.count(c) > 1 and c not in extra_stopwords}
    useful_edges: List[Tuple[str, str]] = [e for e in edges if e[1] in connecteurs]
    exclu_connecteurs: List[str] = list(connecteurs - set(tous_mots_cibles))
    print(useful_edges)
    graphe_file.write("graph coocs{")
    for i, mot_cible in enumerate(tous_mots_cibles):
        graphe_file.write(f'\n m{i} [label="{mot_cible}" shape=box];')
    for i, con in enumerate(exclu_connecteurs):
        graphe_file.write(f'\n c{i} [label="{con}"];')
    for ue in useful_edges:
        if ue[1] in tous_mots_cibles:
            graphe_file.write(f'\n m{tous_mots_cibles.index(ue[0])} -- m{tous_mots_cibles.index(ue[1])};')
        else:
            graphe_file.write(f'\n m{tous_mots_cibles.index(ue[0])} -- c{exclu_connecteurs.index(ue[1])} [style=dotted];')
    graphe_file.write("\n}")


if __name__ == "__main__":
    # bag_of_words()
    # contexte(candidats, diegetique, metaphores)
    frequences()
    # cooccurences(candidats, diegetique, metaphores)
    # graphe_coocs(candidats)
