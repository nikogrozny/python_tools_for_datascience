from re import sub
from time import time
import numpy as np
import pandas as pd
from unidecode import unidecode


def clean_commune(df: pd.DataFrame, f: str) -> pd.DataFrame:
    t0 = time()
    print("Nettoyage nom ville")
    df.loc[~df[f].isna(), f] = df.loc[~df[f].isna(), f].str.lower().apply(unidecode).str.strip().str.replace("-",
                                                                                                             " ").str.replace(
        "'", " ")
    df.loc[~df[f].isna(), f] = df.loc[~df[f].isna(), f].apply(lambda x: sub(r"'", " ", x))
    df.loc[~df[f].isna(), f] = df.loc[~df[f].isna(), f].str.replace(r"^paris (.*)", r"paris").str.replace(
        r".*chesnay.*", r"le chesnay rocquencourt").str.replace(
        r".*herblay.*", "herblay sur seine").str.replace(r'(.*)kreml(.*)', r'le kremlin bicetre').str.replace(
        r".*levalloi.*", "levallois perret").str.replace(
        r".*aix en p.*", "aix en provence").str.replace(r".*aubervillier.*", "aubervilliers").str.replace(
        r".*montpe.*", "montpellier").str.replace(r".*cergy.*", "cergy").str.replace(r".*hellem.*",
                                                                                     "lille").str.replace(
        r".*clichy la g.*", "clichy").str.replace(
        r".*blanc mesnil.*", "le blanc mesnil").str.replace(r".*courcouro.*", "evry courcouronnes").str.replace(
        r"[0-9]", "").str.replace(r" st ", r" saint ").str.replace(
        r'^st (.*)', r'saint \1').str.replace(r'^ste (.*)', 'sainte ').str.replace(r' cedex', r'').str.replace(
        r"(.*) d (.*)", r"\1 d'\2").str.replace(
        r"(.*) l (.*)", r"\1 l'\2").str.replace(r' / ', r' sur ').str.replace(' /s', ' sur ').str.replace('/', ' sur ')
    df.loc[df[f] == 'saint maur', f] = 'saint maur des fosses'
    df.loc[df[f] == 'boulogne', f] = 'boulogne billancourt'
    df.loc[df[f] == 'boulogne billlancourt', f] = 'boulogne billancourt'
    df.loc[df[f] == 'champagne', f] = 'champagne au mont d or'
    df.loc[df[f] == 'bures', f] = 'bures sur yvette'
    df.loc[df[f] == 'saint germain', f] = 'saint germain en laye'
    df.loc[df[f] == 'asnieres', f] = 'asnieres sur seine'
    df.loc[df[f] == 'forges', f] = 'forges les bains'
    df.loc[df[f] == 'bretigny', f] = 'bretigny sur orge'
    df.loc[df[f] == 'villefranche', f] = 'villefranche sur saone'
    df.loc[df[f] == 'boisse', f] = 'la boisse'
    df.loc[df[f] == 'bellegarde', f] = 'bellegarde sur valserine'
    df.loc[df[f] == 'cournon', f] = 'cournon d auvergne'
    df.loc[df[f] == 'le blanc', f] = 'le blanc mesnil'
    df.loc[df[f] == 'corbeil', f] = 'corbeil essonnes'
    df.loc[df[f] == 'lagny', f] = 'lagny sur marne'
    df.loc[df[f] == "eragny sur oise", f] = 'eragny'
    df.loc[df[f] == "franconville la garenne", f] = 'franconville'
    df.loc[df[f] == "montfort l'amaury", f] = 'montfort l amaury'
    df.loc[df[f] == "saint germains les corbeils", f] = 'saint germain les corbeil'
    df.loc[df[f] == "pavillons sous bois", f] = 'les pavillons sous bois'
    df.loc[df[f] == "ballancourt", f] = 'ballancourt sur essonne'
    df.loc[df[f] == "saint thibault des vignens", f] = 'saint thibault des vignes'
    df.loc[df[f] == "conflans saint honorine", f] = 'conflans sainte honorine'
    df.loc[df[f] == "conflans ste honorine", f] = 'conflans sainte honorine'
    df.loc[df[f] == "vert sur marne", f] = 'vaires sur marne'
    df.loc[df[f] == "saint mandee", f] = 'saint mande'
    df.loc[df[f] == "monfermeil", f] = 'montfermeil'
    df.loc[df[f] == "saint maclou de follevile", f] = 'saint maclou de folleville'
    df.loc[df[f] == "maison alfort", f] = 'maisons alfort'
    df.loc[df[f] == "saint genevieves des bois", f] = 'sainte genevieve des bois'
    df.loc[df[f] == "sainte genenvieve des bois", f] = 'sainte genevieve des bois'
    df.loc[df[f] == "boulogne jean jaures", f] = 'boulogne billancourt'
    df.loc[df[f] == "la ferte st. aubin", f] = 'la ferte saint aubin'
    df.loc[df[f] == "saint anne", f] = 'sainte anne'
    df.loc[df[f] == "enghien", f] = 'enghien les bains'
    df.loc[df[f] == "asniere sur seine", f] = 'asnieres sur seine'
    df.loc[df[f] == "montreuil sous bois", f] = 'montreuil'
    df.loc[df[f] == "le perreux", f] = 'le perreux sur marne'
    df.loc[df[f] == "alforville", f] = 'alfortville'
    df.loc[df[f] == "levallois", f] = 'levallois perret'
    df.loc[df[f] == "saint raphaël", f] = 'saint raphael'
    df.loc[df[f] == "deui la barre", f] = 'deuil la barre'
    df.loc[df[f] == "quatres bornes", f] = 'quatre bornes'
    df.loc[df[f] == "l hays les roses", f] = 'l hay les roses'
    df.loc[df[f] == "dammartin en goële", f] = 'dammartin en goele'
    df.loc[df[f] == "montereau fault sur yonne", f] = 'montereau fault yonne'
    df.loc[df[f] == "perigny sur yerres", f] = 'perigny'
    df.loc[df[f] == "le chamblac", f] = 'chamblac'
    df.loc[df[f] == "cheveuse", f] = 'chevreuse'
    df.loc[df[f] == "la varenne saint hilaire", f] = 'saint maur des fosses'
    df.loc[df[f] == "velizy", f] = 'velizy villacoublay'
    df.loc[df[f] == "puisuex en france", f] = 'puiseux en france'
    df.loc[df[f] == "fresnes cauverville", f] = 'fresne cauverville'
    df.loc[df[f] == "loognes", f] = 'lognes'
    df.loc[df[f] == "thoury ferrottes", f] = 'thoury ferottes'
    df.loc[df[f] == "saint jusaint saint rambert", f] = 'saint just saint rambert'
    df.loc[df[f] == "caluire", f] = 'caluire et cuire'
    df.loc[df[f] == "saint priesaint en jarez", f] = 'saint priest en jarez'
    df.loc[df[f] == "saint genesaint lerpt", f] = 'saint genest lerpt'
    df.loc[df[f] == "decines", f] = 'decines charpieu'
    df.loc[df[f] == "decines charpieux", f] = 'decines charpieu'
    df.loc[df[f] == "rilleux la pape", f] = 'rillieux la pape'
    df.loc[df[f] == "saint clement sur valsonne", f] = 'tarare'
    df.loc[df[f] == "mornex", f] = 'monnetier mornex'
    df.loc[df[f] == "prevessin", f] = 'prevessin moens'
    df.loc[df[f] == "tassin", f] = 'tassin la demi lune'
    df.loc[df[f] == "meribel les allues", f] = 'les allues'
    df.loc[df[f] == "mribel", f] = 'les allues'
    df.loc[df[f] == "la voulte", f] = 'la voulte sur rhone'
    df.loc[df[f] == "la tesainte de buch", f] = 'la teste de buch'
    df.loc[df[f] == "trinite", f] = 'la trinite'
    df.loc[df[f] == "estree sur noye", f] = 'estrees sur noye'
    df.loc[df[f] == "balnquefort", f] = 'blanquefort'
    df.loc[df[f] == "couëron", f] = 'coueron'
    df.loc[df[f] == "ambares", f] = 'ambares et lagrave'
    df.loc[df[f] == "taillan medoc", f] = 'le taillan medoc'
    df.loc[df[f] == "saint pierre d'irube", f] = 'saint pierre d irube'
    df.loc[df[f] == "chateau d olonnne", f] = 'le chateau d olonnne'
    df.loc[df[f] == "saint leon sur l'ile", f] = 'saint leon sur l isle'
    df.loc[df[f] == "castets et castillon", f] = 'castillon de castets'
    df.loc[df[f] == "hossegor", f] = 'soorts hossegor'
    df.loc[df[f] == "gaillan", f] = 'gaillan en medoc'
    df.loc[df[f] == "bordeaux cauderan", f] = 'bordeaux'
    df.loc[df[f] == "cap ferret", f] = 'lege cap ferret'
    df.loc[df[f] == "les buges", f] = 'sainte severe'
    df.loc[df[f] == "grenade sur adour", f] = 'grenade sur l adour'
    df.loc[df[f] == "saint legers des bois", f] = 'saint leger des bois'
    df.loc[df[f] == "saint michel l ecluse et leparon", f] = 'la roche chalais'
    df.loc[df[f] == "grenade sur l adours", f] = 'grenade sur l adour'
    df.loc[df[f] == "pyla sur mer", f] = 'la teste de buch'
    df.loc[df[f] == "beigles", f] = 'begles'
    df.loc[df[f] == "semussaec", f] = 'semussac'
    df.loc[df[f] == "lesparre", f] = 'lesparre medoc'
    df.loc[df[f] == "andernos", f] = 'andernos les bains'
    df.loc[df[f] == "anglar saint felix", f] = 'anglars saint felix'
    df.loc[df[f] == "saint maure des fosses", f] = 'saint maur des fosses'
    df.loc[df[f] == "montgailhard", f] = 'montgaillard'
    df.loc[df[f] == "clara villerach", f] = 'clara'
    df.loc[df[f] == "saint jean deluz", f] = 'saint jean de luz'
    df.loc[df[f] == "tampon", f] = 'le tampon'
    df.loc[df[f] == "orhtez", f] = 'orthez'
    df.loc[df[f] == "cazeres sur garonne", f] = 'cazeres'
    df.loc[df[f] == "saint sulpice la pointe", f] = 'saint sulpice'
    df.loc[df[f] == "ramonville", f] = 'ramonville saint agne'
    df.loc[df[f] == "pont de l arn", f] = 'mazamet'
    df.loc[df[f] == "les loges marguerons", f] = 'les loges margueron'
    df.loc[df[f] == "la baule", f] = 'la baule escoublac'
    df.loc[df[f] == "sarlat", f] = 'sarlat la caneda'
    df.loc[df[f] == "castelnau d'estretefonds", f] = 'castelnau d estretefonds'
    df.loc[df[f] == "saint martinlalande", f] = 'saint martin lalande'
    df.loc[df[f] == "lisle en dodon", f] = 'l isle en dodon'
    df.loc[df[f] == "cazlihac", f] = 'cazilhac'
    df.loc[df[f] == "montreal du gers", f] = 'montreal'
    df.loc[df[f] == "la cadiere", f] = 'la cadiere d azur'
    df.loc[df[f] == "la cadiere d'azur", f] = 'la cadiere d azur'
    df.loc[df[f] == "seillons", f] = 'seillons source d argens'
    df.loc[df[f] == "puyricard", f] = 'aix en provence'
    df.loc[df[f] == "treilleres", f] = 'treillieres'
    df.loc[df[f] == "cannes la bocca", f] = 'cannes'
    df.loc[df[f] == "thouare", f] = 'thouare sur loire'
    df.loc[df[f] == "saint heblain", f] = 'saint herblain'
    df.loc[df[f] == "domfront en poiraie", f] = 'domfront'
    df.loc[df[f] == "saint ouen d'aunis", f] = 'saint ouen d aunis'
    df.loc[df[f] == "saint jeean de boiseau", f] = 'saint jean de boiseau'
    df.loc[df[f] == "louvigne du dt", f] = 'louvigne du desert'
    df.loc[df[f] == "amfreville saint amand", f] = 'amfreville la campagne'
    df.loc[df[f] == "plouër sur rance", f] = 'plouer sur rance'
    df.loc[df[f] == "muzilac", f] = 'muzillac'
    df.loc[df[f] == "montsecret clairefougere", f] = 'montsecret'
    df.loc[df[f] == "saint george des groseillers", f] = 'saint georges des groseillers'
    df.loc[df[f] == "portitagnes plage", f] = 'portitagnes'
    df.loc[df[f] == "anses d'arlet", f] = 'anses d arlet'
    df.loc[df[f] == "fraïsse sur agout", f] = 'fraisse sur agout'
    df.loc[df[f] == "six fours les palges", f] = 'six fours les plages'
    df.loc[df[f] == "montellier", f] = 'montpellier'
    df.loc[df[f] == "isle sur la sorgue", f] = 'l isle sur la sorgue'
    df.loc[df[f] == "sainte marie la mer", f] = 'canet'
    df.loc[df[f] == "latte", f] = 'lattes'
    df.loc[df[f] == "juans les pins", f] = 'antibes'
    df.loc[df[f] == "golfe juan", f] = 'vallauris'
    df.loc[df[f] == "chemin de l esperes", f] = 'la colle sur loup'
    df.loc[df[f] == "allemond", f] = 'allemont'
    df.loc[df[f] == "saint aygulf", f] = 'frejus'
    df.loc[df[f] == "saint mandrier", f] = 'saint mandrier sur mer'
    df.loc[df[f] == "marienthal", f] = 'haguenau'
    df.loc[df[f] == "wintzfelden", f] = 'soultzmatt'
    df.loc[df[f] == "illkirch", f] = 'illkirch graffenstaden'
    df.loc[df[f] == "foucherons", f] = 'foucherans'
    df.loc[df[f] == "wintzenheim hochersberg", f] = 'wintzenheim kochersberg'
    df.loc[df[f] == "bergholtzzell", f] = 'bergholtz zell'
    df.loc[df[f] == "behlenheim", f] = 'truchtersheim'
    df.loc[df[f] == "reitwiller", f] = 'berstett'
    df.loc[df[f] == "müllheim", f] = 'mullheim'
    df.loc[df[f] == "kientzville", f] = 'scherwiller'
    df.loc[df[f] == "wangenbourg", f] = 'wangenbourg engenthal'
    df.loc[df[f] == "servigny les sainte barb", f] = 'servigny les sainte barbe'
    df.loc[df[f] == "quesnoy sur deûle", f] = 'quesnoy sur deule'
    df.loc[df[f] == "saint waasaint la vallee", f] = 'saint waast'
    df.loc[df[f] == "teteghem coudekerque village", f] = 'teteghem'
    df.loc[df[f] == "roilaye", f] = 'saint etienne roilaye'
    df.loc[df[f] == "flines les raches", f] = 'flines lez raches'
    df.loc[df[f] == "aulnoy les valenciennes", f] = 'aulnoy lez valenciennes'
    df.loc[df[f] == "marcq en baroeuil", f] = 'marcq en baroeul'
    df.loc[df[f] == "douai frais marais", f] = 'douai'
    df.loc[df[f] == "campdeville", f] = 'milly sur therain'
    df.loc[df[f] == "arnouville", f] = 'arnouville les gonesse'
    df.loc[df[f] == "saint lambert des bois", f] = 'saint lambert'
    df.loc[df[f] == "limours", f] = 'limours en hurepoix'
    df.loc[df[f] == "ile saint denis", f] = 'l ile saint denis'
    df.loc[df[f] == "grandpuits", f] = 'grandpuits bailly carrois'
    df.loc[df[f] == "la chapelle largeau", f] = 'mauleon'
    df.loc[df[f] == "dagnieux", f] = 'dagneux'
    df.loc[df[f] == "lugon et l ile du carney", f] = 'lugon et l ile du carnay'
    df.loc[df[f] == "le passage d agen", f] = 'le passage'
    df.loc[df[f] == "coëx", f] = 'coex'
    df.loc[df[f] == "sainte", f] = 'saintes'
    df.loc[df[f] == "saint remy les chevreuses", f] = 'saint remy les chevreuse'
    df.loc[df[f] == "meudon la foret", f] = 'meudon'
    df.loc[df[f] == "villemonble", f] = 'villemomble'
    df.loc[df[f] == "vitry", f] = 'vitry sur seine'
    df.loc[df[f] == "reuil malmaison", f] = 'rueil malmaison'
    df.loc[df[f] == "rueil", f] = 'rueil malmaison'
    df.loc[df[f] == "garge les gonesses", f] = 'garges les gonesse'
    df.loc[df[f] == "garges les gonesses", f] = 'garges les gonesse'
    df.loc[df[f] == "la plaine saint denis", f] = 'aubervilliers'
    df.loc[df[f] == "saint ouen sur seine", f] = 'saint ouen'
    df.loc[df[f] == "charenton", f] = 'charenton le pont'
    df.loc[df[f] == "port marly", f] = 'le port marly'
    df.loc[df[f] == "chatelet en brie", f] = 'le chatelet en brie'
    df.loc[df[f] == "portitagnes", f] = 'portiragnes'
    df.loc[df[f] == "carriere sur seine", f] = 'carrieres sur seine'
    df.loc[df[f] == "genevilliers", f] = 'gennevilliers'
    df.loc[df[f] == "chennevieres", f] = 'chennevieres sur marne'
    df.loc[df[f] == "serran", f] = 'sevran'
    df.loc[df[f] == "arnouville les gonesses", f] = 'arnouville les gonesse'
    df.loc[df[f] == "savingy le temple", f] = 'savigny le temple'
    df.loc[df[f] == "villers saint frabourg", f] = 'villers saint frambourg'
    df.loc[df[f] == "la penne s sur  huveaune", f] = 'la penne sur huveaune'
    df.loc[df[f] == "choignes", f] = 'chamarandes choignes'
    df.loc[df[f] == "ormex", f] = 'ornex'
    df.loc[df[f] == "st. denis", f] = 'saint denis'
    df.loc[df[f] == "satigny", f] = 'savigny'
    df.loc[df[f] == "solan", f] = 'la bastide d angras'
    df.loc[df[f] == "saint andre de farivillers", f] = 'saint andre farivillers'
    df.loc[df[f] == "genneviliers", f] = 'gennevilliers'
    df.loc[df[f] == "le val saint perre", f] = 'le val saint pierre'
    df.loc[df[f] == "le cheylard", f] = 'villers saint frambourg'
    df.loc[df[f] == "villiers saint paul", f] = 'villers saint paul'
    df.loc[df[f] == "viarmes   ()", f] = 'viarmes'
    df.loc[df[f] == "voisins le btx", f] = 'voisins le bretonneux'
    df.loc[df[f] == "peroles", f] = 'perols'
    df.loc[df[f] == "contances", f] = 'coutances'
    df.loc[df[f] == "saint etienne de rouvray", f] = 'saint etienne du rouvray'
    df.loc[df[f] == "les parrichets", f] = 'mouroux'
    df.loc[df[f] == "sarge les   le mans", f] = 'sarge les le mans'
    df.loc[df[f] == "pont de casse", f] = 'pont du casse'
    df.loc[df[f] == "dombasle", f] = 'dombasle sur meurthe'
    df.loc[df[f] == "carnoux", f] = 'carnoux en provence'
    df.loc[df[f] == "saint germain en layes", f] = 'saint germain en laye'
    df.loc[df[f] == "plaine saint denis", f] = 'aubervilliers'
    df.loc[df[f] == "issy sur moulineaux", f] = 'issy les moulineaux'
    df.loc[df[f] == "villemoisson", f] = 'villemoisson sur orge'
    df.loc[df[f] == "chatou   ()", f] = 'chatou'
    df.loc[df[f] == "villeuneuve saint georges", f] = 'villeneuve saint georges'
    df.loc[df[f] == "cherbourg en cotentin (cherbourg)", f] = 'cherbourg en cotentin'
    df.loc[df[f] == "clairefontaine", f] = 'clairefontaine en yvelines'
    df.loc[df[f] == "chilli mazarin", f] = 'chilly mazarin'
    df.loc[df[f] == "sainte ", f] = 'saintes'
    df.loc[df[f] == "martoury", f] = 'saint genest lerpt'
    df.loc[df[f] == "ivry", f] = 'ivry sur seine'
    df.loc[df[f] == "rosay en multien", f] = 'rosoy en multien'
    df.loc[df[f] == "bourg beudoin", f] = 'bourg beaudoin'
    df.loc[df[f] == "bourg beudouin", f] = 'bourg beaudoin'
    df.loc[df[f] == 'equeur dreville', f] = 'equeurdreville'
    df.loc[df[f] == "aut reville saint lambert", f] = 'autreville saint lambert'
    df.loc[df[f] == "villeneuve lez avignon", f] = 'villeneuve les avignon'
    df.loc[df[f] == "puisseux en france", f] = 'puiseux en france'
    df.loc[df[f] == "optevor", f] = 'optevoz'
    df.loc[df[f] == "mandelieu", f] = 'mandelieu la napoule'
    df.loc[df[f] == "herblay sur seine", f] = 'herblay'
    df.loc[df[f] == "sauxemesnil", f] = 'saussemesnil'
    df.loc[df[f] == "le teil", f] = 'le teil d ardeche'
    df.loc[df[f] == "saint  prix", f] = 'saint prix'
    df.loc[df[f] == "soissy sur ecole", f] = 'soisy sur ecole'
    df.loc[df[f] == "fontenay les briis", f] = 'fontenay les bris'
    df.loc[df[f] == "oz en oisan", f] = 'oz'
    df.loc[df[f] == "cuerse", f] = 'uers'
    df.loc[df[f] == "thuit signol", f] = 'le thut signol'
    df.loc[df[f] == "bormeuil sur marne", f] = 'bonneuil sur marne'
    df.loc[df[f] == "voisins le bx", f] = 'voisins le bretonneux'
    df.loc[df[f] == "cely en biere", f] = 'cely'
    df.loc[df[f] == "carnoux en pce", f] = 'carnoux en provence'
    df.loc[df[f] == "vigneux", f] = 'vigneux sur seine'
    df.loc[df[f] == "rillieux", f] = 'rillieux la pape'
    df.loc[df[f] == "villars sur ollen", f] = 'villars sur ollon'
    df.loc[df[f] == "conches", f] = 'conches en ouche'
    df.loc[df[f] == "charettes", f] = 'charette'
    df.loc[df[f] == "issy lesmoulineaux", f] = 'issy les moulineaux'
    df.loc[df[f] == "le val saint pierre", f] = 'le val saint pere'
    df.loc[df[f] == "dammartin en goelle", f] = 'dammartin en goele'
    df.loc[df[f] == "ivrys sur seine", f] = 'ivry sur seine'
    df.loc[df[f] == "le chesnay rocquencourt", f] = 'le chesnay'
    df.loc[df[f] == "adge", f] = 'agde'
    df.loc[df[f] == "sain thibault des vignes", f] = 'saint thibault des vignes'
    df.loc[df[f] == "amphion les bains", f] = 'publier'
    df.loc[df[f] == "publier (amphion les bains)", f] = 'publier'
    df.loc[df[f] == "le lamertin", f] = 'le lamentin'
    df.loc[df[f] == "chevilly  larue", f] = 'chevilly larue'
    df.loc[df[f] == "bouvieux", f] = 'gouvieux'
    df.loc[df[f] == "jouars pontchartain", f] = 'jouars pontchartrain'
    df.loc[df[f] == "vendin les bethunes", f] = 'vendin les bethune'
    df.loc[df[f] == "essey les ponts", f] = 'chateauvillain'
    df.loc[df[f] == "aulnot", f] = 'saulnot'
    # df.loc[df[f] == "evry courcouronnes", f] = 'evry'
    df.loc[df[f] == "rilleux le pape", f] = 'rillieux la pape'
    df.loc[df[f] == "puget sur agens", f] = 'puget sur argens'
    df.loc[df[f] == "beychac sur caillau", f] = 'beychac et caillau'
    df.loc[df[f] == "vigoulet d  avena", f] = 'vigoulet auzil'
    df.loc[df[f] == "nante", f] = 'nantes'
    df.loc[df[f] == "saint bry sous foret", f] = 'saint brice sous foret'
    df.loc[df[f] == "morvant", f] = 'miniac morvan'
    df.loc[df[f] == "le perrey en yvelines", f] = 'le perray en yvelines'
    df.loc[df[f] == "equeudreville", f] = 'cherbourg en cotentin'
    df.loc[df[f] == "equeurdreville", f] = 'cherbourg en cotentin'
    df.loc[df[f] == "saint georges didonne", f] = 'saint georges de didonne'
    df.loc[df[f] == "saint foy les lyon", f] = 'sainte foy les lyon'
    df.loc[df[f] == "le loroux boterreau", f] = 'le loroux bottereau'
    df.loc[df[f] == "maison laffitte", f] = 'maisons laffitte'
    df.loc[df[f] == "villeneuve d'ascq", f] = 'villeneuve d ascq'
    df.loc[df[f] == "ville d'avray", f] = 'ville d avray'
    df.loc[df[f] == "collonges au mont d'or", f] = 'collonges au mont d or'
    df.loc[df[f] == "saint cyr l'ecole", f] = 'saint cyr l ecole'
    df.loc[df[f] == "saint cyr au mont d'or", f] = 'saint cyr au mont d or '
    df.loc[df[f] == "villenave d'ornon", f] = 'villenave d ornon'
    df.loc[df[f] == "l isle d'abeau", f] = 'l isle d abeau'
    df.loc[df[f] == "tain l'hermitage", f] = 'tain l hermitage'
    df.loc[df[f] == "saint didier au mont d'or ", f] = 'saint didier au mont d or'
    df.loc[df[f] == "saint didier au mont d'or", f] = 'saint didier au mont d or'
    df.loc[df[f] == "saint symphorien d'ozon", f] = 'saint symphorien d ozon'
    df.loc[df[f] == "la chapelle d'armentieres", f] = 'la chapelle d armentieres'
    df.loc[df[f] == "pont l'abbe  ", f] = 'pont l abbe'
    df.loc[df[f] == "pont l'abbe", f] = 'pont l abbe'
    df.loc[df[f] == "marcy l'etoile", f] = 'marcy l etoile'
    df.loc[df[f] == "brive", f] = 'brives'
    df.loc[df[f] == "chazay d'azergues", f] = 'chazay d azergues'
    df.loc[df[f] == "cap d'ail", f] = 'cap d ail'
    df.loc[df[f] == "bagnoles de l'orne", f] = 'bagnoles de l orne'
    df.loc[df[f] == "berre l'etang", f] = 'berre l etang'
    df.loc[df[f] == "bois d'arcy", f] = 'bois d arcy'
    df.loc[df[f] == "blainville sur l'eau", f] = 'blainville sur l eau'
    df.loc[df[f] == "carnon", f] = 'mauguio'
    df.loc[df[f] == "charvieu", f] = 'charvieu chavagneux'
    df.loc[df[f] == "chateau d''olonne", f] = 'chateau d olonne'
    df.loc[df[f] == "cournon d'auvergne", f] = 'cournon d auvergne'
    df.loc[df[f] == "couzon au mont d'or", f] = 'couzon au mont d or'
    df.loc[df[f] == "filliere (st martin bellevue)", f] = 'saint martin bellevue'
    df.loc[df[f] == "gages", f] = 'montrozier'
    df.loc[df[f] == "grenade sur l'adour", f] = 'grenade sur l adour'
    df.loc[df[f] == "la hague", f] = 'acqueville'
    df.loc[df[f] == "les adrets de l'esterel", f] = 'les adrets de l esterel'
    df.loc[df[f] == "les cotes d'arey", f] = 'les cotes d arey'
    df.loc[df[f] == "marseille e arrondissement   ()", f] = 'marseille'
    df.loc[df[f] == "oloron ste marie", f] = 'oloron sainte marie'
    df.loc[df[f] == "pont d'ain", f] = 'pont d ain'
    df.loc[df[f] == "pont de l'arn", f] = 'pont de larn'
    df.loc[df[f] == "rieux vol vestre", f] = 'rieux'
    df.loc[df[f] == "rieux volvestre", f] = 'rieux'
    df.loc[df[f] == "les issambres", f] = 'roquebrune sur argens'
    df.loc[df[f] == "saint georges d'orques", f] = 'saint georges d orques'
    df.loc[df[f] == "saint jean d'illac", f] = 'saint jean d illac'
    df.loc[df[f] == "saint laurent d'arce", f] = 'saint laurent d arce'
    df.loc[df[f] == "saint rambert d'albon", f] = 'saint rambert d albon'
    df.loc[df[f] == "saint remy l'honore", f] = 'saint remy l honore'
    df.loc[df[f] == "chateau d'olonne", f] = 'chateau d olonne'
    df.loc[df[f] == "le chateau d'olonne", f] = 'chateau d olonne'
    # df.loc[df[f] == "saint die des vosges", f] = 'saint die'
    df.loc[df[f] == "villeveque", f] = 'rives du loir en anjou'
    df.loc[df[f] == "la neuville d'aumont", f] = 'la drenne'
    df.loc[df[f] == "la neuville d aumont", f] = 'la drenne'
    df.loc[df[f] == "tremblay", f] = 'tremblay en france'
    df.loc[df[f] == "argentat", f] = 'argentat sur dordogne'
    df.loc[df[f] == "fontenany les bris", f] = 'fontenay les bris'
    df.loc[df[f] == "le thut signol", f] = 'le thuit signol'
    df.loc[df[f] == "bage la ville", f] = 'bage dommartin'
    df.loc[df[f] == "champagne en mont d'or", f] = 'champagne en mont d or'
    df.loc[df[f] == "vernoil", f] = 'vernoil le fourrier'
    df.loc[df[f] == "collonges au mt d'or", f] = 'collonges au mont d or'
    df.loc[df[f] == "macilly d'azergues", f] = 'macilly d azergues'
    df.loc[df[f] == "plan de la tour", f] = 'le plan de la tour'
    df.loc[df[f] == "albens", f] = 'entrelacs'
    df.loc[df[f] == "moulin sur yevres", f] = 'moulins sur yevre'
    df.loc[df[f] == "grosbliederestroff", f] = 'grosbliederstroff'
    df.loc[df[f] == "les carroz d'araches", f] = 'araches la frasse'
    df.loc[df[f] == "vigoulet d' avena", f] = 'vigoulet auzil'
    df.loc[df[f] == "cherbourg octeville", f] = 'cherbourg en cotentin'
    df.loc[df[f] == "orleans la source", f] = 'orleans'
    df.loc[df[f] == "masevaux", f] = 'niederbruck'
    df.loc[df[f] == "le fief sauvin", f] = 'montrevault sur evre'
    df.loc[df[f] == "garanciere en drouais", f] = 'garancieres en drouais'
    df.loc[df[f] == "belleville sur vie", f] = 'bellevigny'
    df.loc[df[f] == "belleville sur saone", f] = 'belleville en beaujolais'
    df.loc[df[f] == "siorac", f] = 'siorac en perigord'
    df.loc[df[f] == "vileurbanne", f] = 'villeurbanne'
    df.loc[df[f] == "dammartin en geole", f] = 'dammartin en goele'
    df.loc[df[f] == "dagny sur meuse", f] = 'dugny sur meuse'
    df.loc[df[f] == "champ saint pere", f] = 'le champ saint pere'
    df.loc[df[f] == "la varenne", f] = 'oree d anjou'
    df.loc[df[f] == "fontaine sur saone", f] = 'fontaines sur saone'
    df.loc[df[f] == "dinsheim sur bruches", f] = 'dinsheim sur bruche'
    df.loc[df[f] == "maur des fosses", f] = 'saint maur des fosses'
    df.loc[df[f] == "saint martin d'aux", f] = 'saint martin d auxigny'
    df.loc[df[f] == "clermond ferrand", f] = 'clermont ferrand'
    df.loc[df[f] == "la garennes colombes", f] = 'la garenne colombes'
    df.loc[df[f] == "lezignan l'eveque", f] = 'nezignan l eveque'
    df.loc[df[f] == "villennes s seine", f] = 'villennes sur seine'
    df.loc[df[f] == "euralille", f] = 'lille'
    df.loc[df[f] == "saint  etienne", f] = 'saint etienne'
    df.loc[df[f] == "nuelles", f] = 'saint germain nuelles'
    df.loc[df[f] == "besseney", f] = 'bessenay'
    df.loc[df[f] == "vigne la cote", f] = 'vignes la cote'
    df.loc[df[f] == "le perray", f] = 'le perray en yvelines'
    df.loc[df[f] == "l  etang la ville", f] = 'l etang la ville'
    df.loc[df[f] == "jouy le  moutier", f] = 'jouy le moutier'
    df.loc[df[f] == "menvielle", f] = 'villemade'
    df.loc[df[f] == "velaine en haye", f] = 'bois de haye'
    df.loc[df[f] == "grand d'agde", f] = 'grau d agde'
    df.loc[df[f] == "la chapelle basse mer", f] = 'divatte sur loire'
    df.loc[df[f] == "brain sur longuenee", f] = 'erdre en anjou'
    df.loc[df[f] == "canjuers", f] = 'aiguines'
    df.loc[df[f] == "boissy maugis", f] = 'cour maugis sur huisne'
    df.loc[df[f] == "carrieres sur  seine", f] = 'carrieres sur seine'
    df.loc[df[f] == "moretsur loing", f] = 'moret sur loing'
    df.loc[df[f] == "marly le  roi", f] = 'marly le roi'
    df.loc[df[f] == "saint maur des  fosses", f] = 'saint maur des fosses'
    df.loc[df[f] == "querqueville", f] = 'cherbourg en cotentin'
    df.loc[df[f] == "marcillyen villette", f] = 'marcilly en villette'
    df.loc[df[f] == "monthlery", f] = 'montlhery'
    df.loc[df[f] == "la teste de  buch", f] = 'la teste de buch'
    df.loc[df[f] == "saint pol sur mer", f] = 'dunkerke'
    df.loc[df[f] == "boulogne sur  mer", f] = 'boulogne sur mer'
    df.loc[df[f] == "saint suzanne", f] = 'sainte suzanne'
    df.loc[df[f] == "saint bonnet de mures", f] = 'saint bonnet de mure'
    df.loc[df[f] == "lacanau (lacanau ocean)", f] = 'lacanau'
    df.loc[df[f] == "terres de caux (bennetot)", f] = 'bennetot'
    df.loc[df[f] == "le tampon (la plaine des cafres)", f] = 'le tampon'
    df.loc[df[f] == "les avirons (tevelave)", f] = 'les avirons'
    df.loc[df[f] == "saint leu (la chaloupe)", f] = 'saint leu'
    df.loc[df[f] == "annecy (meythet)", f] = 'annecy'
    df.loc[df[f] == "bois de nefles saint paul", f] = 'bois de nefles'
    df.loc[df[f] == "saint paul (bois de nefles saint paul)", f] = 'bois de nefles'
    df.loc[df[f] == "la hague (biville)", f] = 'biville'
    df.loc[df[f] == "jaunay marigny", f] = 'jaunay clan'
    df.loc[df[f] == "saint etienne (terrenoire)", f] = 'saint etienne'
    df.loc[df[f] == "carcassonne (villalbe)", f] = 'carcassonne'
    df.loc[df[f] == "grasse (magagnosc)", f] = 'grasse'
    df.loc[df[f] == "villages du lac de paladru (le pin)", f] = 'le pin'
    df.loc[df[f] == "saint maximin la ste baume", f] = 'saint maximin la sainte baume'
    df.loc[df[f] == "pouge chambalud", f] = 'bouge chambalud'
    df.loc[df[f] == "loireauxence (la rouxiere)", f] = 'la rouxiere'
    df.loc[df[f] == "saint   andre", f] = 'saint andre'
    df.loc[df[f] == "saint denis le sponts", f] = 'saint denis les ponts'
    df.loc[df[f] == "vovic", f] = 'volvic'
    df.loc[df[f] == "cloyes les trois rivieres", f] = 'cloyes sur le loir'
    df.loc[df[f] == "juan les pins", f] = 'antibes'
    df.loc[df[f] == "mont boron", f] = 'nice'
    df.loc[df[f] == "deuil labarre", f] = 'deuil la barre'
    df.loc[df[f] == "vovic", f] = 'volvic'
    df.loc[df[f] == "la chapelle longueville", f] = 'saint just'
    df.loc[df[f] == "sablons sur huisne", f] = 'condeau'
    df.loc[df[f] == "saint jean d elle", f] = 'rouxeville'
    df.loc[df[f] == "les baroches", f] = 'briey'
    df.loc[df[f] == "montrozier (gages)", f] = 'montrozier'
    df.loc[df[f] == "saint cecile plage", f] = 'sainte cecile plage'
    df.loc[df[f] == "orleans (la source)", f] = 'orleans'
    df.loc[df[f] == "dunkerque (st pol sur mer)", f] = 'dunkerke'
    df.loc[df[f] == "berstett (rumersheim)", f] = 'berstett'
    df.loc[df[f] == "sanghin en melantois", f] = 'sainghin en melantois'
    df.loc[df[f] == " bis rue de chatillon, draveil", f] = 'draveil'
    df.loc[df[f] == "la fare des oliviers", f] = 'la fare les oliviers'
    df.loc[df[f] == "annecy (pringy)", f] = 'annecy'
    df.loc[df[f] == "besle sur vilaine", f] = 'guemene penfao'
    df.loc[df[f] == "braine l'alleud", f] = 'braine l alleud'
    df.loc[df[f] == "saint denis (la plaine saint denis)", f] = 'saint denis'
    df.loc[df[f] == "villeurbane", f] = 'villeurbanne'
    df.loc[df[f] == "sablons sur huisne (conde sur huisne)", f] = 'sablons sur huisne'
    df.loc[df[f] == "gray sur mer", f] = 'graye sur mer'
    df.loc[df[f] == "dinan (lehon)", f] = 'dinan'
    df.loc[df[f] == "puygouzon (labastide denat)", f] = 'puygouzon'
    df.loc[df[f] == "val de briey (briey)", f] = 'val de briey'
    df.loc[df[f] == "theix noyalo (theix)", f] = 'theix noyalo'
    df.loc[df[f] == "six four", f] = 'six fours les plages'
    df.loc[df[f] == "castets et castillon (castets en dorthe)", f] = 'castets et castillon'
    df.loc[df[f] == "nice (st isidore)", f] = 'nice'
    df.loc[df[f] == "beauvoir sur niort (la revetizon)", f] = 'beauvoir sur niort'
    df.loc[df[f] == "hayange (marspich)", f] = 'hayange'
    df.loc[df[f] == "saint jean d'elle (rouxeville)", f] = 'saint jean d elle'
    df.loc[df[f] == "bourgs sur colagne", f] = 'chirac'
    df.loc[df[f] == "bourg saint maurice (les arcs)", f] = 'bourg saint maurice'
    df.loc[df[f] == "roquebrune sur argens (les issambres)", f] = 'roquebrune sur argens'
    df.loc[df[f] == "compiegnes", f] = 'compiegne'
    df.loc[df[f] == "moissac delle vue", f] = 'moissac bellevue'
    df.loc[df[f] == "cannes (cannes la bocca)", f] = 'cannes'
    df.loc[df[f] == "saint paul (st gilles les hauts)", f] = 'saint paul'
    df.loc[df[f] == "la chapelle longueville (st just)", f] = 'la chapelle longueville'
    df.loc[df[f] == "chemille en anjou (chemille melay)", f] = 'chemille melay'
    df.loc[df[f] == "la cannet", f] = 'le cannet'
    df.loc[df[f] == "annecy (seynod)", f] = 'annecy'
    df.loc[df[f] == "antibes (la fontonne)", f] = 'antibes'
    df.loc[df[f] == "marseille (chateau gombert)", f] = 'marseille'
    df.loc[df[f] == "marseillle", f] = 'marseille'
    df.loc[df[f] == "noyant villages (broc)", f] = 'broc'
    df.loc[df[f] == "bras panon (riviere du mat)", f] = 'bras panon'
    df.loc[df[f] == "meudon (meudon la foret)", f] = 'meudon'
    df.loc[df[f] == "nice ", f] = 'nice'
    df.loc[df[f] == "juans les pins", f] = 'antibes'
    df.loc[df[f] == "marseille (croix rouge)", f] = 'marseille'
    df.loc[df[f] == "savigny sur orges", f] = 'savigny sur orge'
    df.loc[df[f] == "narbonne (narbonne plage)", f] = 'narbonne'
    df.loc[df[f] == "evry gregy sur yerre (gregy sur yerre)", f] = 'evry gregy sur yerre'
    df.loc[df[f] == "frontignan (la peyrade)", f] = 'frontignan'
    df.loc[df[f] == "toulon le revest les eaux", f] = 'toulon'
    df.loc[df[f] == "les pennes mirabaud", f] = 'les pennes mirabeau'
    df.loc[df[f] == "lancon de provence", f] = 'lancon provence'
    df.loc[df[f] == "annecy (pringy)", f] = 'annecy'
    df.loc[df[f] == "javerlhac la chapelle saint robert", f] = 'javerlhac et la chapelle saint robert'
    df.loc[df[f] == "dunkerque (malo les bains)", f] = 'dunkerke'
    df.loc[df[f] == "grasse (magagnosc)", f] = 'grasse'
    df.loc[df[f] == "annecy (cran gevrier)", f] = 'annecy'
    df.loc[df[f] == "meudon (meudon la foret)", f] = 'meudon'
    df.loc[df[f] == "indre (haute indre)", f] = 'indre'
    df.loc[df[f] == "rives en seine (caudebec en caux)", f] = 'rives en seine'
    df.loc[df[f] == "villefranche s sur mer", f] = 'villefranche sur sur mer'
    df.loc[df[f] == "pont ste maxence", f] = 'pont sainte maxence'
    df.loc[df[f] == "bonniere sur reine", f] = 'bonnieres sur seine'
    df.loc[df[f] == "mougon thorigne (thorigne)", f] = 'mougon thorigne'
    df.loc[df[f] == "soisy sous momenrency", f] = 'soisy sous montmorency'
    df.loc[df[f] == "chennevieres sur marnes", f] = 'chennevieres sur marne'
    df.loc[df[f] == "saint vicenc de montalt", f] = 'sant vicenc de montalt'
    df.loc[df[f] == "saint michel chef chef (tharon plage)", f] = 'saint michel chef chef'
    df.loc[df[f] == "thizy les bourgs", f] = 'bourg de thizy'
    df.loc[df[f] == "boulazac isle manoire (boulazac)", f] = 'boulazac isle manoire'
    df.loc[df[f] == "saint radegonde", f] = 'sainte radegonde'
    df.loc[df[f] == "port ste foy et ponchapt", f] = 'port sainte foy et ponchapt'
    df.loc[df[f] == "le teil d ardeche", f] = 'le teil'
    df.loc[df[f] == "val de briey (briey)", f] = 'val de briey'
    df.loc[df[f] == "longjumeau (balizy)", f] = 'longjumeau'
    df.loc[df[f] == "cambon d'albi", f] = 'cambon d albi'
    df.loc[df[f] == "frejus (st aygulf)", f] = 'frejus'
    df.loc[df[f] == "vair sur loire (st herblon)", f] = 'vair sur loire'
    df.loc[df[f] == "jaunay marigny (jaunay clan)", f] = 'jaunay marigny'
    df.loc[df[f] == "bourgoin", f] = 'bourgoin jallieu'
    df.loc[df[f] == "bourgoin jaillieu", f] = 'bourgoin jallieu'
    df.loc[df[f] == "frontignan (la peyrade)", f] = 'frontignan'
    df.loc[df[f] == "grau du roi", f] = 'le grau du roi'
    df.loc[df[f] == "vouneuil sous biard (pouzioux la jarrie)", f] = 'vouneuil sous biard'
    df.loc[df[f] == "veuzain sur loire (onzain)", f] = 'veuzain sur loire'
    df.loc[df[f] == "chatellrault", f] = 'chatellerault'
    df.loc[df[f] == "templeuve en pevele", f] = 'templeuve'
    df.loc[df[f] == "meulan en yvelines", f] = 'meulan'
    df.loc[df[f] == "mulhouse (bourtzwiller)", f] = 'mulhouse'
    df.loc[df[f] == "st.paul", f] = 'saint paul'
    df.loc[df[f] == "hermitage", f] = 'l hermitage'
    df.loc[df[f] == "petit ebersviller", f] = 'macheren'
    df.loc[df[f] == "fuveau (la barque)", f] = 'fuveau'
    df.loc[df[f] == "roumazieres", f] = 'roumazieres loubert'
    df.loc[df[f] == "saint leu (le piton saint leu)", f] = 'saint leu'
    df.loc[df[f] == "thizy les bourgs (bourg de thizy)", f] = 'bourg de thizy'
    df.loc[df[f] == "st.paul", f] = 'saint paul'
    df.loc[df[f] == "veuzain sur loire", f] = 'onzain'
    df.loc[df[f] == "vair sur loire", f] = 'saint herblon'
    df.loc[df[f] == "boulazac isle manoire", f] = 'boulazac'
    df.loc[df[f] == "mougon thorigne", f] = 'mougon'
    df.loc[df[f] == "roustrel", f] = 'rustrel'
    df.loc[df[f] == "rives en seine", f] = 'caudebec en caux'
    df.loc[df[f] == "villefranche sur mer", f] = 'nice'
    df.loc[df[f] == "arles (raphele les arles)", f] = 'arles'
    df.loc[df[f] == "macon (sennece les macon)", f] = 'macon'
    df.loc[df[f] == "lille (lomme)", f] = 'lille'
    df.loc[df[f] == "pre en pail saint samson (st samson)", f] = 'saint samson'
    df.loc[df[f] == "grandparigny (parigny)", f] = 'parigny'
    df.loc[df[f] == "saint christol lez ales", f] = 'saint christol les ales'
    df.loc[df[f] == "mauges sur loire", f] = 'la pommeraye'
    df.loc[df[f] == "banassac canilhac (banassac)", f] = 'banassac'
    df.loc[df[f] == "les abrets en dauphine (les abrets)", f] = 'les abrets'
    df.loc[df[f] == "montaigu vendee (st georges de montaigu)", f] = 'saint georges de montaigu'
    df.loc[df[f] == "valenton (val pompadour)", f] = 'valenton'
    df.loc[df[f] == "montrevault sur evre (st remy en mauges)", f] = 'saint remy en mauges'
    df.loc[df[f] == "pont des francais (le mont dore)", f] = 'mont dore'
    df.loc[df[f] == "marseilel", f] = 'marseiille'
    df.loc[df[f] == "meudon (meudon la foret)", f] = 'meudon'
    df.loc[df[f] == "montaigu vendee (montaigu)", f] = 'montaigu'
    df.loc[df[f] == "brantome en perigord (eyvirat)", f] = 'eyvirat'
    df.loc[df[f] == "   nice", f] = 'nice'
    df.loc[df[f] == "druelle balsac (druelle)", f] = 'druelle'
    df.loc[df[f] == "lille (euralille)", f] = 'lille'
    df.loc[df[f] == "rillieux la pape (crepieux la pape)", f] = 'rillieux la pape'
    df.loc[df[f] == "truchtersheim (behlenheim)", f] = 'behlenheim'
    df.loc[df[f] == "saint jacques de la landes", f] = 'saint jacques de la lande'
    df.loc[df[f] == "brunstatt didenheim (brunstatt)", f] = 'brunstatt'
    df.loc[df[f] == "terres de haute charente (roumazieres)", f] = 'roumazieres'
    df.loc[df[f] == "saintegenevievedesbois", f] = 'sainte genevieve des bois'
    df.loc[df[f] == "chevillylarue", f] = 'chevilly larue'
    df.loc[df[f] == "   villeurbanne", f] = 'villeurbanne'
    df.loc[df[f] == "villeurbanne(france)", f] = 'villeurbanne'
    df.loc[df[f] == "villeurbanne ()", f] = 'villeurbanne'
    df.loc[df[f] == "villeurbanne a lyon", f] = 'villeurbanne'
    df.loc[df[f] == "saint  genis laval", f] = 'saint genis laval'
    df.loc[df[f] == "lyon e ()", f] = 'lyon'
    df.loc[df[f] == "villeurbanne sur charpennes", f] = 'villeurbanne'
    df.loc[df[f] == "fontaines  saint martin", f] = 'fontaines saint martin'
    df.loc[df[f] == "lyon ", f] = 'lyon'
    df.loc[df[f] == "tassin la demie lune", f] = 'tassin la demi lune'
    df.loc[df[f] == "lyon eme", f] = 'lyon'
    df.loc[df[f] == "	sausset les pins", f] = 'sausset les pins'
    df.loc[df[f] == "   gentilly", f] = 'gentilly'
    df.loc[df[f] == "   le golfe juan", f] = 'le golfe juan'
    df.loc[df[f] == "   lyon ", f] = 'lyon'
    df.loc[df[f] == " lunel", f] = 'lunel'
    df.loc[df[f] == "achicoirt", f] = 'achicourt'
    df.loc[df[f] == "agde (le cap d'agde)", f] = 'agde'
    df.loc[df[f] == "aime la plagne", f] = 'aime'
    df.loc[df[f] == "aime la plagne (granier)", f] = 'granier'
    df.loc[df[f] == "ajaccio (mezzavia)", f] = 'ajaccio'
    df.loc[df[f] == "alecon", f] = 'alencon'
    df.loc[df[f] == "allainville aux bois", f] = 'allainville'
    df.loc[df[f] == "allez et casseneuve", f] = 'allez et cazeneuve'
    df.loc[df[f] == "alloinay (les alleuds)", f] = 'alloinay'
    df.loc[df[f] == "amber", f] = 'ambert'
    df.loc[df[f] == "amberieu en buget", f] = 'amberieu en bugey'
    df.loc[df[f] == "amelie les bains", f] = 'amelie les bains palalda'
    df.loc[df[f] == "amency", f] = 'amancy'
    df.loc[df[f] == "amfreville saint amant", f] = 'amfreville saint amand'
    df.loc[df[f] == "amneville les thermes", f] = 'amneville'
    df.loc[df[f] == "ancenis saint gereon (ancenis)", f] = 'ancenis'
    df.loc[df[f] == "ancy dornot", f] = 'ancy sur moselle'
    df.loc[df[f] == "ancy dornot (ancy sur moselle)", f] = 'ancy sur moselle'
    df.loc[df[f] == "andrezieu", f] = 'andrezieux'
    df.loc[df[f] == "anduz", f] = 'anduze'
    df.loc[df[f] == "anghien les bains", f] = 'enghien les bains'
    df.loc[df[f] == "angoulins sur mer", f] = 'angoulins'
    df.loc[df[f] == "angoustrine", f] = 'angoustrine villeneuve des escaldes'
    df.loc[df[f] == "annecy (annecy le vieux)", f] = 'annecy'
    df.loc[df[f] == "anthony", f] = 'antony'
    df.loc[df[f] == "antibes (cap d'antibes)", f] = 'antibes'
    df.loc[df[f] == "antibes (juan les pins)", f] = 'antibes'
    df.loc[df[f] == "arcy ste restitue", f] = 'arcy sainte restitue'
    df.loc[df[f] == "ardres (bois en ardres)", f] = 'bois en ardres'
    df.loc[df[f] == "argentueil", f] = 'argenteuil'
    df.loc[df[f] == "argentuil", f] = 'argenteuil'
    df.loc[df[f] == "argerteuil", f] = 'argenteuil'
    df.loc[df[f] == "arlay (st germain les arlay)", f] = 'arlay'
    df.loc[df[f] == "artannes", f] = 'artannes sur indre'
    df.loc[df[f] == "artigues pres bdx", f] = 'artigues pres bordeaux'
    df.loc[df[f] == "artzwiller", f] = 'arzwiller'
    df.loc[df[f] == "asienires", f] = 'asnieres sur seine'
    df.loc[df[f] == "asniaire sur seine", f] = 'asnieres sur seine'
    df.loc[df[f] == "aspach michelbach (michelbach)", f] = 'michelbach'
    df.loc[df[f] == "athis val de rouvre", f] = 'athis de l orne'
    df.loc[df[f] == "aubervillers", f] = 'aubervilliers'
    df.loc[df[f] == "aubigny  en artois", f] = 'aubigny en artois'
    df.loc[df[f] == "aubigny les clouzeaux", f] = 'aubigny'
    df.loc[df[f] == "audresy", f] = 'andresy'
    df.loc[df[f] == "aulnay (salles les aulnay)", f] = 'salles les aulnay'
    df.loc[df[f] == "aulnay s bois", f] = 'aulnay sous bois'
    df.loc[df[f] == "auneau bleury saint symphorien (auneau)", f] = 'auneau'
    df.loc[df[f] == "aure sur mer (ste honorine des pertes)", f] = 'sainte honorine des pertes'
    df.loc[df[f] == "autrans meaudre en vercors (meaudre)", f] = 'meaudre'
    df.loc[df[f] == "autun (st pantaleon)", f] = 'autun'
    df.loc[df[f] == "avecnes les guesde", f] = 'avesne les guesde'
    df.loc[df[f] == "aveyres", f] = 'arveyres'
    df.loc[df[f] == "avignon (montfavet)", f] = 'avignon'
    df.loc[df[f] == "avise", f] = 'avize'
    df.loc[df[f] == "ay champagne", f] = 'mareuil sur ay'
    df.loc[df[f] == "ay champagne (mareuil sur ay)", f] = 'mareuil sur ay'
    df.loc[df[f] == "aymes", f] = 'mizoen'
    df.loc[df[f] == "baccarat (badmenil)", f] = 'badmenil'
    df.loc[df[f] == "bage dommartin (bage la ville)", f] = 'bage la ville'
    df.loc[df[f] == "bagnere de luchon", f] = 'bagneres de luchon'
    df.loc[df[f] == "bagneres de luchon (superbagneres)", f] = 'bagneres de luchon'
    df.loc[df[f] == "bagnoles de l'orne normandie", f] = 'bagnoles de l orne'
    df.loc[df[f] == "baignes ste radegonde", f] = 'baignes sainte radegonde'
    df.loc[df[f] == "bailleul sur thenain", f] = 'bailleul sur therain'
    df.loc[df[f] == "bain sur oust", f] = 'bains sur oust'
    df.loc[df[f] == "balainvillier", f] = 'balainvilliers'
    df.loc[df[f] == "ballainuilliers", f] = 'balainvilliers'
    df.loc[df[f] == "ballaruc", f] = 'balaruc les bains'
    df.loc[df[f] == "ballon saint mars", f] = 'ballon'
    df.loc[df[f] == "banassac canilhac", f] = 'canilhac'
    df.loc[df[f] == "bardonville", f] = 'bardouville'
    df.loc[df[f] == "bargemont", f] = 'bargemon'
    df.loc[df[f] == "barisis aux bois", f] = 'barisis'
    df.loc[df[f] == "barre les alpes", f] = 'pelvoux'
    df.loc[df[f] == "bartenhein", f] = 'barteneim'
    df.loc[df[f] == "baschepe", f] = 'boeschepe'
    df.loc[df[f] == "bassillac", f] = 'bassilac'
    df.loc[df[f] == "bassillac et auberoche", f] = 'bassilac'
    df.loc[df[f] == "bassillac et auberoche (bassillac)", f] = 'bassilac'
    df.loc[df[f] == "bauge en anjou (bauge)", f] = 'bauge'
    df.loc[df[f] == "bauge en anjou (fougere)", f] = 'fougere'
    df.loc[df[f] == "baumes de venise", f] = 'beaumes de venise'
    df.loc[df[f] == "bazencourt", f] = 'bazancourt'
    df.loc[df[f] == "bazincort sur saulx", f] = 'bazincourt sur saulx'
    df.loc[df[f] == "beaufort en anjou (beaufort en vallee)", f] = 'beaufort en vallee'
    df.loc[df[f] == "beaulieu sous bressuire", f] = 'bressuire'
    df.loc[df[f] == "beaumont du gatinois", f] = 'beaumont du gatinais'
    df.loc[df[f] == "beaumont louestault (beaumont la ronce)", f] = 'beaumont la ronce'
    df.loc[df[f] == "beaumont saint cyr (beaumont)", f] = 'beaumonr'
    df.loc[df[f] == "beaumont saint cyr (st cyr)", f] = 'saint cyr'
    df.loc[df[f] == "beaumontois en perigord", f] = 'beaumont du perigord'
    # df.loc[df[f] == "beaupreau en mauges", f] = 'beaupreau'
    df.loc[df[f] == "beaupreau en mauges (beaupreau)", f] = 'beaupreau'
    df.loc[df[f] == "beaupreau en mauges (geste)", f] = 'geste'
    df.loc[df[f] == "beaupreau en mauges (jallais)", f] = 'jallais'
    df.loc[df[f] == "beaurains les noyons", f] = 'beaurains les noyon'
    df.loc[df[f] == "beaussais vitre (vitre)", f] = 'vitre'
    df.loc[df[f] == "beauvallon (st andeol le chateau)", f] = 'saint andeol le chateau'
    df.loc[df[f] == "beauvy la foret", f] = 'beuvry la foret'
    df.loc[df[f] == "begnolet", f] = 'bagnolet'
    df.loc[df[f] == "belbeze en lomagne", f] = 'belbese'
    df.loc[df[f] == "belin beliet (beliet)", f] = 'beliet'
    df.loc[df[f] == "belleau (morey)", f] = 'morey'
    df.loc[df[f] == "bellegrade en marche", f] = 'bellegarde en marche'
    df.loc[df[f] == "bellevigne", f] = 'eraville'
    df.loc[df[f] == "bellevigny (belleville sur vie)", f] = 'belleville sur vie'
    df.loc[df[f] == "bellevigny (saligny)", f] = 'saligny'
    df.loc[df[f] == "belleville en beaujolais", f] = 'belleville'
    df.loc[df[f] == "belleville en beaujolais (belleville)", f] = 'belleville'
    df.loc[df[f] == "bellinghem (inghem)", f] = 'inghem'
    df.loc[df[f] == "berck sur mer", f] = 'berck'
    df.loc[df[f] == "bergesserim", f] = 'bergesserin'
    df.loc[df[f] == "bernes les fontaines", f] = 'pernes les fontaines'
    df.loc[df[f] == "berniere sur mer", f] = 'bernieres sur mer'
    df.loc[df[f] == "bernwiller (ammertzwiller)", f] = 'ammertzwiller'
    df.loc[df[f] == "besse sur issol", f] = 'besse sur issole'
    df.loc[df[f] == "bettainvilles", f] = 'bettainvillers'
    df.loc[df[f] == "betton bettonnet", f] = 'betton bettonet'
    df.loc[df[f] == "bevry la foret", f] = 'beuvry la foret'
    df.loc[df[f] == "beychac et cailleau", f] = 'beychac et caillau'
    df.loc[df[f] == "bezon", f] = 'bezons'
    df.loc[df[f] == "bieville beauville", f] = 'bieville beuville'
    df.loc[df[f] == "bievre", f] = 'bievres'
    df.loc[df[f] == "binic etables sur mer", f] = 'binic'
    df.loc[df[f] == "binic etables sur mer (binic)", f] = 'binic'
    df.loc[df[f] == "binic etables sur mer (etables sur mer)", f] = 'etables sur mer'
    df.loc[df[f] == "binon sur verdon", f] = 'vinon sur verdon'
    df.loc[df[f] == "biscaros", f] = 'biscarrosse'
    df.loc[df[f] == "biscarosse", f] = 'biscarrosse'
    df.loc[df[f] == "biscarrosse (biscarrosse plage)", f] = 'biscarrosse'
    df.loc[df[f] == "bischeim", f] = 'bischheim'
    df.loc[df[f] == "blainville leau", f] = 'blainville sur l eau'
    df.loc[df[f] == "blancs coteaux (voipreux)", f] = 'voipreux'
    df.loc[df[f] == "blodelstein", f] = 'blodelsheim'
    df.loc[df[f] == "bobigy", f] = 'bobigny'
    df.loc[df[f] == "bodeaux", f] = 'bordeaux'
    df.loc[df[f] == "boirargues", f] = 'lattes'
    df.loc[df[f] == "bois colombe", f] = 'bois colombes'
    df.loc[df[f] == "bois d'arey", f] = 'bois d arey'
    df.loc[df[f] == "bois_colombes", f] = 'bois colombes'
    df.loc[df[f] == "boischampre", f] = 'saint christophe le jajolet'
    df.loc[df[f] == "boissise", f] = 'boissise le roi'
    df.loc[df[f] == "boissys saint leger", f] = 'boissy saint leger'
    df.loc[df[f] == "boistroff", f] = 'boustroff'
    df.loc[df[f] == "bon repos sur blavet (laniscat)", f] = 'laniscat'
    df.loc[df[f] == "bonneville saint avit de fumadieres", f] = 'bonneville et saint avit de fumadieres'
    df.loc[df[f] == "bordeaux ", f] = 'bordeaux'
    df.loc[df[f] == "bormes", f] = 'bormes les mimosas'
    df.loc[df[f] == "bosseneville", f] = 'basseneville'
    df.loc[df[f] == "bougenais", f] = 'bouguenais'
    df.loc[df[f] == "bougoin jallieu", f] = 'bourgoin jallieu'
    df.loc[df[f] == "bouguenais les couets", f] = 'bouguenais'
    df.loc[df[f] == "bouguenay", f] = 'bouguenais'
    df.loc[df[f] == "bouilladisse", f] = 'la bouilladisse'
    df.loc[df[f] == "boularzax", f] = 'boulazac'
    df.loc[df[f] == "boulay", f] = 'boulay moselle'
    df.loc[df[f] == "boulazac isle manoire (atur)", f] = 'atur'
    df.loc[df[f] == "bouloc en quercy", f] = 'bouloc'
    df.loc[df[f] == "boulogne billancourt, jour", f] = 'boulogne billancourt'
    df.loc[df[f] == "boulognes billancourt", f] = 'boulogne billancourt'
    df.loc[df[f] == "bourcefranc", f] = 'bourcefranc le chapus'
    df.loc[df[f] == "bourg sur gironde", f] = 'bourg'
    df.loc[df[f] == "bourgoin jailleu", f] = 'bourgoin jallieu'
    df.loc[df[f] == "bourgtheroulde", f] = 'bourgtheroulde infreville'
    df.loc[df[f] == "bourgvallees (la mancelliere sur vire)", f] = 'la mancelliere sur vire'
    df.loc[df[f] == "bourgvallees (st romphaire)", f] = 'saint romphaire'
    df.loc[df[f] == "bourgvallees (st samson de bonfosse)", f] = 'saint samson de bonfosse'
    df.loc[df[f] == "boutiers", f] = 'boutiers saint trojan'
    df.loc[df[f] == "bouxienes aux dames", f] = 'bouxieres aux dames'
    df.loc[df[f] == "brantome en perigord (brantome)", f] = 'brantome'
    df.loc[df[f] == "bressac", f] = 'saint lager bressac'
    df.loc[df[f] == "bressuire (terves)", f] = 'bressuire'
    df.loc[df[f] == "bretteville en saire", f] = 'bretteville'
    df.loc[df[f] == "breuil en vexin", f] = 'brueil en vexin'
    df.loc[df[f] == "bricquebec en cotentin (bricquebec)", f] = 'bricquebec'
    df.loc[df[f] == "brie compte robert", f] = 'brie comte robert'
    df.loc[df[f] == "brie conte robert", f] = 'brie comte robert'
    df.loc[df[f] == "brignis", f] = 'brignais'
    df.loc[df[f] == "brissac loire aubance (brissac quince)", f] = 'brissac quince'
    df.loc[df[f] == "bruay", f] = 'bruay la bruissiere'
    df.loc[df[f] == "bruguiere", f] = 'bruguieres'
    df.loc[df[f] == "bruyere", f] = 'bruyeres le chatel'
    df.loc[df[f] == "bruyere le chatel", f] = 'bruyeres le chatel'
    df.loc[df[f] == "bruyeres de chatel", f] = 'bruyere le chatel'
    df.loc[df[f] == "buis les baronies", f] = 'buis les baronnies'
    df.loc[df[f] == "buissy saint georges", f] = 'bussy saint georges'
    df.loc[df[f] == "burnhapt le haut", f] = 'burnhaupt le haut'
    df.loc[df[f] == "bussy saint george", f] = 'bussy saint georges'
    df.loc[df[f] == "bussy saintgeorges", f] = 'bussy saint georges'
    df.loc[df[f] == "buzet", f] = 'buzet sur baise'
    df.loc[df[f] == "cabanac et villagrins", f] = 'cabanac et villagrains'
    df.loc[df[f] == "cabbries", f] = 'cabries'
    df.loc[df[f] == "cabries (calas)", f] = 'cabries'
    df.loc[df[f] == "cabries d'avignon", f] = 'cabries'
    df.loc[df[f] == "cadiere d'azur", f] = 'cadiere d azur'
    df.loc[df[f] == "cadouin", f] = 'buisson de cadouin'
    df.loc[df[f] == "cagne sur mer", f] = 'cagnes sur mer'
    df.loc[df[f] == "cagnes", f] = 'cagnes sur mer'
    df.loc[df[f] == "cagnes s sur  mer", f] = 'cagnes sur mer'
    df.loc[df[f] == "cagnes sur mer (cros de cagnes)", f] = 'cagnes sur mer'
    df.loc[df[f] == "caillaiel crepigny", f] = 'caillouel crepigny'
    df.loc[df[f] == "cailleux", f] = 'cayeux sur mer'
    df.loc[df[f] == "callac de bretagne", f] = 'callac'
    df.loc[df[f] == "camaret", f] = 'camaret sur mer'
    df.loc[df[f] == "camblanes", f] = 'camblanes et meynac'
    df.loc[df[f] == "cambon d albi", f] = 'cambon'
    df.loc[df[f] == "campagne les wardreques", f] = 'campagne les wardrecques'
    df.loc[df[f] == "campet", f] = 'campet et lamoleres'
    df.loc[df[f] == "campiegne", f] = 'compiegne'
    df.loc[df[f] == "canaret sur aygues", f] = 'camaret sur aygues'
    df.loc[df[f] == "candrange", f] = 'gandrange'
    df.loc[df[f] == "canet en rousillon", f] = 'canet plage'
    df.loc[df[f] == "canet en roussillon (canet plage)", f] = 'canet plage'
    df.loc[df[f] == "cannes de la bocca", f] = 'cannes'
    df.loc[df[f] == "capavenir vosges (thaon les vosges)", f] = 'thaon les vosges'
    df.loc[df[f] == "capdenac le haut", f] = 'capdenac'
    df.loc[df[f] == "capvern les bains", f] = 'capvern'
    df.loc[df[f] == "carbries", f] = 'cabries'
    df.loc[df[f] == "carcares ste croix", f] = 'carcares sainte croix'
    df.loc[df[f] == "carcasonne", f] = 'carcassonne'
    df.loc[df[f] == "carcassonne (maquens)", f] = 'carcassonne'
    df.loc[df[f] == "carcassonne (montlegun)", f] = 'carcassonne'
    df.loc[df[f] == "carcelles", f] = 'sarcelles'
    df.loc[df[f] == "carentan les marais (carentan)", f] = 'carentan'
    df.loc[df[f] == "carhaix", f] = 'carhaix plouguer'
    df.loc[df[f] == "carignan bordeaux", f] = 'carignan de bordeaux'
    df.loc[df[f] == "carqueranne", f] = 'carqueiranne'
    df.loc[df[f] == "carriere sous poissy", f] = 'carrieres sous poissy'
    df.loc[df[f] == "carsac", f] = 'carsac aillac'
    df.loc[df[f] == "cassac", f] = 'radenac'
    df.loc[df[f] == "castagnier", f] = 'castagniers'
    df.loc[df[f] == "castaignes soustons", f] = 'castaignos soustons'
    df.loc[df[f] == "castelginet", f] = 'castelginest'
    df.loc[df[f] == "castelnau d'estretefond", f] = 'castelnau d estretefond'
    df.loc[df[f] == "castelnau du medoc", f] = 'castelnau de medoc'
    df.loc[df[f] == "castelnau le lez aubagne", f] = 'castelnau le lez'
    df.loc[df[f] == "castelnou d'estretefonds", f] = 'castelnou d estretefonds'
    df.loc[df[f] == "castet en dorthe", f] = 'castets en dorthe'
    df.loc[df[f] == "castets et castillon", f] = 'castillon de castets'
    df.loc[df[f] == "castrie", f] = 'castres'
    df.loc[df[f] == "catres", f] = 'castres'
    df.loc[df[f] == "cattenon", f] = 'cattenom'
    df.loc[df[f] == "cavalaire", f] = 'cavalaire sur mer'
    df.loc[df[f] == "cehvilier la rue", f] = 'chevilly larue'
    df.loc[df[f] == "ceignac", f] = 'calmont'
    df.loc[df[f] == "ceirp gaud", f] = 'cierp gaud'
    df.loc[df[f] == "cely en biere (cely)", f] = 'cely'
    df.loc[df[f] == "cenac saint julien", f] = 'cenac et saint julien'
    df.loc[df[f] == "cernuay", f] = 'cernay'
    df.loc[df[f] == "cessonsevigne", f] = 'casson sevigne'
    df.loc[df[f] == "cezardrieux", f] = 'lezardrieux'
    df.loc[df[f] == "chabaniere", f] = 'saint maurice sur dargoire'
    df.loc[df[f] == "chabaniere (st maurice sur dargoire)", f] = 'saint maurice sur dargoire'
    df.loc[df[f] == "chabaniere (st sorlin)", f] = 'saint sorlin'
    df.loc[df[f] == "chagnolet", f] = 'dompierre sur mer'
    df.loc[df[f] == "chalanpe", f] = 'chalampe'
    df.loc[df[f] == "chaleims", f] = 'chaleins'
    df.loc[df[f] == "chambaron sur morge", f] = 'cellule'
    df.loc[df[f] == "chambley", f] = 'chambley bussieres'
    df.loc[df[f] == "champigny sur marcne", f] = 'champigny sur marne'
    df.loc[df[f] == "champigny sur marne (coeuilly)", f] = 'champigny sur marne'
    df.loc[df[f] == "champigny sur yonne", f] = 'champigny'
    df.loc[df[f] == "champneville", f] = 'champneuville'
    df.loc[df[f] == "champs saint marnes", f] = 'champs sur marne'
    df.loc[df[f] == "champtoceau", f] = 'champtoceaux'
    df.loc[df[f] == "champvecinel", f] = 'champcevinel'
    df.loc[df[f] == "chanverrie (la verrie)", f] = 'la verrie'
    df.loc[df[f] == "charanton le pont", f] = 'charenton le pont'
    df.loc[df[f] == "charbonniere les bains", f] = 'charbonnieres les bains'
    df.loc[df[f] == "charenton le port", f] = 'charenton le pont'
    df.loc[df[f] == "charleville meziere", f] = 'charleville mezieres'
    df.loc[df[f] == "charly sur marne", f] = 'charly'
    df.loc[df[f] == "charnay sur ain", f] = 'charnoz sur ain'
    df.loc[df[f] == "charnoz", f] = 'charnoz sur ain'
    df.loc[df[f] == "charteloup les vignes", f] = 'chanteloup les vignes'
    df.loc[df[f] == "chartre de bretagne", f] = 'chartres de bretagne'
    df.loc[df[f] == "charvieu chavagneux (chavagneux)", f] = 'chavagneux'
    df.loc[df[f] == "charvieux chavagneux", f] = 'chavagneux'
    df.loc[df[f] == "chateau d'oleron", f] = 'chateau d oleron'
    df.loc[df[f] == "chateau gautier", f] = 'chateau gontier'
    df.loc[df[f] == "chateau gontier sur mayenne", f] = 'chateau gontier'
    df.loc[df[f] == "chateau neuf les martigues", f] = 'chateauneuf les martigues'
    df.loc[df[f] == "chateau salin", f] = 'chateau salins'
    df.loc[df[f] == "chateaubriand", f] = 'chateaubriant'
    df.loc[df[f] == "chateaugiron (osse)", f] = 'ose'
    df.loc[df[f] == "chatel guyon", f] = 'saint hippolyte'
    df.loc[df[f] == "chatel guyon (st hippolyte)", f] = 'saint hippolyte'
    df.loc[df[f] == "chatelaillon", f] = 'chatelaillon plage'
    df.loc[df[f] == "chatelet sur retourne", f] = 'le chatelet sur retourne'
    df.loc[df[f] == "chatellerault (targe)", f] = 'chatellerault'
    df.loc[df[f] == "chatemois", f] = 'chatenois'
    df.loc[df[f] == "chatenay mallabry", f] = 'chatenay malabry'
    df.loc[df[f] == "chatenoy malabry", f] = 'chatenay malabry'
    df.loc[df[f] == "chateny malbry", f] = 'chatenay malabry'
    df.loc[df[f] == "chatillon  sur chalarone", f] = 'chatillon sur chalaronne'
    df.loc[df[f] == "chaton", f] = 'vendrest'
    df.loc[df[f] == "chatuzange", f] = 'chatuzange le goubet'
    df.loc[df[f] == "chaumes en retz (arthon en retz)", f] = 'arthon en retz'
    df.loc[df[f] == "chaumon porcie", f] = 'chaumont porcien'
    df.loc[df[f] == "chauveyriat", f] = 'chaveyriat'
    df.loc[df[f] == "chavilly laue", f] = 'chevilly larue'
    df.loc[df[f] == "chembourcy", f] = 'chambourcy'
    df.loc[df[f] == "chemille en anjou (la tourlandry)", f] = 'la tourlandry'
    df.loc[df[f] == "chemille en anjou (ste christine)", f] = 'la tourlandry'
    df.loc[df[f] == "chemille melay", f] = 'chemille'
    df.loc[df[f] == "chemire le godin", f] = 'chemire le gaudin'
    df.loc[df[f] == "chenneviere", f] = 'chennevieres sur marne'
    df.loc[df[f] == "chenneviere sur marne", f] = 'chennevieres sur marne'
    df.loc[df[f] == "chennevieres sim", f] = 'chennevieres sur marne'
    df.loc[df[f] == "chenoise cucharmoy (chenoise)", f] = 'chenoise'
    df.loc[df[f] == "cherbourg", f] = 'cherbourg en cotentin'
    df.loc[df[f] == "cherbourg en cotentin (la glacerie)", f] = 'cherbourg en cotentin'
    df.loc[df[f] == "cherbourg en cotentin (octeville)", f] = 'cherbourg en cotentin'
    df.loc[df[f] == "cherbourg en cotentin (querqueville)", f] = 'cherbourg en cotentin'
    df.loc[df[f] == "cherbourg en cotentin (tourlaville)", f] = 'cherbourg en cotentin'
    df.loc[df[f] == "chetenay malabry", f] = 'chatenay malabry'
    df.loc[df[f] == "cheuilly larue", f] = 'chevilly larue'
    df.loc[df[f] == "cheville la rue", f] = 'chevilly larue'
    df.loc[df[f] == "chilly  masarin", f] = 'chilly mazarin'
    df.loc[df[f] == "chilly masarain", f] = 'chilly mazarin'
    df.loc[df[f] == "chilly masarin", f] = 'chilly mazarin'
    df.loc[df[f] == "chiry ourscamp", f] = 'ourscamp'
    df.loc[df[f] == "chiry ourscamp", f] = 'ourscamp'
    df.loc[df[f] == "choisy de roi", f] = 'choisy le roi'
    df.loc[df[f] == "cholet (le puy saint bonnet)", f] = 'cholet'
    df.loc[df[f] == "chraenton le pont", f] = 'charenton le pont'
    df.loc[df[f] == "chrbonniere les bains", f] = 'charbonniere les bains'
    df.loc[df[f] == "cire d'avris", f] = 'cire d avris'
    df.loc[df[f] == "civreac de blaye", f] = 'civrac de blaye'
    df.loc[df[f] == "clamar", f] = 'clamart'
    df.loc[df[f] == "clapier", f] = 'clapiers'
    df.loc[df[f] == "claye sailly", f] = 'claye souilly'
    df.loc[df[f] == "clayes souilly", f] = 'claye souilly'
    df.loc[df[f] == "cleden", f] = 'cleden cap sizun'
    df.loc[df[f] == "clermont  ferrand", f] = 'clermont ferrand"'
    df.loc[df[f] == "clermont de l'oise", f] = 'clermont de l oise'
    df.loc[df[f] == "clion sur seugne", f] = 'clion'
    df.loc[df[f] == "clos des erables", f] = 'neville sur saone'
    df.loc[df[f] == "cochere", f] = 'cocheren'
    df.loc[df[f] == "coinchy", f] = 'cuinchy'
    df.loc[df[f] == "collonge au mont d'or", f] = 'collonge au mont d or'
    df.loc[df[f] == "colomby anguerny (anguerny)", f] = 'anguerny'
    df.loc[df[f] == "conde en normandie (conde sur noireau)", f] = 'conde sur noireau'
    df.loc[df[f] == "conde sainte libiare", f] = 'conde sainte libiaire'
    df.loc[df[f] == "conde ste libiaire", f] = 'conde sainte libiaire'
    df.loc[df[f] == "conflans", f] = 'conflans sainte honorine'
    df.loc[df[f] == "contamine montjoie", f] = 'les contamines montjoie'
    df.loc[df[f] == "contoire hamel", f] = 'contoire'
    df.loc[df[f] == "corbeil essones", f] = 'corbeil essonnes'
    df.loc[df[f] == "corbeil essonne", f] = 'corbeil essonnes'
    df.loc[df[f] == "corbeille essone", f] = 'corbeil essonnes'
    df.loc[df[f] == "corbeilles essonnes", f] = 'corbeil essonnes'
    df.loc[df[f] == "corbieres en provence", f] = 'corbieres'
    df.loc[df[f] == "cormeille en parisis", f] = 'cormeilles en parisis'
    df.loc[df[f] == "cornebarieux", f] = 'cornebarrieu'
    df.loc[df[f] == "cornion", f] = 'cornillon'
    df.loc[df[f] == "corqueironne", f] = 'carqueiranne'
    df.loc[df[f] == "coudekerque village", f] = 'dunkerke'
    df.loc[df[f] == "coulans", f] = 'coulans sur gee'
    df.loc[df[f] == "coulomiers", f] = 'coulommiers'
    df.loc[df[f] == "coulouneix charmier", f] = 'coulounieix chamiers'
    df.loc[df[f] == "coulounieix et chamiers", f] = 'coulounieix chamiers'
    df.loc[df[f] == "coulvagny", f] = 'saint amand sur fion'
    df.loc[df[f] == "courcelles chaussy (landonvillers)", f] = 'landonvillers'
    df.loc[df[f] == "courcelles sur vesle", f] = 'courcelles sur vesles'
    df.loc[df[f] == "courneuve", f] = 'la courneuve'
    df.loc[df[f] == "courriere", f] = 'courrieres'
    df.loc[df[f] == "cours (cours la ville)", f] = 'cours la ville'
    df.loc[df[f] == "coux et bigaroque mouzens", f] = 'mouzens'
    df.loc[df[f] == "cozon", f] = 'crozon'
    df.loc[df[f] == "crenay", f] = 'foulain'
    df.loc[df[f] == "crepy les reaux", f] = 'crepy en valois'
    df.loc[df[f] == "crespiere", f] = 'crespieres'
    df.loc[df[f] == "creteil (bonneuil sur marne)", f] = 'bonneuil sur marne'
    df.loc[df[f] == "crets en belledone", f] = 'crets en belledonne'
    df.loc[df[f] == "creully sur seulles (creully)", f] = 'creully'
    df.loc[df[f] == "creutznold", f] = 'creutzwald'
    df.loc[df[f] == "croissy", f] = 'croissy sur seine'
    df.loc[df[f] == "croix,", f] = 'croix'
    df.loc[df[f] == "crozon (morgat)", f] = 'morgat'
    df.loc[df[f] == "cucq (stella)", f] = 'cucq'
    df.loc[df[f] == "cuges les pin", f] = 'cuges les pins'
    df.loc[df[f] == "cuziau", f] = 'cuzieu'
    df.loc[df[f] == "cyfour", f] = 'six fours les plages'
    df.loc[df[f] == "dabo (schaeferhof)", f] = 'schaeferhof'
    df.loc[df[f] == "dainvile", f] = 'dainville'
    df.loc[df[f] == "dambach   la   ville", f] = 'dambach la ville'
    df.loc[df[f] == "dammantin en goele", f] = 'dammartin en goele'
    df.loc[df[f] == "dammarie leslys", f] = 'dammarie les lys'
    df.loc[df[f] == "dammaries les lys", f] = 'dammarie les lys'
    df.loc[df[f] == "dammartin", f] = 'dammartin en goele'
    df.loc[df[f] == "dampierre en yveline", f] = 'dampierre en yvelines'
    df.loc[df[f] == "darmvillers", f] = 'damvillers'
    df.loc[df[f] == "dauendorf (neubourg)", f] = 'dauendorf'
    df.loc[df[f] == "daummartin en goele", f] = 'dammartin en goele'
    df.loc[df[f] == "deuil la  barre", f] = 'deuil la barre'
    df.loc[df[f] == "deuil la bonne", f] = 'deuil la barre'
    df.loc[df[f] == "dialan sur chaine (jurques)", f] = 'jurques'
    df.loc[df[f] == "dieppe (neuville les dieppe)", f] = 'dieppe'
    df.loc[df[f] == "dinsheim sur bruche", f] = 'dinsheim'
    df.loc[df[f] == "divatte sur loire (barbechat)", f] = 'barbechat'
    df.loc[df[f] == "divonne", f] = 'divonne les bains'
    df.loc[df[f] == "domlemesnil", f] = 'dom le mesnil'
    df.loc[df[f] == "dompart", f] = 'dampmart'
    df.loc[df[f] == "dompierre sur mer (chagnolet)", f] = 'dompierre sur mer'
    df.loc[df[f] == "dompieux sur mer", f] = 'dompierre sur mer'
    df.loc[df[f] == "doncourt les confians", f] = 'doncourt les conflans'
    df.loc[df[f] == "donnemare dontilly", f] = 'donnemarie dontilly'
    df.loc[df[f] == "douai (frais marais)", f] = 'douai'
    df.loc[df[f] == "doue en anjou", f] = 'doue la fontaine'
    df.loc[df[f] == "douvres de delivrande", f] = 'douvres la delivrande'
    df.loc[df[f] == "douvrine", f] = 'douvrin'
    df.loc[df[f] == "ducey les cheris (ducey)", f] = 'ducey'
    df.loc[df[f] == "dunkerque (fort mardyck)", f] = 'dunkerque'
    df.loc[df[f] == "dunkerque (mardyck)", f] = 'dunkerque'
    df.loc[df[f] == "dunkerque (petite synthe)", f] = 'dunkerque'
    df.loc[df[f] == "dunkerque (rosendael)", f] = 'dunkerque'
    df.loc[df[f] == "eaubonn", f] = 'eaubonne'
    df.loc[df[f] == "echilais", f] = 'echillais'
    df.loc[df[f] == "ecourt saint   quentin", f] = 'ecourt saint quentin'
    df.loc[df[f] == "ecouves (vingt hanaps)", f] = 'ecouves'
    df.loc[df[f] == "ecqueuilly", f] = 'ecquevilly'
    df.loc[df[f] == "eguieres", f] = 'eyguieres'
    df.loc[df[f] == "elincourt ste marguerite", f] = 'elincourt sainte marguerite'
    df.loc[df[f] == "ellancourt", f] = 'elancourt'
    df.loc[df[f] == "ensues", f] = 'ensues la redonne'
    df.loc[df[f] == "entressen", f] = 'istres'
    df.loc[df[f] == "epagny metz tessy", f] = 'epagny'
    df.loc[df[f] == "epagny metz tessy (epagny)", f] = 'epagny'
    df.loc[df[f] == "epagny metz tessy (metz tessy)", f] = 'metz tessy'
    df.loc[df[f] == "epemay", f] = 'epernay'
    df.loc[df[f] == "epernry", f] = 'epernay'
    df.loc[df[f] == "eperon", f] = 'epron'
    df.loc[df[f] == "epinay ss senart", f] = 'epinay sur seine'
    df.loc[df[f] == "epiney sur seine", f] = 'epinay sur seine'
    df.loc[df[f] == "equehin plage", f] = 'equihen plage'
    df.loc[df[f] == "equeurdreville hanneville", f] = 'equeurdreville hainneville'
    df.loc[df[f] == "erdre en anjou (gene)", f] = 'gene'
    df.loc[df[f] == "erdre en anjou (la poueze)", f] = 'la poueze'
    df.loc[df[f] == "ermainville", f] = 'emerainville'
    df.loc[df[f] == "erquingem lys", f] = 'erquinghem lys"'
    df.loc[df[f] == "escaupont", f] = 'escautpont'
    df.loc[df[f] == "escourle", f] = 'escource'
    df.loc[df[f] == "escuresur favieres", f] = 'escure sur favieres'
    df.loc[df[f] == "espnasses", f] = 'espinasse'
    df.loc[df[f] == "essarts en bocage (l oie)", f] = 'les essarts'
    df.loc[df[f] == "essarts en bocage (les essarts)", f] = 'les essarts'
    df.loc[df[f] == "essouvert (st denis du pin)", f] = 'saint denis du pin'
    df.loc[df[f] == "esterri d' aneu", f] = 'esterri d aneu'
    df.loc[df[f] == "esterri d'aneu", f] = 'esterri d aneu'
    df.loc[df[f] == "esupcom lille", f] = 'lille'
    df.loc[df[f] == "etaulies", f] = 'etauliers'
    df.loc[df[f] == "etaux", f] = 'etaux'
    df.loc[df[f] == "euville (aulnois sous vertuzey)", f] = 'aulnois sous vertuzey'
    df.loc[df[f] == "evellys (moustoir remungol)", f] = 'moustoir remungol'
    df.loc[df[f] == "evellys (naizin)", f] = 'naizin'
    df.loc[df[f] == "evlian", f] = 'evian'
    df.loc[df[f] == "eyguiere", f] = 'eyguieres'
    df.loc[df[f] == "eyguilles", f] = 'eguilles'
    df.loc[df[f] == "faillans", f] = 'saillans'
    df.loc[df[f] == "falquemont", f] = 'faulquemont'
    df.loc[df[f] == "faverges seythenex", f] = 'faverges'
    df.loc[df[f] == "faverges seythenex (faverges)", f] = 'faverges'
    df.loc[df[f] == "fecherolles", f] = 'feucherolles'
    df.loc[df[f] == "fere champenaise", f] = 'fere champenoise'
    df.loc[df[f] == "fille sur sarthe", f] = 'fille'
    df.loc[df[f] == "filliere", f] = 'groisy'
    df.loc[df[f] == "flers en escreubieu", f] = 'flers en escreubieux'
    df.loc[df[f] == "fleurigny", f] = 'thorigny sur oreuse'
    df.loc[df[f] == "fleury d'aude", f] = 'fleury d aude'
    df.loc[df[f] == "fleury d'aude (st pierre la mer)", f] = 'fleury d aude'
    df.loc[df[f] == "flimalle", f] = 'flemalle'
    df.loc[df[f] == "fontaine saint martin", f] = 'la fontaine saint martin'
    df.loc[df[f] == "fontaine sous bois", f] = 'fontenay sous bois'
    df.loc[df[f] == "fontaineblau", f] = 'fontainebleau'
    df.loc[df[f] == "fontenay   sous   bois", f] = 'fontenay sous bois'
    df.loc[df[f] == "fontenay le mormion", f] = 'fontenay le marmion'
    df.loc[df[f] == "fontenay les roses", f] = 'fontenay aux roses'
    df.loc[df[f] == "fontenay treisigny", f] = 'fontenay tresigny'
    df.loc[df[f] == "fontvielle", f] = 'fontvieille'
    df.loc[df[f] == "forcalquieret", f] = 'forcalqueiret'
    df.loc[df[f] == "foremoutiers", f] = 'faremoutiers'
    df.loc[df[f] == "fort calquier", f] = 'forcalquier'
    df.loc[df[f] == "fort mahon", f] = 'fort mahon plage'
    df.loc[df[f] == "fouqueux", f] = 'fourqueux'
    df.loc[df[f] == "fournes en weppe", f] = 'fournes en weppes'
    df.loc[df[f] == "franchevile", f] = 'francheville'
    df.loc[df[f] == "franconville la garenne (franconville)", f] = 'franconville'
    df.loc[df[f] == "fresnes en waive", f] = 'fresnes en wuerre'
    df.loc[df[f] == "friers faillouel", f] = 'frieres faillouel'
    df.loc[df[f] == "froncourt", f] = 'frontcourt'
    df.loc[df[f] == "fuveay", f] = 'fuveau'
    df.loc[df[f] == "gandies", f] = 'ganties'
    df.loc[df[f] == "gap (romette)", f] = 'gap'
    df.loc[df[f] == "gardanne (biver)", f] = 'biver'
    df.loc[df[f] == "garde colombe (eyguians)", f] = 'eyguians'
    df.loc[df[f] == "garge", f] = 'garges les gonesse'
    df.loc[df[f] == "garge les gonnesse", f] = 'garges les gonesse'
    df.loc[df[f] == "gattiere", f] = 'gattieres'
    df.loc[df[f] == "geis d'oloron", f] = 'geis d oloron'
    df.loc[df[f] == "geiswiller zoebersdorf (geiswiller)", f] = 'geiswiller'
    df.loc[df[f] == "gennevillers", f] = 'gennevilliers'
    df.loc[df[f] == "gennevilluiers", f] = 'gennevilliers'
    df.loc[df[f] == "germusson sur marne", f] = 'ormesson sur marne'
    df.loc[df[f] == "ghyvelde (les moeres)", f] = 'les moeres'
    df.loc[df[f] == "gignac la nerhte", f] = 'laure'
    df.loc[df[f] == "gignac la nerthe   laure", f] = 'laure'
    df.loc[df[f] == "gignac la nerthe (laure)", f] = 'laure'
    df.loc[df[f] == "gimbrett", f] = 'berstett'
    df.loc[df[f] == "givaumont", f] = 'giffaumont champaubert'
    df.loc[df[f] == "gommerville (grandville gaudreville)", f] = 'grandville gaudreville'
    df.loc[df[f] == "gonnesse", f] = 'gonesse'
    df.loc[df[f] == "gouesnac h", f] = 'gouesnach'
    df.loc[df[f] == "gouray sur marne", f] = 'gournay sur marne'
    df.loc[df[f] == "gousainville", f] = 'goussainville'
    df.loc[df[f] == "grand gevrier", f] = 'annecy'
    df.loc[df[f] == "grand port", f] = 'port le grand'
    df.loc[df[f] == "grandchamp des fontaines", f] = 'grandchamps des fontaines'
    df.loc[df[f] == "grande   synthe", f] = 'grande synthe'
    df.loc[df[f] == "grandeville", f] = 'granville'
    df.loc[df[f] == "granges aumontzey (granges sur vologne)", f] = 'granges sur vologne'
    df.loc[df[f] == "grasse (le plan de grasse)", f] = 'grasse'
    df.loc[df[f] == "grayant", f] = 'grayan et l hopital'
    df.loc[df[f] == "grendelbruck", f] = 'grendelbruch'
    df.loc[df[f] == "grenoble ", f] = 'grenoble'
    df.loc[df[f] == "gretz", f] = 'gretz armainvilliers'
    df.loc[df[f] == "grey", f] = 'gray'
    df.loc[df[f] == "grezieux la varenne", f] = 'grezieu la varenne'
    df.loc[df[f] == "griesheim", f] = 'griesheim pres molsheim'
    df.loc[df[f] == "groissy sur seine", f] = 'croissy sur seine'
    df.loc[df[f] == "grorlay", f] = 'groslay'
    df.loc[df[f] == "groslee saint benoit (st benoit)", f] = 'saint benois'
    df.loc[df[f] == "grouvieux", f] = 'gouvieux'
    df.loc[df[f] == "grundwiller", f] = 'grundviller'
    df.loc[df[f] == "guejouls", f] = 'cruejouls'
    df.loc[df[f] == "guemange", f] = 'guemnnge'
    df.loc[df[f] == "guemene penfao (guenouvry)", f] = 'guenouvry'
    df.loc[df[f] == "guenguat", f] = 'guengat'
    df.loc[df[f] == "guigneville sur essone", f] = 'guigneville sur essonne'
    df.loc[df[f] == "guilherand granger", f] = 'guilherand granges'
    df.loc[df[f] == "guilherand oranges", f] = 'guilherand granges'
    df.loc[df[f] == "guilleres", f] = 'les guilleres'
    df.loc[df[f] == "guillestne", f] = 'guillestre'
    df.loc[df[f] == "guipry messac", f] = 'guipry'
    df.loc[df[f] == "guipry messac (guipry)", f] = 'guipry'
    df.loc[df[f] == "guitinirers", f] = 'guitinieres'
    df.loc[df[f] == "guyancourt (bouviers)", f] = 'guyancourt'
    df.loc[df[f] == "guyancourt (villaroy)", f] = 'guyancourt'
    df.loc[df[f] == "guysse", f] = 'guysse'
    df.loc[df[f] == "haguenau (marienthal)", f] = 'haguenau'
    df.loc[df[f] == "haillan", f] = 'le haillan'
    df.loc[df[f] == "hannecourt sur escaut", f] = 'honnecourt sur escaut'
    df.loc[df[f] == "hardelot", f] = 'neufchatel hardelot'
    df.loc[df[f] == "hauchin", f] = 'haulchin'
    df.loc[df[f] == "haut maucq", f] = 'haut mauco'
    df.loc[df[f] == "hauteroche (granges sur baume)", f] = 'granges sur baume'
    df.loc[df[f] == "hauts de bienne (morez)", f] = 'morez'
    df.loc[df[f] == "hay les roses", f] = 'l hay les roses'
    df.loc[df[f] == "hayonge", f] = 'hayange'
    df.loc[df[f] == "hede bazouges", f] = 'bazouges sous hede'
    df.loc[df[f] == "hede bazouges (bazouges sous hede)", f] = 'bazouges sous hede'
    df.loc[df[f] == "helemmes", f] = 'hellemmes'
    df.loc[df[f] == "hemecourt", f] = 'homecourt'
    df.loc[df[f] == "herbignac (pompas)", f] = 'pompas'
    df.loc[df[f] == "heronville saint clair", f] = 'herouville saint clair'
    df.loc[df[f] == "herouville en vexin", f] = 'herouville saint clair'
    df.loc[df[f] == "hescamps (agnieres)", f] = 'agnieres'
    df.loc[df[f] == "hescamps (frettemolle)", f] = 'frettemolle'
    df.loc[df[f] == "hillion (st rene hillion)", f] = 'saint rene hillion'
    df.loc[df[f] == "hiroson", f] = 'hirson'
    df.loc[df[f] == "hochfelden (schaffhouse sur zorn)", f] = 'schaffhouse sur zorn'
    df.loc[df[f] == "hoffen (leiterswiller)", f] = 'leiterswiller'
    df.loc[df[f] == "hollivillers", f] = 'hallivillers"'
    df.loc[df[f] == "horbourg whir", f] = 'horbourg wihr'
    df.loc[df[f] == "hospitalet", f] = 'l hospitalet'
    df.loc[df[f] == "hossgor", f] = 'soorts hossegor'
    df.loc[df[f] == "humes jorquenay (jorquenay)", f] = 'jorquenay'
    df.loc[df[f] == "huy en joses", f] = 'jouy en josas'
    df.loc[df[f] == "hypercourt (pertain)", f] = 'pertain'
    df.loc[df[f] == "ile  d'aix", f] = 'ile d aix'
    df.loc[df[f] == "illeville", f] = 'illeville sur montfort'
    df.loc[df[f] == "illtal (oberdorf)", f] = 'oberdorf'
    df.loc[df[f] == "irmstett", f] = 'scharrachbergheim'
    df.loc[df[f] == "isbergues (berguette)", f] = 'berguette'
    df.loc[df[f] == "isigny le buat (le mesnil thebault)", f] = 'le mesnil thebault'
    df.loc[df[f] == "isle adam", f] = 'l isle adam'
    df.loc[df[f] == "isles sur sorgues", f] = 'l isle sur la sorgue'
    df.loc[df[f] == "istres (entressen)", f] = 'istres'
    df.loc[df[f] == "iversny", f] = 'iverny'
    df.loc[df[f] == "ivry  sur seine", f] = 'ivry sur seine'
    df.loc[df[f] == "ivry gargand", f] = 'livry gargand'
    df.loc[df[f] == "ivry sur sur sur seine", f] = 'ivry sur seine'
    df.loc[df[f] == "ivrysur seine", f] = 'ivry sur seine'
    df.loc[df[f] == "iyon", f] = 'lyon'
    df.loc[df[f] == "izel les hameau", f] = 'izel les hameaux'
    df.loc[df[f] == "jarville la halgrange", f] = 'jarville la malgrange'
    df.loc[df[f] == "jaunay marigny", f] = 'marigny brizay'
    df.loc[df[f] == "jaunay marigny (marigny brizay)", f] = 'marigny brizay'
    df.loc[df[f] == "jonage ()", f] = 'jonac'
    df.loc[df[f] == "joncquieres", f] = 'jonquieres'
    df.loc[df[f] == "jouy le mautier", f] = 'jouy le mautier'
    df.loc[df[f] == "jugon", f] = 'jugon les lacs'
    df.loc[df[f] == "juvigny val d'andaine (sept forges)", f] = 'sept forges'
    df.loc[df[f] == "juvisy", f] = 'juvisy sur orge'
    df.loc[df[f] == "kaysersberg vignoble (kaysersberg)", f] = 'kaysersberg'
    df.loc[df[f] == "kaysersberg vignoble (kientzheim)", f] = 'kientzheim'
    df.loc[df[f] == "l abergement ste colombe", f] = 'l abergement sainte colombe'
    df.loc[df[f] == "l arbret", f] = 'bavincourt'
    df.loc[df[f] == "l haij les roses", f] = 'l hay les roses'
    df.loc[df[f] == "l hay les roges", f] = 'l hay les roses'
    df.loc[df[f] == "l hopital camfrout", f] = 'hopital camfrout'
    df.loc[df[f] == "l iles d'abeau", f] = 'l isle d abeau'
    df.loc[df[f] == "l isle en rigault", f] = 'lisle en rigault'
    df.loc[df[f] == "l isle sur tarn", f] = 'lisle sur tarn'
    df.loc[df[f] == "l&#;aigle", f] = 'l aigle'
    df.loc[df[f] == "la  ciotat", f] = 'la ciotat'
    df.loc[df[f] == "la baronnie (garencieres)", f] = 'garencieres'
    df.loc[df[f] == "la bastide de l'eveque", f] = 'la bastide de l eveque'
    df.loc[df[f] == "la baule (la baule escoublac)", f] = 'la baule escoublac'
    df.loc[df[f] == "la bouilladise", f] = 'la bouilladisse'
    df.loc[df[f] == "la chapelle au bareil", f] = 'la chapelle aubareil'
    df.loc[df[f] == "la chapelle basse sur mer", f] = 'la chapelle basse mer'
    df.loc[df[f] == "la chapelle d&#;armentieres", f] = 'la chapelle d armentieres'
    df.loc[df[f] == "la chapelle du fest", f] = 'saint amand'
    df.loc[df[f] == "la chapelle longueville", f] = 'saint just'
    df.loc[df[f] == "la chapelle saint pierre", f] = 'lachapelle saint pierre'
    df.loc[df[f] == "la charbonniere", f] = 'charbonnieres'
    df.loc[df[f] == "la chatelaine", f] = 'lons le saunier'
    df.loc[df[f] == "la ciotat   ()", f] = 'la ciotat'
    df.loc[df[f] == "la coix valmer", f] = 'la croix valmer'
    df.loc[df[f] == "la colle les bordes", f] = 'la celle les bordes'
    df.loc[df[f] == "la crau (la moutonne)", f] = 'la moutonne'
    df.loc[df[f] == "la croix saint lefroid", f] = 'la croix saint leufroy'
    df.loc[df[f] == "la croix saint ouen", f] = 'lacroix saint ouen'
    df.loc[df[f] == "la farled", f] = 'la farlede'
    df.loc[df[f] == "la ferte en ouche (gauville)", f] = 'gauville'
    df.loc[df[f] == "la ferte en ouche (la ferte frenel)", f] = 'la ferte frenel'
    df.loc[df[f] == "la flotte en re", f] = 'la flotte'
    df.loc[df[f] == "la flotte sur mer", f] = 'la flotte'
    df.loc[df[f] == "la foret ste croix", f] = 'la foret sainte croix'
    df.loc[df[f] == "la garenne colombe", f] = 'la garenne colombes'
    df.loc[df[f] == "la grand combes", f] = 'la grand combe'
    df.loc[df[f] == "la grande croix", f] = 'la grand croix"'
    df.loc[df[f] == "la hague (urville nacqueville)", f] = 'urville nacqueville'
    df.loc[df[f] == "la haye fouassiere", f] = 'la haie fouassiere"'
    df.loc[df[f] == "la lande d'airau", f] = 'la lande d airau'
    df.loc[df[f] == "la londe des maures", f] = 'la lande des maures'
    df.loc[df[f] == "la loye", f] = 'mont sous vaudrey'
    df.loc[df[f] == "la mans", f] = 'le mans'
    df.loc[df[f] == "la milene", f] = 'la malene'
    df.loc[df[f] == "la motte en provence", f] = 'la motte'
    df.loc[df[f] == "la mure d'isere", f] = 'la mure'
    df.loc[df[f] == "la penne sur huveaune (bastidonne)", f] = 'bastidonne'
    df.loc[df[f] == "la peyrouse mornay", f] = 'lapeyrouse mornay'
    df.loc[df[f] == "la pointe de blausasc", f] = 'blausasc'
    df.loc[df[f] == "la roche", f] = 'la roche sur yon'
    df.loc[df[f] == "la roche jaudy (pommerit jaudy)", f] = 'pommerit jaudy'
    df.loc[df[f] == "la roche sue yon", f] = 'la roche sur yon'
    df.loc[df[f] == "la seyne sur mer (les sablettes)", f] = 'la seyne sur mer'
    df.loc[df[f] == "la test de buche", f] = 'la teste de buch'
    df.loc[df[f] == "la teste de buch (cazaux)", f] = 'la teste de buch'
    df.loc[df[f] == "la teste de buch (pyla plage)", f] = 'la teste de buch'
    df.loc[df[f] == "la teste de buch (pyla sur mer)", f] = 'la teste de buch'
    df.loc[df[f] == "la tremblade (ronce les bains)", f] = 'ronce les bains'
    df.loc[df[f] == "la trubie", f] = 'la turbie'
    df.loc[df[f] == "la ville du buis", f] = 'buis les baronnies'
    df.loc[df[f] == "la vineuse sur fregande", f] = 'la vineuse'
    df.loc[df[f] == "la voge les bains (hautmougey)", f] = 'hautmougey'
    df.loc[df[f] == "la voulte sur rhone", f] = 'privas'
    df.loc[df[f] == "labarte sur leze", f] = 'labarthe sur leze'
    df.loc[df[f] == "labasidette", f] = 'labastidette'
    df.loc[df[f] == "labaslide gabausse", f] = 'labastide gabausse'
    df.loc[df[f] == "labastide cezerac", f] = 'labastide cezeracq'
    df.loc[df[f] == "labergement ste marie", f] = 'labergement sainte marie'
    df.loc[df[f] == "lacanau ocean", f] = 'lacanau'
    df.loc[df[f] == "lachaussee (haumont les lachaussee)", f] = 'haumont les lachaussee'
    df.loc[df[f] == "laferte st. aubin", f] = 'la ferte saint aubin'
    df.loc[df[f] == "lafitte vigordame", f] = 'lafitte vigordane'
    df.loc[df[f] == "lagard", f] = 'la garde'
    df.loc[df[f] == "lagneville", f] = 'laigneville'
    df.loc[df[f] == "laissac severac l'eglise", f] = 'laissac'
    df.loc[df[f] == "laissac severac l'eglise (laissac)", f] = 'laissac'
    df.loc[df[f] == "lamarlaye", f] = 'lamorlaye'
    df.loc[df[f] == "lamballe (la poterie)", f] = 'lamballe'
    df.loc[df[f] == "lamballe (meslin)", f] = 'lamballe'
    df.loc[df[f] == "lamballe armor", f] = 'lamballe'
    df.loc[df[f] == "lamballe armor (lamballe)", f] = 'lamballe'
    df.loc[df[f] == "lamballe armor (maroue)", f] = 'lamballe'
    df.loc[df[f] == "lamballe armor (meslin)", f] = 'lamballe'
    df.loc[df[f] == "lamorville (lavigneville)", f] = 'lavigneville'
    df.loc[df[f] == "lamothe labderron", f] = 'lamothe landerron'
    df.loc[df[f] == "lampersheim", f] = 'lampertheim'
    df.loc[df[f] == "lancon de provnce", f] = 'lancon de provrnce",'
    df.loc[df[f] == "laneuveville devant nancy (la madeleine)", f] = 'la madeleine'
    df.loc[df[f] == "laneuville en saulnois", f] = 'laneuveville en saulnois'
    df.loc[df[f] == "lanvallay (tressaint)", f] = 'tressaint'
    df.loc[df[f] == "lapeyrousse fossat", f] = 'lapeyrouse fossat'
    df.loc[df[f] == "lapugnois", f] = 'lapugnoy'
    df.loc[df[f] == "lardin saint lazare", f] = 'le lardin saint lazare'
    df.loc[df[f] == "latremblade", f] = 'la tremblade'
    df.loc[df[f] == "laudun", f] = 'laudun l ardoise'
    df.loc[df[f] == "laure minervais", f] = 'laure minervois'
    df.loc[df[f] == "lavernose lacase", f] = 'lavernose lacasse'
    df.loc[df[f] == "le ban saitn martin", f] = 'le ban saint martin'
    df.loc[df[f] == "le bas segala (la bastide l'eveque)", f] = 'la bastide l eveque'
    df.loc[df[f] == "le blanc menil", f] = 'le blanc mesnil'
    df.loc[df[f] == "le bois doingt", f] = 'le bois d oingt'
    df.loc[df[f] == "le bon saint martin", f] = 'le ban saint martin'
    df.loc[df[f] == "le bosc du theil (st nicolas du bosc)", f] = 'saint nicolas du bosc'
    df.loc[df[f] == "le buisson de cadouin (cadouin)", f] = 'cadouin'
    df.loc[df[f] == "le cannet (rocheville)", f] = 'le cannet'
    df.loc[df[f] == "le cannet rochevil", f] = 'le cannet'
    df.loc[df[f] == "le canon", f] = 'lege cap ferret'
    df.loc[df[f] == "le cap ferret", f] = 'lege cap ferret'
    df.loc[df[f] == "le castellet (ste anne du castellet)", f] = 'sainte anne du castellet'
    df.loc[df[f] == "le catteau", f] = 'le coteau'
    df.loc[df[f] == "le champs pres froges", f] = 'le champ pres froges'
    df.loc[df[f] == "le chateley", f] = 'bletterans'
    df.loc[df[f] == "le chenay", f] = 'le chesnay'
    df.loc[df[f] == "le cordonnois", f] = 'le cardonnois'
    df.loc[df[f] == "le grand fougeretz", f] = 'le grand fougeray'
    df.loc[df[f] == "le hom (curcy sur orne)", f] = 'curcy sur orne'
    df.loc[df[f] == "le home varaville", f] = 'varaville'
    df.loc[df[f] == "le lardin", f] = 'le lardin saint lazare'
    df.loc[df[f] == "le louverot", f] = 'poligny'
    df.loc[df[f] == "le luc en provence", f] = 'le luc'
    df.loc[df[f] == "le malesherbois (malesherbes)", f] = 'malesherbes'
    df.loc[df[f] == "le mene (le gouray)", f] = 'le gouray'
    df.loc[df[f] == "le mene (plessala)", f] = 'plessala'
    df.loc[df[f] == "le mesnil sur ogier", f] = 'le mesnil sur oger'
    df.loc[df[f] == "le mets sur seine", f] = 'le mee sur seine'
    df.loc[df[f] == "le mornac", f] = 'mornac sur seudre'
    df.loc[df[f] == "le mottier", f] = 'mottier'
    df.loc[df[f] == "le nourion en thieiache", f] = 'le nouvion en thierache'
    df.loc[df[f] == "le passage d&#;agen", f] = 'la passage d agen'
    df.loc[df[f] == "le plessis robinson (robinson)", f] = 'le plessis robinson'
    df.loc[df[f] == "le plessis robinsson", f] = 'le plessis robinson'
    df.loc[df[f] == "le plessis trevisse", f] = 'le plessis trevise'
    df.loc[df[f] == "le pres saint gervais", f] = 'le pre saint gervais'
    df.loc[df[f] == "le puy ste reparade", f] = 'le puy sainte reparade'
    df.loc[df[f] == "le relecq kerhuan", f] = 'relecq kerhuon'
    df.loc[df[f] == "le revest", f] = 'le revest les eaux'
    df.loc[df[f] == "le teil", f] = 'le teil d ardeche'
    df.loc[df[f] == "le touquet", f] = 'le touquet paris plage'
    df.loc[df[f] == "le val d'hazey", f] = 'aubevoye'
    df.loc[df[f] == "le val d'hazey (aubevoye)", f] = 'aubevoye'
    df.loc[df[f] == "lecleuves", f] = 'mecleuves'
    df.loc[df[f] == "lege cap ferret (cap ferret)", f] = 'lege cap ferret'
    df.loc[df[f] == "legrand pressigny", f] = 'le grand pressigny'
    df.loc[df[f] == "lendou en quercy (lascabanes)", f] = 'lascabanes'
    df.loc[df[f] == "lepin", f] = 'le pin'
    df.loc[df[f] == "lerheu", f] = 'le rheu'
    df.loc[df[f] == "les abrets en dauphine (fitilieu)", f] = 'fitilieu'
    df.loc[df[f] == "les achards (la mothe achard)", f] = 'la mothe achard'
    df.loc[df[f] == "les adrets de l&amp;amp;amp;#;estere", f] = 'les adrets de l estere'
    df.loc[df[f] == "les arc", f] = 'bourg saint maurice'
    df.loc[df[f] == "les arcs s sur  argens", f] = 'les arcs'
    df.loc[df[f] == "les arcs s sur argens", f] = 'les arcs'
    df.loc[df[f] == "les arcs sur argens", f] = 'les arcs'
    df.loc[df[f] == "les ardrets", f] = 'les adrets'
    df.loc[df[f] == "les ardrets de l'esterel", f] = 'les adrets'
    df.loc[df[f] == "les auxons", f] = 'auxon dessous'
    df.loc[df[f] == "les auxons (auxon dessous)", f] = 'auxon dessous'
    df.loc[df[f] == "les avenieres veyrins thuellin", f] = 'les avenieres'
    df.loc[df[f] == "les belleville (val thorens)", f] = 'val thorens'
    df.loc[df[f] == "les clayes", f] = 'les clayes sous bois'
    df.loc[df[f] == "les coteaux perigourdins (chavagnac)", f] = 'chavagnac'
    df.loc[df[f] == "les deux alpes (venosc les deux alpes)", f] = 'venosc les deux alpes'
    df.loc[df[f] == "les echets", f] = 'miribel'
    df.loc[df[f] == "les eglisottes", f] = 'les eglisottes et chalaures'
    df.loc[df[f] == "les essorts le roi", f] = 'les essarts le roi'
    df.loc[df[f] == "les eyzies", f] = 'les eyzies de tayac sireuil'
    df.loc[df[f] == "les hauts d'anjou (champigne)", f] = 'champigne'
    df.loc[df[f] == "les levees et thoumeyraques", f] = 'les leves et thoumeyragues'
    df.loc[df[f] == "les mautiers", f] = 'les moutiers en retz'
    df.loc[df[f] == "les milles", f] = 'aix en provence'
    df.loc[df[f] == "les monts d'aunay", f] = 'aunay sur odon'
    df.loc[df[f] == "les monts d'aunay (aunay sur odon)", f] = 'aunay sur odon'
    df.loc[df[f] == "les moulins (plemet)", f] = 'plemet'
    df.loc[df[f] == "les pechs du vers (st cernin)", f] = 'saint cernin'
    df.loc[df[f] == "les pennes mirabeau (la gavotte)", f] = 'la gavotte'
    df.loc[df[f] == "les portes du coglais", f] = 'cogles'
    df.loc[df[f] == "les portes du coglais (cogles)", f] = 'cogles'
    df.loc[df[f] == "les roquetes", f] = 'les roquettes'
    df.loc[df[f] == "les rousses", f] = 'morez'
    df.loc[df[f] == "les sables d olonne le chateau d'olonne", f] = 'le chateau d olonne'
    df.loc[df[f] == "les septvallons", f] = 'longueval barbonval'
    df.loc[df[f] == "les sorinnieres", f] = 'les sorinieres'
    df.loc[df[f] == "les vans", f] = 'largentiere'
    df.loc[df[f] == "lescarene", f] = 'l escarene'
    df.loc[df[f] == "lesmontils", f] = 'les montils'
    df.loc[df[f] == "leu vil sur orge", f] = 'leuville sur orge'
    df.loc[df[f] == "levallo perret", f] = 'levallois perret'
    df.loc[df[f] == "librac", f] = 'pibrac'
    df.loc[df[f] == "lieuran   les   beziers", f] = 'lieuran les beziers'
    df.loc[df[f] == "limeil", f] = 'limeil brevannes'
    df.loc[df[f] == "limoges (landouge)", f] = 'limoges'
    df.loc[df[f] == "limzil brevannes", f] = 'limeil brevannes'
    df.loc[df[f] == "listrac", f] = 'listrac medoc'
    df.loc[df[f] == "livron sur rhone", f] = 'livron sur drome'
    df.loc[df[f] == "livy gargan", f] = 'livry gargan'
    df.loc[df[f] == "logny sur marne", f] = 'lagny sur marne'
    df.loc[df[f] == "loire authion", f] = 'andard'
    df.loc[df[f] == "loire authion (andard)", f] = 'andard'
    df.loc[df[f] == "loire authion (la bohalle)", f] = 'la bohalle'
    df.loc[df[f] == "loire authion (la dagueniere)", f] = 'la dagueniere'
    df.loc[df[f] == "loireauxence (varades)", f] = 'varades'
    df.loc[df[f] == "loiron ruille (loiron)", f] = 'loiron'
    df.loc[df[f] == "loix en re", f] = 'loix'
    df.loc[df[f] == "lomesnil saint denis", f] = 'le mesnil saint denis'
    df.loc[df[f] == "lomothe capdeville", f] = 'lamothe capdeville'
    df.loc[df[f] == "longny les villages", f] = 'longny au perche'
    df.loc[df[f] == "longuenee en anjou (la meignanne)", f] = 'la meignanne'
    df.loc[df[f] == "longuyon (noers)", f] = 'noers'
    df.loc[df[f] == "loroux bottereau", f] = 'vallet'
    df.loc[df[f] == "lorp", f] = 'lorp sentaraille'
    df.loc[df[f] == "lourdion ichere", f] = 'lourdios ichere'
    df.loc[df[f] == "luc  sur mer", f] = 'luc sur mer'
    df.loc[df[f] == "luc la primaube", f] = 'la primaube'
    df.loc[df[f] == "luc la primaube (la primaube)", f] = 'la primaube'
    df.loc[df[f] == "luchassagne", f] = 'lachassagne'
    df.loc[df[f] == "luchon", f] = 'bagneres de luchon'
    df.loc[df[f] == "lucon et l'ile du canoy", f] = 'lucon et l ile du canoy'
    df.loc[df[f] == "lussy sou dun", f] = 'mussy sous dun'
    df.loc[df[f] == "machecoul saint meme", f] = 'machecoul'
    df.loc[df[f] == "machecoul saint meme (machecoul)", f] = 'machecoul'
    df.loc[df[f] == "machecoul saint meme (st meme le tenu)", f] = 'machecoul'
    df.loc[df[f] == "macon (loche)", f] = 'loche'
    df.loc[df[f] == "maen roch (st brice en cogles)", f] = 'saint brice en cogles'
    df.loc[df[f] == "maen roch (st etienne en cogles)", f] = 'saint etienne en cogles'
    df.loc[df[f] == "magay la campagne", f] = 'magny la campagne'
    df.loc[df[f] == "magland (flaine)", f] = 'flaine'
    df.loc[df[f] == "maidiere", f] = 'maidieres'
    df.loc[df[f] == "maignelay", f] = 'maignelay montigny'
    df.loc[df[f] == "maisdon", f] = 'maisdon sur sevre'
    df.loc[df[f] == "maison  laffitte", f] = 'maison laffitte'
    df.loc[df[f] == "maison laffite", f] = 'maison laffitte'
    df.loc[df[f] == "maisons  alfort", f] = 'maisons alfort'
    df.loc[df[f] == "maisons laffite", f] = 'maison laffitte'
    df.loc[df[f] == "maiziere les metz", f] = 'maizieres les metz'
    df.loc[df[f] == "malmaison", f] = 'rueil malmaison'
    df.loc[df[f] == "malo les bains", f] = 'dunkerke'
    df.loc[df[f] == "manligny les metz", f] = ''
    df.loc[df[f] == "manlleu", f] = 'montigny les metz'
    df.loc[df[f] == "mans", f] = 'le mans'
    df.loc[df[f] == "mante la ville", f] = 'mantes la ville'
    df.loc[df[f] == "manteau fault yanne", f] = 'montereau fault yonne'
    df.loc[df[f] == "manzieres les metz", f] = 'maizieres les metz'
    df.loc[df[f] == "marange", f] = 'maranges'
    df.loc[df[f] == "marcq en barouel", f] = '"marcq en baroeul'
    df.loc[df[f] == "marcy l' etoile", f] = 'marcy l etoile'
    df.loc[df[f] == "marennes hiers brouage (marennes)", f] = 'marennes'
    df.loc[df[f] == "mareuil en perigord (mareuil)", f] = 'mareuil'
    df.loc[df[f] == "margaux cantenac (cantenac)", f] = 'cantenac'
    df.loc[df[f] == "margaux cantenac (margaux)", f] = 'margaux'
    df.loc[df[f] == "marignanne", f] = 'marignanne'
    df.loc[df[f] == "marigny le lozon", f] = 'marigny'
    df.loc[df[f] == "marly   le   roy", f] = 'marly le roy'
    df.loc[df[f] == "marly la valle", f] = 'marly la vallee'
    df.loc[df[f] == "marolle sur seine", f] = 'marolles sur seine'
    df.loc[df[f] == "marquette", f] = 'marquette lez lille'
    df.loc[df[f] == "marseiille", f] = 'marseille'
    df.loc[df[f] == "marseile", f] = 'marseille'
    df.loc[df[f] == "marseile", f] = 'marseille'
    df.loc[df[f] == "marseill", f] = 'marseille'
    df.loc[df[f] == "marseille (callelongue)", f] = 'marseille'
    df.loc[df[f] == "marseille (la valentine)", f] = 'marseille'
    df.loc[df[f] == "marseille (les olives)", f] = 'marseille'
    df.loc[df[f] == "marselle", f] = 'marseille'
    df.loc[df[f] == "martes tolosane", f] = 'martres tolosane'
    df.loc[df[f] == "martignas", f] = 'martignas sur jalle'
    df.loc[df[f] == "martignas sur jalles", f] = 'martignas sur jalle'
    df.loc[df[f] == "martigues (la couronne carro)", f] = 'martigues'
    df.loc[df[f] == "maseille", f] = 'marseille'
    df.loc[df[f] == "masevaux niederbruck (masevaux)", f] = 'masevaux'
    df.loc[df[f] == "masevaux niederbruck (niederbruck)", f] = 'niederbruck'
    df.loc[df[f] == "mauges sur loire (st florent le vieil)", f] = 'saint florent le vieil'
    df.loc[df[f] == "mauguio (carnon plage)", f] = 'carnon plage'
    df.loc[df[f] == "mauleon (la chapelle largeau)", f] = 'la chapelle largeau'
    df.loc[df[f] == "mauleon (st aubin de baubigne)", f] = 'saint aubin de baubigne'
    df.loc[df[f] == "mauleon soule", f] = 'mauleon licharre'
    df.loc[df[f] == "mauzac et grand castaney", f] = 'mauzac et grand castang'
    df.loc[df[f] == "maxeville (maxeville champleboeuf)", f] = 'maxeville'
    df.loc[df[f] == "medon", f] = 'meudon'
    df.loc[df[f] == "menthon", f] = 'menton'
    df.loc[df[f] == "mercurol veaunes (mercurol)", f] = 'mercurol'
    df.loc[df[f] == "meriganc", f] = 'merignac'
    df.loc[df[f] == "merinldol", f] = 'merindol'
    df.loc[df[f] == "mersheim", f] = 'gladbach'
    df.loc[df[f] == "mery bissieres en auge (mery corbon)", f] = 'mery corbon'
    df.loc[df[f] == "meschers", f] = 'meschers sur gironde'
    df.loc[df[f] == "mesnil en ouche", f] = 'beaumesnil'
    df.loc[df[f] == "mesnil le roi", f] = 'le mesnil le roi'
    df.loc[df[f] == "metaires saint quirin", f] = 'metairies saint quirin'
    df.loc[df[f] == "metrich", f] = 'koenigsmacker'
    df.loc[df[f] == "metz ", f] = 'metz'
    df.loc[df[f] == "meyssiez", f] = 'meyssies'
    df.loc[df[f] == "mezidon vallee d'auge (magny le freule)", f] = 'magny le freule'
    df.loc[df[f] == "mezidon vallee d'auge (mezidon canon)", f] = 'mezidon canon'
    df.loc[df[f] == "mezieux", f] = 'meyzieu'
    df.loc[df[f] == "migrovillard", f] = 'mignovillard'
    df.loc[df[f] == "milizac guipronvel (guipronvel)", f] = 'guipronvel'
    df.loc[df[f] == "milizac guipronvel (milizac)", f] = 'milizac'
    df.loc[df[f] == "millaux", f] = 'millau'
    df.loc[df[f] == "mirandol bourgnaurac", f] = 'mirandol bourgnounac'
    df.loc[df[f] == "miraval cabardis", f] = 'miraval cabardes'
    df.loc[df[f] == "missyllac", f] = 'missillac'
    df.loc[df[f] == "mitry", f] = 'mitry mory'
    df.loc[df[f] == "mittainvilliers verigny", f] = 'mittainvilliers'
    df.loc[df[f] == "moeurs", f] = 'moeurs verdey'
    df.loc[df[f] == "moirieres les avignon", f] = 'morieres les avignon'
    df.loc[df[f] == "mon de marsant", f] = 'mont de marsan'
    df.loc[df[f] == "moncetz", f] = 'moncetz longevas'
    df.loc[df[f] == "monchaux", f] = 'monchaux sur ecaillon'
    df.loc[df[f] == "moncontour (ouzilly vignolles)", f] = 'moncontour'
    df.loc[df[f] == "moncontour de bretagne", f] = 'moncontour'
    df.loc[df[f] == "moncourt fromonville", f] = 'montcourt fromonville'
    df.loc[df[f] == "monfort l'amaury", f] = 'montfort l amaury'
    df.loc[df[f] == "mont bonvillers", f] = 'briey'
    df.loc[df[f] == "mont de lans (l alpe de mont de lans)", f] = 'mont de lans'
    df.loc[df[f] == "mont rabe", f] = 'montrabe'
    df.loc[df[f] == "mont saint aignant", f] = 'mont saint aignan'
    df.loc[df[f] == "montageron", f] = 'montgeron'
    df.loc[df[f] == "montagne sur gironde", f] = 'mortagne sur gironde'
    df.loc[df[f] == "montaigu vendee", f] = 'la guyonniere'
    df.loc[df[f] == "montaigu vendee (la guyonniere)", f] = 'la guyonniere'
    df.loc[df[f] == "montaren", f] = 'montaren et saint mediers'
    df.loc[df[f] == "montbardier", f] = 'montbartier'
    df.loc[df[f] == "montbonnot", f] = 'montbonnot saint martin'
    df.loc[df[f] == "montdiidier", f] = 'montdidier'
    df.loc[df[f] == "montefontaine en thelle", f] = 'mortefontaine en thelle'
    df.loc[df[f] == "montegnon", f] = 'montgeron'
    df.loc[df[f] == "monteraux", f] = 'montereau'
    df.loc[df[f] == "monterson", f] = 'montesson'
    df.loc[df[f] == "montfernier", f] = 'montfermier'
    df.loc[df[f] == "montford sur argens", f] = 'montfort sur argens'
    df.loc[df[f] == "montfort  sur meu", f] = 'montfort sur meu'
    df.loc[df[f] == "montfort sur baulzane", f] = 'montfort sur boulzane'
    df.loc[df[f] == "montgrblanc", f] = 'monterblanc'
    df.loc[df[f] == "monthery", f] = 'montlhery'
    df.loc[df[f] == "montigny le breteneux", f] = 'montigny le bretonneux'
    df.loc[df[f] == "montigny le bretonneaux", f] = 'montigny le bretonneux'
    df.loc[df[f] == "montigny le bxt", f] = 'montigny le bretonneux'
    df.loc[df[f] == "montigny saint christophe", f] = 'montignies saint christophe"'
    df.loc[df[f] == "montlouis sur louis", f] = 'montlouis'
    df.loc[df[f] == "montmartier", f] = 'montmartin'
    df.loc[df[f] == "montmerle", f] = 'montmerle sur saone'
    df.loc[df[f] == "montoy flonville", f] = 'montoy flanville'
    df.loc[df[f] == "montpon", f] = 'montpon menesterol'
    df.loc[df[f] == "montpon menesterole", f] = 'montpon menesterol'
    df.loc[df[f] == "montrerau fault yion", f] = 'montereau fault yonne'
    df.loc[df[f] == "montreuil sur mer", f] = 'montreuil'
    df.loc[df[f] == "montrevault sur evre (la chaussaire)", f] = 'la chaussaire'
    df.loc[df[f] == "montreverd (mormaison)", f] = 'mormaison'
    df.loc[df[f] == "montreverd (st andre treize voies)", f] = 'saint andre treize voies'
    df.loc[df[f] == "montsenelle (pretot ste suzanne)", f] = 'pretot sainte suzanne'
    df.loc[df[f] == "mony de marsan", f] = 'mont de marsan'
    df.loc[df[f] == "morcenx la nouvelle (garrosse)", f] = 'garrosse'
    df.loc[df[f] == "moret loing et orvanne", f] = 'moret sur loing'
    df.loc[df[f] == "moret loing et orvanne (ecuelles)", f] = 'ecuelles'
    df.loc[df[f] == "moret loing et orvanne (episy)", f] = 'episy'
    df.loc[df[f] == "moret loing et orvanne (moret sur loing)", f] = 'moret sur loing'
    df.loc[df[f] == "moret loiny", f] = 'moret sur loing'
    df.loc[df[f] == "morieres", f] = 'morieres les avignon'
    df.loc[df[f] == "morlenheim", f] = 'marlenheim'
    df.loc[df[f] == "morly le roi", f] = 'marly le roi'
    df.loc[df[f] == "mornan", f] = 'mornans'
    df.loc[df[f] == "mortain bocage (mortain)", f] = 'mortain'
    df.loc[df[f] == "motry mory", f] = 'mitry mory'
    df.loc[df[f] == "moucourt", f] = 'mourcourt'
    df.loc[df[f] == "mougon thorigne (mougon)", f] = 'mougon'
    df.loc[df[f] == "mouliders", f] = 'moulidars'
    df.loc[df[f] == "moulin les metz", f] = 'moulins les metz'
    df.loc[df[f] == "moulins en bessin (martragny)", f] = 'martragny'
    df.loc[df[f] == "moult chicheboville (moult)", f] = 'moult'
    df.loc[df[f] == "moutiers tarentaise", f] = 'moutiers'
    df.loc[df[f] == "moyeure grande", f] = 'moyeuvre grande'
    df.loc[df[f] == "moyeuvre grand", f] = 'moyeuvre grande'
    df.loc[df[f] == "nantes ", f] = 'nantes'
    df.loc[df[f] == "nanteuil le haudoin", f] = 'nanteuil le haudouin'
    df.loc[df[f] == "nassandres sur risle (nassandres)", f] = 'nassandres'
    df.loc[df[f] == "naussac fontanes (fontanes)", f] = 'fontanes'
    df.loc[df[f] == "neauphile le vieux", f] = 'neauphle le vieux'
    df.loc[df[f] == "nemilly plaisance", f] = 'neuilly plaisance'
    df.loc[df[f] == "neuilly plaisanee", f] = 'neuilly plaisance'
    df.loc[df[f] == "neuilly sur  seine", f] = 'neuilly sur seine'
    df.loc[df[f] == "neuville sur escault", f] = 'neuville sur escaut'
    df.loc[df[f] == "nice (st roman de bellet)", f] = 'nice'
    df.loc[df[f] == "nielles les andres", f] = 'nielles les ardres'
    df.loc[df[f] == "nierderbron les bains", f] = 'nierderbronn les bains'
    df.loc[df[f] == "nissan les enserune", f] = 'nissan lez enserune'
    df.loc[df[f] == "nogent l'artoud", f] = 'nogent l artaud'
    df.loc[df[f] == "noisy le so", f] = 'noisy le sec'
    df.loc[df[f] == "norbhouse", f] = 'nordhouse'
    df.loc[df[f] == "notre dame de sanihlac", f] = 'notre dame de sanilhac'
    df.loc[df[f] == "noues de sienne (courson)", f] = 'courson'
    df.loc[df[f] == "noues de sienne (mesnil clinchamps)", f] = 'mesnil clinchamps'
    df.loc[df[f] == "nouzanville", f] = 'nouzonville'
    df.loc[df[f] == "noyelle godault", f] = 'noyelles godault'
    df.loc[df[f] == "noyers missy (noyers bocage)", f] = 'noyers bocage'
    df.loc[df[f] == "nueil les aubiers (les aubiers)", f] = 'les aubiers'
    df.loc[df[f] == "obatilly", f] = 'batilly'
    df.loc[df[f] == "octeville cherbourg", f] = 'cherbourg en cotentin'
    df.loc[df[f] == "ogy montoy flanville (montoy flanville)", f] = 'montoy flanville'
    df.loc[df[f] == "ogy montoy flanville (ogy)", f] = 'ogy'
    df.loc[df[f] == "ohlungen (keffendorf)", f] = 'keffendorf'
    df.loc[df[f] == "ohrwiller", f] = 'rohrwiller'
    df.loc[df[f] == "ombree d'anjou (le tremblay)", f] = 'le tremblay'
    df.loc[df[f] == "onesse laharie", f] = 'onesse et laharie'
    df.loc[df[f] == "onet e chateau", f] = 'onet le chateau'
    df.loc[df[f] == "oray la ville", f] = 'orry la ville'
    df.loc[df[f] == "oree d'anjou (bouzille)", f] = 'bouzille'
    df.loc[df[f] == "oree d'anjou (champtoceaux)", f] = 'champtoceaux'
    df.loc[df[f] == "oree d'anjou (drain)", f] = 'drain'
    df.loc[df[f] == "oree d'anjou (la varenne)", f] = 'la varenne'
    df.loc[df[f] == "oree d'anjou (landemont)", f] = 'landemont'
    df.loc[df[f] == "oree d'anjou (lire)", f] = 'lire'
    df.loc[df[f] == "oree d'anjou (st laurent des autels)", f] = 'saint laurent des autels'
    df.loc[df[f] == "orrylaville", f] = 'orry la ville'
    df.loc[df[f] == "orval sur sienne", f] = 'orval'
    df.loc[df[f] == "orval sur sienne (orval)", f] = 'orval'
    df.loc[df[f] == "pagny", f] = 'pagny la ville'
    df.loc[df[f] == "palaisseau", f] = 'palaiseau'
    df.loc[df[f] == "palissieres", f] = 'panissieres'
    df.loc[df[f] == "panthierry", f] = 'ponthierry'
    df.loc[df[f] == "pareboscq", f] = 'parleboscq'
    df.loc[df[f] == "parenties en born", f] = 'parentis en born'
    df.loc[df[f] == "parus", f] = 'paris'
    df.loc[df[f] == "passy (chedde)", f] = 'chedde'
    df.loc[df[f] == "pau ", f] = 'pau'
    df.loc[df[f] == "pavillon sous bois", f] = 'les pavillons sous bois'
    df.loc[df[f] == "pays de belves (belves)", f] = 'belves'
    df.loc[df[f] == "peigut pluviers", f] = 'piegut pluviers'
    df.loc[df[f] == "peille (la grave de peille)", f] = 'la grave de peille'
    df.loc[df[f] == "pelissane", f] = 'pelissanne'
    df.loc[df[f] == "pelissanre", f] = 'pelissanne'
    df.loc[df[f] == "pellisanne", f] = 'pelissanne'
    df.loc[df[f] == "penmarc h", f] = 'penmarch'
    df.loc[df[f] == "peray les gombries", f] = 'peroy les gombries'
    df.loc[df[f] == "percy en normandie (percy)", f] = 'percy'
    df.loc[df[f] == "perigny sur yerres (perigny)", f] = 'perigny'
    df.loc[df[f] == "pernes es fontaines", f] = 'pernes les fontaines'
    df.loc[df[f] == "pernes les fontaine", f] = 'pernes les fontaines'
    df.loc[df[f] == "perray", f] = 'le perray en yvelines'
    df.loc[df[f] == "perrelatte", f] = 'pierrelatte'
    df.loc[df[f] == "perthes en gatinois", f] = 'perthes en gatinais'
    df.loc[df[f] == "petit palais et caremps", f] = 'petit palais et cornemps'
    df.loc[df[f] == "peyrac", f] = 'payrac'
    df.loc[df[f] == "peyrrier", f] = 'perrier'
    df.loc[df[f] == "pian medoc", f] = 'le pian medoc'
    df.loc[df[f] == "picauville (amfreville)", f] = 'amfreville'
    df.loc[df[f] == "pierre fitte", f] = 'pierrefitte sur seine'
    df.loc[df[f] == "pierrefite sur seine", f] = 'pierrefitte sur seine'
    df.loc[df[f] == "plaisance du gers", f] = 'plaisance'
    df.loc[df[f] == "plaiseau", f] = 'palaiseau'
    df.loc[df[f] == "plan d'aups", f] = 'plan d aups sainte baume'
    df.loc[df[f] == "plan d'aups ste baume", f] = 'plan d aups sainte baume'
    df.loc[df[f] == "plan de cucques", f] = 'plan de cuques'
    df.loc[df[f] == "pleine de walsch", f] = 'plaine de walsch'
    df.loc[df[f] == "pleneuf", f] = 'pleneuf val andre'
    df.loc[df[f] == "pleslin trigavou (trigavou)", f] = 'trigavou'
    df.loc[df[f] == "pleslin trigourou", f] = 'trigavou'
    df.loc[df[f] == "plesse (le coudray)", f] = 'le coudray'
    df.loc[df[f] == "plesses trevise", f] = 'le plessis trevise'
    df.loc[df[f] == "plessis bouchard", f] = 'le plessis bouchard'
    df.loc[df[f] == "plessis robinson", f] = 'le plessis robinson'
    df.loc[df[f] == "pleudihen", f] = 'pleudihen sur rance'
    df.loc[df[f] == "ploeuc l'hermitage (l hermitage lorge)", f] = 'l hermitage lorge'
    df.loc[df[f] == "ploeuc l'hermitage (ploeuc sur lie)", f] = 'ploeuc sur lie'
    df.loc[df[f] == "ploudalmezeau (portsall)", f] = 'portsall'
    df.loc[df[f] == "plougerneau", f] = 'plouguerneau'
    df.loc[df[f] == "plouguenast langast", f] = 'plouguenast'
    df.loc[df[f] == "pluviguier", f] = 'pluvigner'
    df.loc[df[f] == "poiseux en france", f] = 'puiseux en france'
    df.loc[df[f] == "polaiseau", f] = 'palaiseau'
    df.loc[df[f] == "pomeuse", f] = 'pommeuse'
    df.loc[df[f] == "pommier la placette", f] = 'pommiers la placette'
    df.loc[df[f] == "pompome", f] = 'pomponne'
    df.loc[df[f] == "pomponne (la pomponnette)", f] = 'pomponne'
    df.loc[df[f] == "ponchery", f] = 'pondichery'
    df.loc[df[f] == "pont de briques", f] = 'saint etienne au mont'
    df.loc[df[f] == "pont de buis", f] = 'pont de buis les quimerch'
    df.loc[df[f] == "pont de claix", f] = 'le pont de claix'
    df.loc[df[f] == "pont de jalans", f] = 'jallans'
    df.loc[df[f] == "pont de l'etoile", f] = 'pont de l etoile'
    df.loc[df[f] == "pont de roide vermondans", f] = 'pont de roide'
    df.loc[df[f] == "pont point", f] = 'pontpoint'
    df.loc[df[f] == "pont sainte maxime", f] = 'pont sainte maxence'
    df.loc[df[f] == "pont ste marie", f] = 'pont sainte marie'
    df.loc[df[f] == "pontault combot", f] = 'pontault combault'
    df.loc[df[f] == "pontchartrain", f] = 'jouars pontchartrain'
    df.loc[df[f] == "ponthenay de bretagne", f] = 'parthenay de bretagne'
    df.loc[df[f] == "pontorson (moidrey)", f] = 'moidrey'
    df.loc[df[f] == "ponts sur seulles (lantheuil)", f] = 'lantheuil'
    df.loc[df[f] == "pornic (ste marie)", f] = 'pornic'
    df.loc[df[f] == "port sainte foy", f] = 'port sainte foy et ponchapt'
    df.loc[df[f] == "port ste foy", f] = 'port sainte foy et ponchapt'
    df.loc[df[f] == "port ste marie", f] = 'port sainte marie'
    df.loc[df[f] == "porte les valence", f] = 'portes les valence'
    df.loc[df[f] == "porte ste foy", f] = 'port sainte foy et ponchapt'
    df.loc[df[f] == "pouligney", f] = 'pouligney lusans'
    df.loc[df[f] == "pourriere", f] = 'pourrieres'
    df.loc[df[f] == "pourrierres", f] = 'pourrieres'
    df.loc[df[f] == "pre en pail saint samson", f] = 'saint samson'
    df.loc[df[f] == "prechac navarex", f] = 'prechac navarrenx'
    df.loc[df[f] == "preyssac d'excideuil ()", f] = 'preyssac d excideuil'
    df.loc[df[f] == "pribrac", f] = 'pibrac'
    df.loc[df[f] == "prigenrieux", f] = 'prigonrieux'
    df.loc[df[f] == "prignac marcamps", f] = 'prignac et marcamps'
    df.loc[df[f] == "prunay en yveline", f] = 'prunay en yvelines'
    df.loc[df[f] == "puget s sur  argens", f] = 'pugets sur  argens'
    df.loc[df[f] == "puget sur argent", f] = 'pugets sur  argens'
    df.loc[df[f] == "puget sur argents", f] = 'pugets sur  argens'
    df.loc[df[f] == "puis justaret", f] = 'pins justaret'
    df.loc[df[f] == "pujol le plan", f] = 'pujols'
    df.loc[df[f] == "pumergat", f] = 'plumergat'
    df.loc[df[f] == "puttelange les thionvilles", f] = 'puttelange les thionville'
    df.loc[df[f] == "quatre borgne", f] = 'quatre bornes'
    df.loc[df[f] == "quatre borgnes", f] = 'quatre bornes'
    df.loc[df[f] == "quatre borne", f] = 'quatre bornes'
    df.loc[df[f] == "quieurechain", f] = 'quievrechain'
    df.loc[df[f] == "quincy voisin", f] = 'quincy voisins'
    df.loc[df[f] == "racquighem", f] = 'racquinghem'
    df.loc[df[f] == "racuquighem", f] = 'racquinghem'
    df.loc[df[f] == "raphele", f] = 'arles'
    df.loc[df[f] == "raphele les arles", f] = 'arles'
    df.loc[df[f] == "rauville", f] = 'rauville la bigot'
    df.loc[df[f] == "rauvrois sur uthain", f] = 'rouvrois sur othain'
    df.loc[df[f] == "reaucourt", f] = 'beaucourt'
    df.loc[df[f] == "regnie durette (durette)", f] = 'durette'
    df.loc[df[f] == "rehon (heumont)", f] = 'heumont'
    df.loc[df[f] == "remalard en perche", f] = 'remalard'
    df.loc[df[f] == "retbel", f] = 'rethel'
    df.loc[df[f] == "reyneq", f] = 'reynes'
    df.loc[df[f] == "reze (pont rousseau)", f] = 'pont rousseau'
    df.loc[df[f] == "rhode saint genese", f] = 'rhodes saint genese'
    df.loc[df[f] == "rhodes saint geneve", f] = 'rhodes saint genese'
    df.loc[df[f] == "rieux la pape", f] = 'rilleux la pape'
    df.loc[df[f] == "risorangis", f] = 'ris orangis'
    df.loc[df[f] == "rive", f] = 'rives'
    df.loc[df[f] == "rives de l'yon (st florent des bois)", f] = 'saint florent des bois'
    df.loc[df[f] == "roches les beaupre", f] = 'roches les beaupres'
    df.loc[df[f] == "rocqiencourt", f] = 'rocquencourt'
    df.loc[df[f] == "roissy sur seine", f] = 'croissy sur seine'
    df.loc[df[f] == "romagnat, ", f] = 'aubiere'
    df.loc[df[f] == "romagny fontenay (romagny)", f] = 'romagny'
    df.loc[df[f] == "romilly sis", f] = 'romilly sur seine'
    df.loc[df[f] == "roque brune", f] = 'roquebrune sur argens'
    df.loc[df[f] == "roquebrune cap martin (carnoles)", f] = 'carnoles'
    df.loc[df[f] == "roquebrune s sur  argens", f] = 'roquebrune sur argens'
    df.loc[df[f] == "roquebrune sur sur sur argens", f] = 'roquebrune sur argens'
    df.loc[df[f] == "roquefort les pin", f] = 'roquefort les pins'
    df.loc[df[f] == "rosny sous bois france", f] = 'rosny sous bois'
    df.loc[df[f] == "roubaix ", f] = 'roubaix'
    df.loc[df[f] == "rouebrune cap martin", f] = 'roquebrune cap martin'
    df.loc[df[f] == "roullet saint estephe (st estephe)", f] = 'saint estephe'
    df.loc[df[f] == "rouvignes", f] = 'rouvignies'
    df.loc[df[f] == "rouvrais sur uthain", f] = 'rouvrois sur othain'
    df.loc[df[f] == "rueil malmaison (buzenval)", f] = 'rueil malmaison'
    df.loc[df[f] == "ruel malmaison", f] = 'rueil malmaison'
    df.loc[df[f] == "ruy montceau", f] = 'bourgoin jallieu'
    df.loc[df[f] == "sabigny sur orge", f] = 'savigny sur orge'
    df.loc[df[f] == "sablons sur huisne", f] = 'conde sur huisne'
    df.loc[df[f] == "sacy en brie", f] = 'sucy en brie'
    df.loc[df[f] == "sain cannat", f] = 'saint cannat'
    df.loc[df[f] == "sain etienne", f] = 'saint etienne'
    df.loc[df[f] == "sain genis laval", f] = 'saint genis laval'
    df.loc[df[f] == "sain raphael", f] = 'saint raphael'
    df.loc[df[f] == "sainghin en melentois", f] = 'sainghin en melantois'
    df.loc[df[f] == "saint   louis", f] = 'saint louis'
    df.loc[df[f] == "saint  herblain", f] = 'saint herblain'
    df.loc[df[f] == "saint  louis", f] = 'saint louis'
    df.loc[df[f] == "saint agnes", f] = 'sainte agnes'
    df.loc[df[f] == "saint aignan de granlieu", f] = 'saint aignan de grandlieu'
    df.loc[df[f] == "saint amand villages (la chapelle du fest)", f] = 'la chapelle du fest'
    df.loc[df[f] == "saint andre (cambuston)", f] = 'cambuston'
    df.loc[df[f] == "saint andre et apprelle", f] = 'saint andre et appelles'
    df.loc[df[f] == "saint andre sangony", f] = 'saint andre sangonis'
    df.loc[df[f] == "saint andre vieux jonc", f] = 'saint andre sur vieux jonc'
    df.loc[df[f] == "saint andres sur cailly", f] = 'saint andre sur cailly'
    df.loc[df[f] == "saint anne sur brivet", f] = 'sainte anne sur brivet'
    df.loc[df[f] == "saint antoine de breuihl", f] = 'saint antoine de breuilh'
    df.loc[df[f] == "saint armand", f] = 'saint armand montrond'
    df.loc[df[f] == "saint arnoult en yveline", f] = 'saint arnoult en yvelines'
    df.loc[df[f] == "saint aubin du medoc", f] = 'saint aubin de medoc'
    df.loc[df[f] == "saint augustin (clarques)", f] = 'clarques'
    df.loc[df[f] == "saint augustin (rebecques)", f] = 'rebecques'
    df.loc[df[f] == "saint averntin", f] = 'saint avertin'
    df.loc[df[f] == "saint avit de souhere", f] = 'saint avit de soulege'
    df.loc[df[f] == "saint barthelemy d'ajou", f] = 'ssaint barthelemy d anjou'
    df.loc[df[f] == "saint barthelemy de beaurepaire", f] = 'saint barthelemy'
    df.loc[df[f] == "saint bathelemy d&#;anjou", f] = 'saint barthelemy d anjou'
    df.loc[df[f] == "saint bazeille", f] = 'sainte bazeille'
    df.loc[df[f] == "saint bel", f] = 'sain bel'
    df.loc[df[f] == "saint benoit (ste anne)", f] = 'saint benoit'
    df.loc[df[f] == "saint bernard du touvet", f] = 'saint bernard'
    df.loc[df[f] == "saint bonnet de croy", f] = 'saint bonnet de cray'
    df.loc[df[f] == "saint bonnet du gare", f] = 'saint bonnet du gard'
    df.loc[df[f] == "saint brevin", f] = 'saint brevin les pins'
    df.loc[df[f] == "saint brevin les pins (st brevin l'ocean)", f] = 'saint brevin les pins'
    df.loc[df[f] == "saint brice courelles", f] = 'saint brice courcelles'
    df.loc[df[f] == "saint camas", f] = 'saint chamas'
    df.loc[df[f] == "saint canat", f] = 'saint cannat'
    df.loc[df[f] == "saint cezaire", f] = 'saint cezaire sur siagne'
    df.loc[df[f] == "saint claire du rhone", f] = 'saint clair du rhone'
    df.loc[df[f] == "saint croix grand tonne", f] = 'sainte croix grand tonne'
    df.loc[df[f] == "saint cyr au mpnt d'or", f] = 'saint cyr au mont d or'
    df.loc[df[f] == "saint cyr en retz", f] = 'villeneuve en retz'
    df.loc[df[f] == "saint cyr sur rhone", f] = 'saint cyr sur le rhone'
    df.loc[df[f] == "saint dennis", f] = 'saint denis'
    df.loc[df[f] == "saint des fosses", f] = 'saint maur des fosses'
    df.loc[df[f] == "saint dionisy", f] = 'saint dionizy'
    df.loc[df[f] == "saint etienne (st victor sur loire)", f] = 'saint etienne'
    df.loc[df[f] == "saint eulalie", f] = 'sainte eulalie'
    df.loc[df[f] == "saint fargau pontthierry", f] = 'ponthierry'
    df.loc[df[f] == "saint fargeau ponthierry (ponthierry)", f] = 'ponthierry'
    df.loc[df[f] == "saint flaive des loups", f] = 'sainte flaive des loups'
    df.loc[df[f] == "saint foy", f] = 'sainte foy tarentaise'
    df.loc[df[f] == "saint foy d'aigrefeuille", f] = 'sainte foy d aigrefeuille'
    df.loc[df[f] == "saint front sur nizanne", f] = 'saint front sur nizonne'
    df.loc[df[f] == "saint gabain", f] = 'saint gobain'
    df.loc[df[f] == "saint genes champanelle (berzet)", f] = 'berzet'
    df.loc[df[f] == "saint genevieve des bois", f] = 'sainte genevieve des bois'
    df.loc[df[f] == "saint genis poully", f] = 'saint genis pouilly'
    df.loc[df[f] == "saint genvievre des bois", f] = 'sainte genevieve des bois'
    df.loc[df[f] == "saint george des coteaux", f] = 'saint georges des coteaux'
    df.loc[df[f] == "saint georges d'oleron (cheray)", f] = 'cheray'
    df.loc[df[f] == "saint georges des groseilles", f] = 'saint georges des groseillers'
    df.loc[df[f] == "saint geours de marenne", f] = 'saint geours de maremne'
    df.loc[df[f] == "saint germain de tallevende", f] = 'saint germain de tallevende la lande vaumont'
    df.loc[df[f] == "saint germain du solembre", f] = 'saint germain du salembre'
    df.loc[df[f] == "saint germain en lay", f] = 'saint germain en laye'
    df.loc[df[f] == "saint germain les arpageons", f] = 'saint germain les arpajons'
    df.loc[df[f] == "saint germain les corbeilles", f] = 'saint germain les corbeil'
    df.loc[df[f] == "saint germain nuelles (nuelles)", f] = 'nuelles'
    df.loc[df[f] == "saint germain su ille", f] = 'saint germain sur ille'
    df.loc[df[f] == "saint germain villag", f] = 'saint germain village'
    df.loc[df[f] == "saint gorge desperanche", f] = 'saint georges d esperanche'
    df.loc[df[f] == "saint gratier", f] = 'saint gratien'
    df.loc[df[f] == "saint helene", f] = 'sainte helene'
    df.loc[df[f] == "saint hilaine saint mesmin", f] = 'saint hilaire saint mesmin'
    df.loc[df[f] == "saint hilaire du harcouet (virey)", f] = 'virey'
    df.loc[df[f] == "saint hilaire sur charlieu", f] = 'saint hilaire sous charlieu'
    df.loc[df[f] == "saint honorine des pertes", f] = 'sainte honorine des pertes'
    df.loc[df[f] == "saint jacques de lande", f] = 'saint jacques de la lande'
    df.loc[df[f] == "saint jean baisants", f] = 'saint jean des baisants'
    df.loc[df[f] == "saint jean cap ferry", f] = 'saint jean cap ferret'
    df.loc[df[f] == "saint jean d elle", f] = 'saint jean des baisants'
    df.loc[df[f] == "saint jean d'elle (st jean des baisants)", f] = 'saint jean des baisants'
    df.loc[df[f] == "saint jean de uz", f] = 'saint jean de luz'
    df.loc[df[f] == "saint jeean koutzerode", f] = 'saint jean kourtzerode'
    df.loc[df[f] == "saint jeoire en faucigny", f] = 'saint jeoire'
    df.loc[df[f] == "saint jouan des gerets", f] = 'saint jouan des guerets'
    df.loc[df[f] == "saint julien genevois", f] = 'saint julien en genevois'
    df.loc[df[f] == "saint just rambert", f] = 'saint just saint rambert'
    df.loc[df[f] == "saint lambert de la potherie", f] = 'saint lambert la potherie'
    df.loc[df[f] == "saint laurat medoc", f] = 'saint laurent medoc'
    df.loc[df[f] == "saint laurence d'arce", f] = 'saint laurence d arce'
    df.loc[df[f] == "saint laurent dagny", f] = 'saint laurent d agny'
    df.loc[df[f] == "saint laurent du medoc", f] = 'saint laurent medoc'
    df.loc[df[f] == "saint laurent manoire", f] = 'saint laurent sur manoire'
    df.loc[df[f] == "saint laurent nouan (nouan sur loire)", f] = 'nouan sur loire'
    df.loc[df[f] == "saint lheurine", f] = 'sainte lheurine'
    df.loc[df[f] == "saint louis (la riviere)", f] = 'la riviere'
    df.loc[df[f] == "saint louis (st louis la chaussee)", f] = 'saint louis la chaussee'
    df.loc[df[f] == "saint macaire en mauge", f] = 'saint macaire en mauges'
    df.loc[df[f] == "saint maen", f] = 'saint meen'
    df.loc[df[f] == "saint malo (chateau malo)", f] = 'saint malo'
    df.loc[df[f] == "saint mand", f] = 'saint mande'
    df.loc[df[f] == "saint marie", f] = 'sainte marie'
    df.loc[df[f] == "saint marie aux chenes", f] = 'sainte marie aux chenes'
    df.loc[df[f] == "saint martin d'huriage", f] = 'saint martin d huriage'
    df.loc[df[f] == "saint martin la pallu (blaslay)", f] = 'blaslay'
    df.loc[df[f] == "saint martin la pallu (charrais)", f] = 'charrais'
    df.loc[df[f] == "saint martin la pallu (vendeuvre du poitou)", f] = 'vendeuvre du poitou'
    df.loc[df[f] == "saint martin le vinou", f] = 'saint martin le vinoux'
    df.loc[df[f] == "saint martin lez tatinghem", f] = 'tatinghem'
    df.loc[df[f] == "saint martin lez tatinghem (tatinghem)", f] = 'tatinghem'
    df.loc[df[f] == "saint martine d'oney", f] = 'sainte martine d oney'
    df.loc[df[f] == "saint maur de foses", f] = 'saint maur des fosses'
    df.loc[df[f] == "saint maur des fausses", f] = 'saint maur des fosses'
    df.loc[df[f] == "saint maur des fesses", f] = 'saint maur des fosses'
    df.loc[df[f] == "saint maur des fossees", f] = 'saint maur des fosses'
    df.loc[df[f] == "saint maur des fosset", f] = 'saint maur des fosses'
    df.loc[df[f] == "saint maximim", f] = 'saint maximin'
    df.loc[df[f] == "saint maximin la saint baume", f] = 'saint maximin la sainte baume'
    df.loc[df[f] == "saint maximin sainte baume", f] = 'saint maximin la sainte baume'
    df.loc[df[f] == "saint medard de guiziere", f] = 'saint medard de guizieres'
    df.loc[df[f] == "saint medard en jalle", f] = 'saint medard en jalles'
    df.loc[df[f] == "saint mein", f] = 'ecoust saint mein'
    df.loc[df[f] == "saint melaine", f] = 'sainte melaine'
    df.loc[df[f] == "saint michel sur orges", f] = 'saint michel sur orge'
    df.loc[df[f] == "saint mitre les rempart", f] = 'saint mitre les remparts'
    df.loc[df[f] == "saint nazaire (st marc sur mer)", f] = 'saint nazaire'
    df.loc[df[f] == "saint nazaire les eynes", f] = 'saint nazaire les eymes'
    df.loc[df[f] == "saint niclas", f] = 'saint nicolas'
    df.loc[df[f] == "saint nom", f] = 'saint nom la breteche'
    df.loc[df[f] == "saint nom le breteche", f] = 'saint nom la breteche'
    df.loc[df[f] == "saint paterne le chevain (st paterne)", f] = 'saint patern'
    df.loc[df[f] == "saint paul  chateaux", f] = 'saint paul trois chateaux'
    df.loc[df[f] == "saint paul flaugnac", f] = 'flaugnac'
    df.loc[df[f] == "saint pere en retz.", f] = 'saint pere en retz'
    df.loc[df[f] == "saint philibert des champs", f] = 'saint philbert des champs'
    df.loc[df[f] == "saint pierre d'imbe", f] = 'saint pierre d irube'
    df.loc[df[f] == "saint pierre du pont", f] = 'pont saint pierre'
    df.loc[df[f] == "saint pierre en auge (boissey)", f] = 'boissey'
    df.loc[df[f] == "saint pierre juillers", f] = 'saint pierre de juillers'
    df.loc[df[f] == "saint pol", f] = 'saint pol de leon'
    df.loc[df[f] == "saint privat de perigord", f] = 'saint privat en perigord'
    df.loc[df[f] == "saint privat des vie", f] = 'saint privat des vieux'
    df.loc[df[f] == "saint puierre de soucy", f] = 'saint pierre de soucy'
    df.loc[df[f] == "saint quentin la motte croix bailly", f] = 'saint quentin la motte croix au bailly'
    df.loc[df[f] == "saint ragotien", f] = 'saint rogatien'
    df.loc[df[f] == "saint raphael (agay)", f] = 'agay'
    df.loc[df[f] == "saint remy en bouzemont", f] = 'saint remy en bouzemont saint genest et isson'
    df.loc[df[f] == "saint remy les chevreux", f] = 'saint remy les chevreuses'
    df.loc[df[f] == "saint remy leschevreux", f] = 'saint remy les chevreuses'
    df.loc[df[f] == "saint remy sur orne", f] = 'saint remy'
    df.loc[df[f] == "saint romain de juliunus", f] = 'saint romain de jalionas'
    df.loc[df[f] == "saint romain les arthaux", f] = 'saint romain les arthaud'
    df.loc[df[f] == "saint saturin sur loire", f] = 'saint saturnin sur loire'
    df.loc[df[f] == "saint saufflieu", f] = 'saint sauflieu'
    df.loc[df[f] == "saint savine", f] = 'sainte savine'
    df.loc[df[f] == "saint senis", f] = 'saint denis'
    df.loc[df[f] == "saint servant", f] = 'ploermel'
    df.loc[df[f] == "saint seven", f] = 'saint sever'
    df.loc[df[f] == "saint siman de bordes", f] = 'saint simon de bordes'
    df.loc[df[f] == "saint sir la foret", f] = 'saint cyr la foret'
    df.loc[df[f] == "saint soulle", f] = 'sainte soulle'
    df.loc[df[f] == "saint soupplet", f] = 'saint soupplets'
    df.loc[df[f] == "saint suivre d'indre", f] = 'sainte severe sur indre'
    df.loc[df[f] == "saint sulpice de faleyron", f] = 'saint sulpice de faleyrens'
    df.loc[df[f] == "saint sulpice de pommeray", f] = 'onzain'
    df.loc[df[f] == "saint sulpice et cannejac", f] = 'saint sulpice et cameyrac'
    df.loc[df[f] == "saint suplice de faleyens", f] = 'saint sulpice de faleyrens'
    df.loc[df[f] == "saint suplice la foret", f] = 'saint sulpice la foret'
    df.loc[df[f] == "saint symphorien dozon", f] = 'saint symphorien d ozon'
    df.loc[df[f] == "saint thegonnec loc eguiner", f] = 'saint thegonnec'
    df.loc[df[f] == "saint victor sur loire", f] = 'saint etienne'
    df.loc[df[f] == "saint vincent sur graon (st sornin)", f] = 'saint sornin'
    df.loc[df[f] == "saint vincente brany", f] = 'saint vincent bragny'
    df.loc[df[f] == "saint vraim", f] = 'saint vrain'
    df.loc[df[f] == "sainte foi les lyon", f] = 'sainte foy les lyon'
    df.loc[df[f] == "sainte foi les lyons", f] = 'sainte foy les lyon'
    df.loc[df[f] == "sainte foy les lyob", f] = 'sainte foy les lyon'
    df.loc[df[f] == "sainte gemmes d'aubigne", f] = 'sainte gemmes d aubigne'
    df.loc[df[f] == "sainte geneviene des bois", f] = 'sainte genevieve des bois'
    df.loc[df[f] == "sainte lucie de porto vecchio", f] = 'zonza'
    df.loc[df[f] == "sainte palais de phiolin", f] = 'saint palais de phiolin'
    df.loc[df[f] == "sainte suzane", f] = 'sainte suzanne'
    df.loc[df[f] == "saintry", f] = 'saintry sur seine'
    df.loc[df[f] == "salanche", f] = 'sallanches'
    df.loc[df[f] == "salaunez", f] = 'salaunes'
    df.loc[df[f] == "salignac eyuigues", f] = 'salignac eyvigues'
    df.loc[df[f] == "salin de giraud", f] = 'arles'
    df.loc[df[f] == "saline (sannerville)", f] = 'sannerville'
    df.loc[df[f] == "sallarches", f] = 'sallanches'
    df.loc[df[f] == "salles et partviel", f] = 'salles et pratviel'
    df.loc[df[f] == "samois", f] = 'samois sur seine'
    df.loc[df[f] == "sanebourg", f] = 'sarrebourg'
    df.loc[df[f] == "sangatte (bleriot)", f] = 'bleriot'
    df.loc[df[f] == "sanilhac (marsaneix)", f] = 'marsaneix'
    df.loc[df[f] == "sanilhac (notre dame de sanilhac)", f] = 'notre dame de sanilhac'
    df.loc[df[f] == "sanite clotilde", f] = 'sainte clotilde'
    df.loc[df[f] == "sanois", f] = 'sannois'
    df.loc[df[f] == "sanveny", f] = 'santeny'
    df.loc[df[f] == "sap en auge (le sap)", f] = 'le sap'
    df.loc[df[f] == "sarcelle", f] = 'sarcelles'
    df.loc[df[f] == "sarcelles du bois", f] = 'sarcelles'
    df.loc[df[f] == "sarlat la canedat", f] = 'sarlat la caneda'
    df.loc[df[f] == "sartilly baie bocage (sartilly)", f] = 'sartilly'
    df.loc[df[f] == "sassenge", f] = 'sassenage'
    df.loc[df[f] == "satrouville", f] = 'sartrouville'
    df.loc[df[f] == "saul les chantreux", f] = 'saulx les chartreux'
    df.loc[df[f] == "saulain", f] = 'saulaine'
    df.loc[df[f] == "saulces monclins", f] = 'saulces monclin'
    df.loc[df[f] == "sauvigny en terre plaine", f] = 'savigny en terre plaine'
    df.loc[df[f] == "sauzerais", f] = 'saizerais'
    df.loc[df[f] == "scherwiller (kientzville)", f] = 'kientzville'
    df.loc[df[f] == "segre en anjou bleu", f] = 'segre'
    df.loc[df[f] == "segre en anjou bleu (segre)", f] = 'segre'
    df.loc[df[f] == "seillons source d'arjens", f] = 'seillons source d argens'
    df.loc[df[f] == "senille saint sauveur (st sauveur)", f] = 'saint sauveur'''
    df.loc[df[f] == "septemes les vallons (la rougiere)", f] = 'septemes les vallons'
    df.loc[df[f] == "septiemes les vallons", f] = 'septemes les vallons'
    df.loc[df[f] == "septmes les vallons", f] = 'septemes les vallons'
    df.loc[df[f] == "serbanne", f] = 'serbannes'
    df.loc[df[f] == "sernaise", f] = 'sermaise'
    df.loc[df[f] == "servigny les ste barbe", f] = 'servigny les sainte barbe'
    df.loc[df[f] == "sevremoine (le longeron)", f] = 'le longeron'
    df.loc[df[f] == "sevremoine (montfaucon montigne)", f] = 'montfaucon montigne'
    df.loc[df[f] == "sevremoine (st andre de la marche)", f] = 'saint andre de la marche'
    df.loc[df[f] == "sevremoine (st crespin sur moine)", f] = 'saint crespin sur moine'
    df.loc[df[f] == "sevremoine (tillieres)", f] = 'tillieres'
    df.loc[df[f] == "sevremoine (torfou)", f] = 'torfou'
    df.loc[df[f] == "sevremont (la flocelliere)", f] = 'la flocelliere'
    df.loc[df[f] == "sevremont (la pommeraie sur sevre)", f] = 'la pommeraie sur sevre'
    df.loc[df[f] == "seyne les alpes", f] = 'seyne'
    df.loc[df[f] == "seyreste", f] = 'ceyreste'
    df.loc[df[f] == "siant andre de cubzac", f] = 'saint andre de cubzac'
    df.loc[df[f] == "signe", f] = 'signes'
    df.loc[df[f] == "sivry caurty", f] = 'sivry courtry'
    df.loc[df[f] == "sivry country", f] = 'sivry courtry'
    df.loc[df[f] == "six four les plages", f] = 'six fours les plages'
    df.loc[df[f] == "six fours", f] = 'six fours les plages'
    df.loc[df[f] == "sobiers", f] = 'sorbiers'
    df.loc[df[f] == "solies ville", f] = 'sollies ville'
    df.loc[df[f] == "sollies", f] = 'sollies pont'
    df.loc[df[f] == "sollies  pont", f] = 'sollies pont'
    df.loc[df[f] == "solus d'oleron", f] = 'dolus d oleron'
    df.loc[df[f] == "someville", f] = 'chevillon'
    df.loc[df[f] == "sommerau (allenwiller)", f] = 'allenwiller'
    df.loc[df[f] == "sommerau (singrist)", f] = 'singrist'
    df.loc[df[f] == "sophia antipolis (biot)", f] = 'biot'
    df.loc[df[f] == "sorges et ligueux en perigord (sorges)", f] = 'sorges'
    df.loc[df[f] == "souleuvre en bocage (carville)", f] = 'carville'
    df.loc[df[f] == "souleuvre en bocage (etouvy)", f] = 'etouvy'
    df.loc[df[f] == "souleuvre en bocage (montchauvet)", f] = 'montchauvet'
    df.loc[df[f] == "souleuvre en bocage (ste marie laumont)", f] = 'sainte marie laumont'
    df.loc[df[f] == "soultzmatt (wintzfelden)", f] = 'wintzfelden'
    df.loc[df[f] == "sourdeval (vengeons)", f] = 'vengeons'
    df.loc[df[f] == "souston", f] = 'soustons'
    df.loc[df[f] == "souvelade", f] = 'sauvelade'
    df.loc[df[f] == "spechbach (spechbach le bas)", f] = 'pechbach'
    df.loc[df[f] == "spiecheren", f] = 'spicheren'
    df.loc[df[f] == "sr doruit sur l'herbasse", f] = 'saint donat sur l herbasse'
    df.loc[df[f] == "staint nazaire", f] = 'saint nazaire'
    df.loc[df[f] == "stiring werdel", f] = 'stiring wendel'
    df.loc[df[f] == "stutzheim offenheim (offenheim)", f] = 'offenheim'
    df.loc[df[f] == "suresne", f] = 'suresnes'
    df.loc[df[f] == "sussy en bry", f] = 'sucy en bry'
    df.loc[df[f] == "tauxigny saint bauld (st bauld)", f] = 'saint bauld'
    df.loc[df[f] == "tavergny", f] = 'taverny'
    df.loc[df[f] == "taverni", f] = 'taverny'
    df.loc[df[f] == "terranjou (martigne briand)", f] = 'martigne briand'
    df.loc[df[f] == "terrasson", f] = 'terrasson lavilledieu'
    df.loc[df[f] == "terrasson lavilledie", f] = 'terrasson lavilledieu'
    df.loc[df[f] == "terres de caux", f] = 'fauville en caux'
    df.loc[df[f] == "terres de druance (st jean le blanc)", f] = 'saint jean le blanc'
    df.loc[df[f] == "tessy bocage (tessy sur vire)", f] = 'tessy sur vire'
    df.loc[df[f] == "teteghem coudekerque village (teteghem)", f] = 'teteghem'
    df.loc[df[f] == "teteghen", f] = 'teteghem'
    df.loc[df[f] == "tevelave", f] = 'le tevelave'
    df.loc[df[f] == "theix noyalo", f] = 'noyalo'
    df.loc[df[f] == "theix noyalo (noyalo)", f] = 'noyalo'
    df.loc[df[f] == "thereval (hebecrevon)", f] = 'hebecrevon'
    df.loc[df[f] == "thery glimont", f] = 'thezy glimont'
    df.loc[df[f] == "thezy saint martin", f] = 'thezey saint martin'
    df.loc[df[f] == "thies", f] = 'thiers'
    df.loc[df[f] == "thionville (garche)", f] = 'thionville'
    df.loc[df[f] == "thones, ", f] = 'thones'
    df.loc[df[f] == "thonev", f] = 'thones'
    df.loc[df[f] == "thonex", f] = 'thones'
    df.loc[df[f] == "thoure sur loire", f] = 'thouare sur loire'
    df.loc[df[f] == "thovigny sur marne", f] = 'thorigny sur marne'
    df.loc[df[f] == "thrith saint leger", f] = 'trith saint leger'
    df.loc[df[f] == "tignieu", f] = 'tignieu jameyzieu'
    df.loc[df[f] == "tilloy les cambrai", f] = 'tilloy lez cambrai'
    df.loc[df[f] == "torigny les villes", f] = 'brectouville'
    df.loc[df[f] == "torigny les villes (brectouville)", f] = 'brectouville'
    df.loc[df[f] == "torigny les villes (gieville)", f] = 'gieville'
    df.loc[df[f] == "torigny les villes (guilberville)", f] = 'gieville'
    df.loc[df[f] == "touet  l'escarene", f] = 'touet l escarene'
    df.loc[df[f] == "toulose", f] = 'toulouse'
    df.loc[df[f] == "toulouse ", f] = 'toulouse'
    df.loc[df[f] == "tour de salvagny", f] = 'la tour de salvagny'
    df.loc[df[f] == "tourette", f] = 'tourrette levens'
    df.loc[df[f] == "tourette sur loup", f] = 'tourrettes sur loup'
    df.loc[df[f] == "tourettes sur loup", f] = 'tourrettes sur loup'
    df.loc[df[f] == "tournant en brie", f] = 'tournan en brie'
    df.loc[df[f] == "tournefeille", f] = 'tournefeuille'
    df.loc[df[f] == "tourouvre au perche (prepotin)", f] = 'prepotin'
    df.loc[df[f] == "tourouvre au perche (randonnai)", f] = 'randonnai'
    df.loc[df[f] == "tourrette", f] = 'tourrette levens'
    df.loc[df[f] == "tourrette   levens", f] = 'tourrette levens'
    df.loc[df[f] == "tramblay en france", f] = 'tremblay en france'
    df.loc[df[f] == "tredrez locquemeau (locquemeau)", f] = 'locquemeau'
    df.loc[df[f] == "treffor cuisiat", f] = 'treffort cuisiat'
    df.loc[df[f] == "tremeny", f] = 'tremery'
    df.loc[df[f] == "tressange (bure)", f] = 'bure'
    df.loc[df[f] == "trois fontaines", f] = 'troisfontaines'
    df.loc[df[f] == "trolissac", f] = 'trelissac'
    df.loc[df[f] == "truchtersheim (pfettisheim)", f] = 'pfettisheim'
    df.loc[df[f] == "tucquenieux", f] = 'tucquegnieux'
    df.loc[df[f] == "vair sur loire (anetz)", f] = 'anetz'
    df.loc[df[f] == "vaire (vaire arcier)", f] = 'vaire arcier'
    df.loc[df[f] == "vaires sur marnes", f] = 'vaires sur marne'
    df.loc[df[f] == "vaires sur seine", f] = 'vaires sur marne'
    df.loc[df[f] == "val buech meouge (ribiers)", f] = 'ribiers'
    df.loc[df[f] == "val d oingt (le bois d'oingt)", f] = 'le bois d oingt'
    df.loc[df[f] == "val d oingt (st laurent d'oingt)", f] = 'saint laurent d oingt'
    df.loc[df[f] == "val d'anast", f] = 'campel'
    df.loc[df[f] == "val d'anast (campel)", f] = 'campel'
    df.loc[df[f] == "val d'anast (maure de bretagne)", f] = 'maure de bretagne'
    df.loc[df[f] == "val d'erdre auxence", f] = 'le louroux beconnais'
    df.loc[df[f] == "val d'oingt (oingt)", f] = 'oingt'
    df.loc[df[f] == "val d'oust (le roc saint andre)", f] = 'le roc saint andre'
    df.loc[df[f] == "val de briey", f] = 'mance'
    df.loc[df[f] == "val de briey (mance)", f] = 'mance'
    df.loc[df[f] == "val de briey (mancieulles)", f] = 'mancieulles'
    df.loc[df[f] == "val de drome (sept vents)", f] = 'sept vents'
    df.loc[df[f] == "val de louyre et caudeau (cendrieux)", f] = 'cendrieux'
    df.loc[df[f] == "val de moder (pfaffenhoffen)", f] = 'pfaffenhoffen'
    df.loc[df[f] == "val de moder (uberach)", f] = 'uberach'
    df.loc[df[f] == "val de virvee (aubie et espessas)", f] = 'aubie et espessas'
    df.loc[df[f] == "val de virvee (salignac)", f] = 'salignac'
    df.loc[df[f] == "val du maine (ballee)", f] = 'ballee'
    df.loc[df[f] == "val en vignes (cersay)", f] = 'cersay'
    df.loc[df[f] == "valady (nuces)", f] = 'nuces'
    df.loc[df[f] == "valambray (poussy la campagne)", f] = 'poussy la campagne'
    df.loc[df[f] == "valbonne (sophia antipolis)", f] = 'sophia antipolis'
    df.loc[df[f] == "valdalliere (bernieres le patry)", f] = 'bernieres le patry'
    df.loc[df[f] == "valdalliere (vassy)", f] = 'vassy'
    df.loc[df[f] == "valdalliere (viessoix)", f] = 'viessoix'
    df.loc[df[f] == "valdeancourt", f] = 'valdelancourt'
    df.loc[df[f] == "valdoule (bruis)", f] = 'bruis'
    df.loc[df[f] == "valence d'agen", f] = 'valence d agen'
    df.loc[df[f] == "valenciennes, france", f] = 'valenciennes'
    df.loc[df[f] == "vallauris (le golfe juan)", f] = 'vallauris'
    df.loc[df[f] == "vals pres le puis", f] = 'vals pres le puy'
    df.loc[df[f] == "valserhone (bellegarde sur valserine)", f] = 'bellegarde sur valserine'
    df.loc[df[f] == "vandoeuvre", f] = 'vandoeuvre les nancy'
    df.loc[df[f] == "vannnes", f] = 'vannes'
    df.loc[df[f] == "varouis et chaignot", f] = 'varois et chaignot'
    df.loc[df[f] == "vauchelles les quesnoys", f] = 'vauchelles les quesnoy'
    df.loc[df[f] == "vauclin", f] = 'le vauclin'
    df.loc[df[f] == "vaucresse", f] = 'vaucresson'
    df.loc[df[f] == "vaudemange", f] = 'vaudemanges'
    df.loc[df[f] == "vauert", f] = 'vauvert'
    df.loc[df[f] == "vaux warnimont", f] = 'cosnes et romain'
    df.loc[df[f] == "vedrin", f] = 'namur'
    df.loc[df[f] == "vegrines de vergt", f] = 'veyrines de vergt'
    df.loc[df[f] == "velizy villacoubay", f] = 'velizy villacoublay'
    df.loc[df[f] == "velledieu", f] = 'villedieu'
    df.loc[df[f] == "vendin lez bethune", f] = 'vendin les bethune'
    df.loc[df[f] == "venosc (les deux alpes)", f] = 'venosc'
    df.loc[df[f] == "verchain maugne", f] = 'verchain maugre'
    df.loc[df[f] == "verdegies sur ecaillon", f] = 'vendegies sur ecaillon'
    df.loc[df[f] == "verdin le vieil", f] = 'vendin le vieil'
    df.loc[df[f] == "vergeraux", f] = 'vergeroux'
    df.loc[df[f] == "verleuil", f] = 'verneuil sur seine'
    df.loc[df[f] == "verneuil en halette", f] = 'verneuil en halatte'
    df.loc[df[f] == "verns sur seiche", f] = 'vern sur seiche'
    df.loc[df[f] == "verosvres, les essertines", f] = 'verosvres'
    df.loc[df[f] == "verquigneuil", f] = 'bethune'
    df.loc[df[f] == "verquigneul", f] = 'bethune'
    df.loc[df[f] == "verriere le buisson", f] = 'verrieres le buisson'
    df.loc[df[f] == "verrieres le buissan", f] = 'verrieres le buisson'
    df.loc[df[f] == "vers le buisson", f] = 'verrieres le buisson'
    df.loc[df[f] == "versaille", f] = 'versailles'
    df.loc[df[f] == "vertou (beautour)", f] = 'beautour'
    df.loc[df[f] == "vesinet", f] = 'le vesinet'
    df.loc[df[f] == "veson", f] = 'verson'
    df.loc[df[f] == "veuchette", f] = 'veauchette'
    df.loc[df[f] == "vexin sur epte (fourges)", f] = 'fourges'
    df.loc[df[f] == "vexin sur epte (fours en vexin)", f] = 'fours en vexin'
    df.loc[df[f] == "veyrier", f] = 'veyrier du lac'
    df.loc[df[f] == "veyrins   thuellin", f] = 'veyrins thuellin'
    df.loc[df[f] == "veille toulouse", f] = 'vieille toulouse'
    df.loc[df[f] == "vieux   habitants", f] = 'vieux habitants'
    df.loc[df[f] == "vieux berquine", f] = 'vieux berquin'
    df.loc[df[f] == "vigne aux bois", f] = 'vrigne aux bois'
    df.loc[df[f] == "vigneux sur  seine", f] = 'vigneux sur seine'
    df.loc[df[f] == "viilliers le bel", f] = 'villiers le bel'
    df.loc[df[f] == "vilieresnes", f] = 'villecresnes'
    df.loc[df[f] == "villard bonnot (lancey)", f] = 'lancey'
    df.loc[df[f] == "ville d'auray", f] = 'ville d avray'
    df.loc[df[f] == "ville du bois", f] = 'la ville du bois'
    df.loc[df[f] == "ville la frand", f] = 'ville la grand'
    df.loc[df[f] == "ville neuve le roi", f] = 'villeneuve le roi'
    df.loc[df[f] == "ville thierry", f] = 'ville thierry'
    df.loc[df[f] == "villecresne", f] = 'villecresnes'
    df.loc[df[f] == "villejuste", f] = 'villejust'
    df.loc[df[f] == "villelongue del monts", f] = 'villelongue dels monts'
    df.loc[df[f] == "villemaury (civry)", f] = 'civry'
    df.loc[df[f] == "villenave de navarrenx", f] = 'viellenave de navarrenx'
    df.loc[df[f] == "villeneoy", f] = 'villenoy'
    df.loc[df[f] == "villeneuve  d'ascq", f] = 'villeneuve d ascq'
    df.loc[df[f] == "villeneuve d'acq", f] = 'villeneuve d ascq'
    df.loc[df[f] == "villeneuve d&#;ascq", f] = 'villeneuve d ascq'
    df.loc[df[f] == "villeneuve dascq", f] = 'villeneuve d ascq'
    df.loc[df[f] == "villeneuve en retz (bourgneuf en retz)", f] = 'bourgneuf en retz'
    df.loc[df[f] == "villeneuve en retz (fresnay en retz)", f] = 'fresnay en retz'
    df.loc[df[f] == "villeneuve les avignons", f] = 'villeneuve les avignon'
    df.loc[df[f] == "villeneuve sur yonnes", f] = 'villeneuve sur yonne'
    df.loc[df[f] == "villenoble", f] = 'villemomble'
    df.loc[df[f] == "villenveuve saint geroge", f] = 'villeneuve saint georges'
    df.loc[df[f] == "villers outneaux", f] = 'villers outreaux'
    df.loc[df[f] == "villers outreau", f] = 'villers outreaux'
    df.loc[df[f] == "villers saint etrambourg", f] = 'villers saint frambourg'
    df.loc[df[f] == "villerubanne", f] = 'villeurbanne'
    df.loc[df[f] == "villerupt (cantebonne)", f] = 'cantebonne'
    df.loc[df[f] == "villesegure", f] = 'viellesegure'
    df.loc[df[f] == "villeuneuve loubet", f] = 'villeneuve loubet'
    df.loc[df[f] == "villeuneuve sur lot", f] = 'villeneuve sur lot'
    df.loc[df[f] == "villier le bel", f] = 'villiers le bel'
    df.loc[df[f] == "villier sur marne", f] = 'villiers sur marne'
    df.loc[df[f] == "villier sur orge", f] = 'villiers sur orge'
    df.loc[df[f] == "villiers saint sepulcre", f] = 'villers saint sepulcre'
    df.loc[df[f] == "villiers sous prez", f] = 'villiers sous grez'
    df.loc[df[f] == "villieu loyes mollon (loyes)", f] = 'loyes'
    df.loc[df[f] == "villieu loyes mollon (mollon)", f] = 'mollon'
    df.loc[df[f] == "vilmoisson sur orge", f] = 'villemoisson sur orge'
    df.loc[df[f] == "vindry sur turdine", f] = 'pontcharra sur turdine'
    df.loc[df[f] == "viny naureuil", f] = 'viny noureuil'
    df.loc[df[f] == "vire normandie", f] = 'vire'
    df.loc[df[f] == "vire normandie (maisoncelles la jourdan)", f] = 'maisoncelles la jourdan'
    df.loc[df[f] == "vire normandie (roullours)", f] = 'roullours'
    df.loc[df[f] == "vire normandie (vaudry)", f] = 'vaudry'
    df.loc[df[f] == "vire normandie (vire)", f] = 'vire'
    df.loc[df[f] == "vitrai", f] = 'vitrai sous laigle'
    df.loc[df[f] == "void vacon (vacon)", f] = 'vacon'
    df.loc[df[f] == "votrolles", f] = 'vitrolles'
    df.loc[df[f] == "vouel", f] = 'tergnier'
    df.loc[df[f] == "walcheid", f] = 'walscheid'
    df.loc[df[f] == "wingersheim les quatre bans", f] = 'hohatzenheim'
    df.loc[df[f] == "wintzenheim (logelbach)", f] = 'logelbach'
    df.loc[df[f] == "worhmout", f] = 'wormhout'
    df.loc[df[f] == "wottersdorf", f] = 'wittersdorf'
    df.loc[df[f] == "yeres", f] = 'yerres'
    df.loc[df[f] == "yerres (montgeron)", f] = 'yerres'
    df.loc[df[f] == "zutherque", f] = 'zutkerque'
    df.loc[df[f] == "muy", f] = 'le muy'
    df.loc[df[f] == "marseille saint charles", f] = 'marseille'
    df.loc[df[f] == "limoges benedictins", f] = 'limoges'
    df.loc[df[f] == "saint raphael valescure", f] = 'saint raphael'
    df.loc[df[f] == "caussade tarn et garonne", f] = 'caussade'
    df.loc[df[f] == "bordeaux saint jean", f] = 'bordeaux'
    df.loc[df[f] == "les arcs draguignan", f] = 'draguignan'
    df.loc[df[f] == "les aubrais orleans", f] = 'orleans'
    df.loc[df[f] == "montauban ville bourbon", f] = 'montauban'
    df.loc[df[f] == "port vendres ville", f] = 'port vendres'
    df.loc[df[f] == "riom chatel guyon", f] = 'riom'
    df.loc[df[f] == "nice ville", f] = 'nice'
    df.loc[df[f] == "vierzon ville", f] = 'vierzon'
    df.loc[df[f] == "bastille", f] = 'paris'
    df.loc[df[f] == "la chapelle gauthier ()", f] = 'la chapelle gauthier'
    df.loc[df[f] == "issy", f] = 'issy les moulineaux'
    df.loc[df[f] == "boulogne billancourt ", f] = 'boulogne billancourt'
    df.loc[df[f] == "la defense", f] = 'puteaux'
    df.loc[df[f] == "pairs", f] = 'paris'
    df.loc[df[f] == "sophia antipolis", f] = 'valbonne'
    df.loc[df[f] == "val de fontenay", f] = 'fontenay sous bois'
    df.loc[df[f] == "begles ", f] = 'begles'
    df.loc[df[f] == "talence ()", f] = 'talence'
    df.loc[df[f] == "taix les milles", f] = 'aix en provence'
    df.loc[df[f] == "valbonne ", f] = 'valbonne'
    df.loc[df[f] == "toulouse lagebe", f] = 'toulouse'
    df.loc[df[f] == "talence ()", f] = 'talence'
    df.loc[df[f] == "talence ()", f] = 'talence'
    df.loc[df[f] == "l isle sur sorgue", f] = "l isle sur la sorgue"
    df.loc[df[f] == "lamentin", f] = "le lamentin"
    df.loc[df[f] == "bagnol sur ceze", f] = "bagnols sur ceze"
    df.loc[df[f] == "bagnos sur ceze", f] = "bagnols sur ceze"
    df.loc[df[f] == "le  raincy", f] = "le raincy"
    df.loc[df[f] == "leraincy", f] = "le raincy"
    df.loc[df[f] == "principaute de monaco", f] = "monaco"
    df.loc[df[f] == "deanin", f] = "denain"
    df.loc[df[f] == "villelongue del mont", f] = "villelongue dels monts"
    df.loc[df[f] == "saint benois", f] = "saint benoit"
    df.loc[df[f] == "bishwiller", f] = "bischwiller"
    df.loc[df[f] == "dunkerque ", f] = "dunkerque"
    df.loc[df[f] == "dunkerques ", f] = "dunkerque"
    df.loc[df[f] == "dunkerque ", f] = "dunkerque"
    df.loc[df[f] == "dunkkerque ", f] = "dunkerque"
    df.loc[df[f] == "dukerque ", f] = "dunkerque"
    df.loc[df[f] == "avirons", f] = "les avirons"
    df.loc[df[f] == "belignat", f] = "bellignat"
    df.loc[df[f] == "antibe", f] = "antibes"
    df.loc[df[f] == "atnibes", f] = "antibes"
    df.loc[df[f] == "antibes juans les pins", f] = "antibes"
    df.loc[df[f] == "asnieres sur seines", f] = "asnieres sur seine"
    df.loc[df[f] == "pointes a pitre", f] = "pointe a pitre"
    df.loc[df[f] == "abymes", f] = "pointe a pitre"
    df.loc[df[f] == "les abymes", f] = "pointe a pitre"
    df.loc[df[f] == "fort de france martinique", f] = "fort de france"
    df.loc[df[f] == "charbonnieres", f] = "charbonnieres les bains"
    df.loc[df[f] == "letampon", f] = "le tampon"
    df.loc[df[f] == "becon les granites", f] = "becon les granits"
    df.loc[df[f] == "rennes ", f] = "rennes"
    df.loc[df[f] == "clermont (de l'oise)", f] = "clermont"
    df.loc[df[f] == "thionbille", f] = "thionville"
    df.loc[df[f] == "longperier", f] = "longperrier"
    df.loc[df[f] == "andorra la vella", f] = "andorre la vielle"
    df.loc[df[f] == "la celle saint coud", f] = "la celle saint"
    df.loc[df[f] == "orasay", f] = "orsay"
    df.loc[df[f] == "kremelin bicetre", f] = "le kremlin bicetre"
    df.loc[df[f] == "pavillions sous bois", f] = "les pavillons sous bois"
    df.loc[df[f] == "ulis", f] = "les ulis"
    df.loc[df[f] == "possession", f] = "la possession"
    df.loc[df[f] == "le brede", f] = "la brede"
    df.loc[df[f] == "sarcellles", f] = "sarcelles"
    df.loc[df[f] == "trois bassins", f] = "les trois bassins"
    df.loc[df[f] == "place de la republic", f] = "paris"
    df.loc[df[f] == "conde sur escaut", f] = "conde sur l escaut"
    df.loc[df[f] == "villemendeur", f] = "villemandeur"
    df.loc[df[f] == "vincenne", f] = "vincennes"
    df.loc[df[f] == "chatellereau", f] = "chatellerault"
    df.loc[df[f] == "nopgent sur marne", f] = "nogent sur marne"
    df.loc[df[f] == "barbezieux", f] = "barbezieux saint hilaire"
    df.loc[df[f] == "montigny le bretnneux", f] = "montigny le bretonneux"
    df.loc[df[f] == "saint livrade sur lot", f] = "sainte livrade sur lot"
    df.loc[df[f] == "issy le moulineaux", f] = "issy les moulineaux"
    df.loc[df[f] == "fontainbleau", f] = "fontainebleau"
    df.loc[df[f] == "andrezieux", f] = "andrezieux boutheon"
    df.loc[df[f] == "frenes", f] = "fresnes"
    df.loc[df[f] == "cegy", f] = "cergy"
    df.loc[df[f] == "gennevillier", f] = "gennevilliers"
    df.loc[df[f] == "aire sur l adour", f] = "aire sur la??dour"
    df.loc[df[f] == "rive de giee", f] = "rive de gier"
    df.loc[df[f] == "luzarche", f] = "luzarches"
    df.loc[df[f] == "limel brevannes", f] = "limeil brevannes"
    df.loc[df[f] == "ermon eaubonne", f] = "ermont"
    df.loc[df[f] == "rueil mailmaison", f] = "rueil malmaison"
    df.loc[df[f] == "rueil malmasionn", f] = "rueil malmaison"
    df.loc[df[f] == "sarreguemine", f] = "sarreguemine"
    df.loc[df[f] == "issy les moulinaux", f] = "issy les moulineaux"
    df.loc[df[f] == "claamart", f] = "clamart"
    df.loc[df[f] == "lonjgumeau", f] = "longjumeau"
    df.loc[df[f] == "blayes", f] = "blaye"
    df.loc[df[f] == "agen ", f] = "agen"
    df.loc[df[f] == "capentras", f] = "carpentras"
    df.loc[df[f] == "la queue lez yvelines", f] = "la queue les yvelines"
    df.loc[df[f] == "saint andree", f] = "saint andre"
    df.loc[df[f] == "saint ouen l?aumone", f] = "saint ouen l aumone"
    df.loc[df[f] == "marcq en bar?ul", f] = "marcq en baroeul"
    df.loc[df[f] == "monpellier", f] = "montpellier"
    df.loc[df[f] == "l?isle jourdain", f] = "l isle jourdain"
    df.loc[df[f] == "oloron", f] = "oloron sainte marie"
    df.loc[df[f] == "curepie", f] = "curepipe"
    df.loc[df[f] == "thiovile", f] = "thionville"
    df.loc[df[f] == "thionvile", f] = "thionville"
    df.loc[df[f] == "neuilly", f] = "neuilly sur seine"
    df.loc[df[f] == "neuilyy sur seine", f] = "neuilly sur seine"
    df.loc[df[f] == "jarville", f] = "jarville la malgrange"
    df.loc[df[f] == "amiens ", f] = "amiens"
    df.loc[df[f] == "amienss ", f] = "amiens"
    df.loc[df[f] == "amien ", f] = "amiens"
    df.loc[df[f] == "champage sur seine", f] = "champagne sur seine"
    df.loc[df[f] == "nuomea", f] = "noumea"
    df.loc[df[f] == "eysine", f] = "eysines"
    df.loc[df[f] == "savenre", f] = "saverne"
    df.loc[df[f] == "la rochellle", f] = "la rochelle"
    df.loc[df[f] == "oullins", f] = "oulins"
    df.loc[df[f] == "carrieres s seine", f] = "carrieres sur seine"
    df.loc[df[f] == "saint denis reunion", f] = "saint denis"
    df.loc[df[f] == "sainte denis", f] = "saint denis"
    df.loc[df[f] == "villefranche sur soane", f] = "villefranche sur saone"
    df.loc[df[f] == "lunel", f] = "lunel viel"
    df.loc[df[f] == "luxeuil", f] = "luxeuil les bains"
    df.loc[df[f] == "moslheim", f] = "molsheim"
    df.loc[df[f] == "sait paul", f] = "saint paul"
    df.loc[df[f] == "champigny", f] = "champigny sur marne"
    df.loc[df[f] == "begerac", f] = "bergerac"
    df.loc[df[f] == "bishheim", f] = "bischheim"
    df.loc[df[f] == "chamber", f] = "chambery"
    df.loc[df[f] == "cotonou (benin)", f] = "cotonou"
    df.loc[df[f] == "courbevois", f] = "courbevoie"
    df.loc[df[f] == "plessy robinson l", f] = "le plessy robinson"
    df.loc[df[f] == "fontenay le compte", f] = "fontenay le comte"
    df.loc[df[f] == "neuville su saone", f] = "neuville sur saone"
    df.loc[df[f] == "villefranche s sur s", f] = "villefranche sur saone"
    df.loc[df[f] == "saint amant les eaux", f] = "saint amand les eaux"
    df.loc[df[f] == "paullac", f] = "pauillac"
    df.loc[df[f] == "bourgnoin joillieu", f] = "bourgoin jallieu"
    df.loc[df[f] == "fontenay s sur s bois", f] = "fontenay sous bois"
    df.loc[df[f] == "champagnol", f] = "champagnole"
    df.loc[df[f] == "louvre", f] = "louvres"
    df.loc[df[f] == "basse terrre", f] = "basse terre"
    df.loc[df[f] == "isle d'abeau", f] = "l isle d abeau"
    df.loc[df[f] == "biscwiller", f] = "bischwiller"
    df.loc[df[f] == "font romeu", f] = "fond romeu odeillo via"
    df.loc[df[f] == "neufhateau", f] = "neufchateau"
    df.loc[df[f] == "cosne sur loire", f] = "cosne cours sur loire"
    df.loc[df[f] == "digne", f] = "digne les bains"
    df.loc[df[f] == "tunis el mahrajene", f] = "tunis"
    df.loc[df[f] == "saint orens", f] = "saint orens de gameville"
    df.loc[df[f] == "la >rochelle", f] = "la rochelle"
    df.loc[df[f] == "aulay sous bois", f] = "aulnay sous bois"
    df.loc[df[f] == "abyme", f] = "pointe a pitre"
    df.loc[df[f] == "reuil", f] = "rueil malmaison"
    df.loc[df[f] == "neuville s sur saone", f] = "neuville sur saone"
    df.loc[df[f] == "neuville sur soane", f] = "neuville sur saone"
    df.loc[df[f] == "villeneueve d'ascq", f] = "villeneuve d ascq"
    df.loc[df[f] == "saint sylvain d'anjou", f] = "saint sylvain d anjou"
    df.loc[df[f] == "this mons", f] = "athis mons"
    df.loc[df[f] == "parentis", f] = "parentis en born"
    df.loc[df[f] == "motigny le bretonneux", f] = "montigny le bretonneux"
    df.loc[df[f] == "montigny le bretonneu", f] = "montigny le bretonneux"
    df.loc[df[f] == "saint laurent sur evre", f] = "saint laurent sur sevre"
    df.loc[df[f] == "perigeux", f] = "perigueux"
    df.loc[df[f] == "petigueux", f] = "perigueux"
    df.loc[df[f] == "bordeau", f] = "bordeaux"
    df.loc[df[f] == "corbeil esonnes", f] = "corbeil essonnes"
    df.loc[df[f] == "agnaux", f] = "agneaux"
    df.loc[df[f] == "wingle", f] = "wingles"
    df.loc[df[f] == "aulnays sous bois", f] = "aulnay sous bois"
    df.loc[df[f] == "guyancourct", f] = "guyancourt"
    df.loc[df[f] == 'saint bieuc', f] = 'saint brieuc'
    df.loc[df[f] == 'cherbourg en conrentin', f] = 'cherbourg en cotentin'
    df.loc[df[f] == 'ormes son sur marne', f] = 'ormesson sur marne'
    df.loc[df[f] == 'dunkerke', f] = 'dunkerque'
    df.loc[df[f] == 'plessis robinson l', f] = 'le pless robon'
    df.loc[df[f] == 'garge les gonesse', f] = 'garges les gonesse'
    df.loc[df[f] == 'bourgnoin jallieu', f] = 'bourgoin jallieu'
    df.loc[df[f] == 'ancenis', f] = 'ancenis saint gereon'
    df.loc[df[f] == 'ribecourt', f] = 'ribecourt dreslincourt'
    df.loc[df[f] == 'cherbourg en corentin', f] = 'cherbourg en cotentin'
    df.loc[df[f] == 'saint luce', f] = 'sainte luce'
    df.loc[df[f] == 'evry', f] = 'evry courcouronnes'
    df.loc[df[f] == 'champ sur marne', f] = 'champs sur marne'
    df.loc[df[f] == 'avesnes sur helpes', f] = 'avesnes sur helpe'
    df.loc[df[f] == 'quimper cdx', f] = 'quimper'
    df.loc[df[f] == 'boulgne sur mer', f] = 'boulogne sur mer'
    df.loc[df[f] == 'le menil esnard', f] = 'le mesnil esnard'
    df.loc[df[f] == 'mortain', f] = 'mortain bocage'
    df.loc[df[f] == 'saint die', f] = 'saint die des vosges'
    df.loc[df[f] == 'cannes la bocca', f] = 'cannes'
    df.loc[df[f] == 'brayenval', f] = 'bray saint aignan'
    df.loc[df[f] == 'aussillou', f] = 'aussillon'
    df.loc[df[f] == 'gujanmestras', f] = 'gujan mestras'
    df.loc[df[f] == 'garnat', f] = 'gannat'
    df.loc[df[f] == 'fourques', f] = 'arles'
    df.loc[df[f] == 'bagnac', f] = 'bagnac sur cele'
    df.loc[df[f] == 'avrec sur loire', f] = 'aurec sur loire'
    df.loc[df[f] == 'gallargue le montueux', f] = 'gallargues le montueux'
    df.loc[df[f] == 'la beausset', f] = 'le beausset'
    df.loc[df[f] == 'fonsurbes', f] = 'fonsorbes'
    df.loc[df[f] == 'laye les mines', f] = 'blaye les mines'
    df.loc[df[f] == 'le cannet (var)', f] = 'le cannet'
    df.loc[df[f] == 'aigues vive ()', f] = 'aigues vives'
    df.loc[df[f] == 'le tailleur medoc', f] = 'le taillan medoc'
    df.loc[df[f] == 'latour bas eine', f] = 'latour bas elne'
    df.loc[df[f] == 'largillay', f] = 'largillay marsonnay'
    df.loc[df[f] == 'la crav ()', f] = 'la crau'
    df.loc[df[f] == 'la lavandou', f] = 'le lavandou'
    df.loc[df[f] == 'la londes les maures', f] = 'la londe les maures'
    df.loc[df[f] == 'la seyne sur mer ()', f] = 'la seyne sur mer'
    df.loc[df[f] == 'la souterraine (creuse)', f] = 'la souterraine'
    df.loc[df[f] == 'la teste', f] = 'la teste de buch'
    df.loc[df[f] == 'lafihevigordane', f] = 'lafitte vigordane'
    df.loc[df[f] == 'lagarrique', f] = 'lagarrigue'
    df.loc[df[f] == 'lapeyrouse fossut', f] = 'lapeyrouse fossat'
    df.loc[df[f] == 'lapeyrouse rossat', f] = 'lapeyrouse fossat'
    df.loc[df[f] == 'fonsegrives', f] = 'quint fonsegrives'
    df.loc[df[f] == 'cateaurenard ()', f] = 'chateaurenard'
    df.loc[df[f] == 'caubon', f] = 'caubon saint sauveur'
    df.loc[df[f] == 'cazaux villecontal', f] = 'cazaux villecomtal'
    df.loc[df[f] == 'chambix', f] = 'chamonix mont blanc'
    df.loc[df[f] == 'chamonix', f] = 'chamonix mont blanc'
    df.loc[df[f] == 'bolleme', f] = 'bollene'
    df.loc[df[f] == 'brou sous chanteraine', f] = 'brou sur chanteraine'
    df.loc[df[f] == 'cabanc seyuenville', f] = 'cabanc seguenville'
    df.loc[df[f] == 'cadoun', f] = 'cadouin'
    df.loc[df[f] == 'cagnes sur mer ()', f] = 'cagnes sur mer'
    df.loc[df[f] == 'braudivy', f] = 'brandivy'
    df.loc[df[f] == 'cambon albi', f] = 'cambon d albi'
    df.loc[df[f] == 'coussay les boas', f] = 'coussay les bois'
    df.loc[df[f] == 'eastelnaudary', f] = 'eastelnaudary'
    df.loc[df[f] == 'entraigues', f] = 'entraigues sur la sorgue'
    df.loc[df[f] == 'evin malmaison (pas de calais )', f] = 'evin malmaison'
    df.loc[df[f] == 'fargeau', f] = 'saint fargeau'
    df.loc[df[f] == 'choiseneuil du poitou', f] = 'chasseneuil du poitou'
    df.loc[df[f] == 'cheingy', f] = 'chaingy'
    df.loc[df[f] == 'cheptorinville', f] = 'cheptainville'
    df.loc[df[f] == 'berk sur mer', f] = 'berck sur mer'
    df.loc[df[f] == 'saitpol de leon', f] = 'saint pol de leon'
    df.loc[df[f] == 'sauveterre (gard)', f] = 'sauveterre'
    df.loc[df[f] == 'sennecy le grand', f] = 'sennecey le grand'
    df.loc[df[f] == 'saint ybais', f] = 'saint ybars'
    df.loc[df[f] == 'saint denis en val (loiret )', f] = 'saint denis en val'
    df.loc[df[f] == 'saint denis en val (loiret)', f] = 'saint denis en val'
	

    


    df.loc[~df[f].isna(), f] = df.loc[~df[f].isna(), f].str.replace(r"la mulatiere(.*)", "la mulatiere", ).str.replace(
        r"orsay(.*)", "orsay").str.replace(r"chen(.*)sur marne", "chennevieres sur marne").str.replace(
        r"orleans(.*)", "orleans").str.replace(r"decines(.*)", "decines charpieu").str.replace(r"douala(.*)",
                                                                                               "douala").str.replace(
        r"valbonne(.*)", "valbonne").str.replace(r"longwy(.*)", "longwy").str.replace(
        r"tournon(.*)", "tournons sur rhone").str.replace(r"saint(.*)etienne(.*)", "saint etienne").str.replace(
        r"beziers(.*)", "beziers").str.replace(r"antibes(.*)", "antibes").str.replace(r'lille(.*)',
                                                                                      r'lille').str.strip()

    print("Nettoyage terminé en {0}s.".format(time() - t0))
    return df


def clean_lycee(df: pd.DataFrame, f: str) -> pd.DataFrame:
    t0 = time()
    print("Nettoyage nom lycée")
    df.loc[~df[f].isna(), f] = df.loc[~df[f].isna(), f].str.lower().apply(unidecode).str.strip().str.replace("-",
                                                                                                             " ").str.replace(
        "'", " ").str.replace(
        r" st ", " saint ").str.replace(r'plyvalent', 'polyvalent').str.replace(r'^st ', 'saint ').str.replace(
        r'profesonnelle', 'professionnel').str.replace(r'lycee des metiers des sciences et des techniques (.*)',
                                                       r'\1').str.replace(
        r'professionnel, (.*)', '\1').str.replace(r"lycee d enseignement general et technologique agricole (.*)",
                                                  r'\1').str.replace(
        r'lycee des metiers de la productique des automatismes et des energies renouvelables (.*)', r'\1').str.replace(
        r'(lycee professionnel)', '').str.replace(
        r'Lycée des métiers de la productique, des automatismes et des énergies renouvelables (.*)', r'\1').str.replace(
        r'lycee des metiers de la logistique et des services (.*)', r'\1').str.replace(
        r'lycee climatique et sportif (.*)', r'\1').str.replace(
        r"lycee des metiers de l ingenierie et des creations industrielles (.*)", r'\1').str.replace(
        r'lycee des metiers d art et de la maitrise de l energie electrique (.*)', r'\1').str.replace(
        r'lycee professionnel de la communication et industries (.*)', r'\1').str.replace(
        r'lycee des metiers de la gestion d energie et des process (.*)', r'\1').str.replace(
        r'lycee des metiers de l optique (.*)', r'\1').str.replace("ins.", "").str.replace(r'(.*) site(.*)',
                                                                                           r'\1').str.replace(
        r'(.*) site principal(.*)', r'\1').str.replace(r'cfa (.*)', r'\1').str.replace(
        r'lycee professionnel regional (.*)', r'\1').str.replace(
        "institut", "").str.replace("centre scolaire", "").str.replace(r'lycer', 'lycee').str.replace("cedex",
                                                                                                      "").str.replace(
        "lycee", "").str.replace("lg", "").str.replace("^lp$", "").str.replace("eic", "").str.replace(" ste ",
                                                                                                      " sainte ").str.replace(
        " ins ", " ").str.replace(
        " ntre ", " notre").str.replace("enseignement general", "").str.replace("general", "").str.replace("hoteliers",
                                                                                                           "hotelier").str.replace(
        "enseignement professionnel", "professionnel").str.replace(r'lycee professionnel (.*)', '\1').str.replace(
        "international", "").str.replace("da??etat", "").str.replace("pasteurs", "pasteur").str.replace(
        "prive ", " ").str.replace("ensemble", "").str.replace("college", "").str.replace("polyvalent", "").str.replace(
        "ecole", "").str.replace("lyee", "").str.replace(
        "institution", "").str.replace("cned rennes", "cned").str.replace("general et technologique", "").str.replace(
        "privee", "").str.replace("mixte", "").str.replace(
        "moderne", "").str.replace("bilingue", "").str.replace("active", "").str.replace("profesionnel",
                                                                                         "professionnel").str.replace(
        "scolaire", "").str.replace(
        "professionel", "professionnel").str.replace("sartres", "sartre").str.replace("cegep", "").str.replace(
        "des metiers", "").str.replace("&", "et").str.replace(r'et professionnel (.*)', r'\1').str.replace(
        r'lp (.*)', r'\1').str.replace(r'^lpo (.*)', r'\1').str.replace(r'public (.*)', r'\1').str.replace(
        r'externat (.*)', '\1').str.replace(r'^lt (.*)', r'\1').str.replace(r'^ltp (.*)', r'\1').str.replace(
        "technique de monaco", 'technique et hotelier de monaco').str.replace("technique hotelier de monaco",
                                                                              'technique et hotelier de monaco').str.replace(
        "lasalle", "la salle").str.replace("baptiste la salle", "baptiste de la salle").str.replace("doubs",
                                                                                                    "pontarlier").str.replace(
        "w.a.", "wolfgang amadeus ").str.replace("moazrt", "mozart").str.replace("wingles", "wingle").str.replace(
        "violet le", "viollet le").str.replace(
        "bar?ul", "baroeul").str.replace("billingue", "").str.replace("dumas aer", "dumas").str.replace("agricole",
                                                                                                        "").str.replace(
        "|", "l").str.replace(
        "17026", "la rochelle").str.replace("/", "").str.replace(":", "").str.replace("seminaire de walbourg",
                                                                                      "seminaire des jeunes").str.replace(
        "saint andree", "saint andre").str.replace("schuman haguenau", "schuman").str.replace("guemar",
                                                                                              "colmar").str.replace(
        "pascal colmar", "pascal").str.replace(
        "schweizer", "schweitzer").str.replace("superieure", "").str.replace("vauvenargues", "vauvenargue").str.replace(
        "t.aubanel", "theodore aubanel").str.replace(
        "technologique", "").str.replace("superieur", "").str.replace("gignac la nerthe", "gignac").str.replace(
        "a. rimbaud", "arthur rimbaus").str.replace(
        "a dam de", "adam de").str.replace("joesph", "joseph").str.replace("hugo caen", "hugo").str.replace(
        "cherbourg octeville", "cherbourg en cotentin").str.replace(
        "cdx", "").str.replace(
        "assomption retiers", "assomption").str.replace("a. r. lesage", "alain rene lesage").str.replace(
        "hugo marrakech", "hugo").str.replace("hugo lunel", "hugo").str.replace(
        "dumas, aer", "dumas").str.replace("stanislas sacre coeur", "stanislas").str.replace("a. camus",
                                                                                             "albert camus").str.replace(
        "de lasalle", "de la salle").str.replace(
        "amiral bouvet", "amiral pierre bouvet").str.replace("catholique", "").str.replace("l isle",
                                                                                           "isle").str.replace(
        "mezieres", "meziere").str.replace(
        "dominique nancy", "dominique").str.replace(
        "thionbille", "thionville").str.replace("sccwartz", "schwartz").str.replace("saint jean baptiste",
                                                                                    "jean baptiste").str.replace(
        "batiste", "baptiste").str.replace(
        "lamache", "la mache").str.replace("maurice la mache", "la mache").str.replace("simone weil",
                                                                                       "simone veil").str.replace(
        "alexie", "alexis").str.replace(
        "\"", "").str.replace(
        "a. renoir", "auguste renois").str.replace("sud medoc la boetie", "sud medoc").str.replace("le taillan",
                                                                                                   "taillan").str.replace(
        "saint aubin de me", "taillan").str.replace(
        "kaslter", "kastler").str.replace("hugo ?", "hugo").str.replace("hessel jolimont", "hessel").str.replace(
        "d avila", "avila").str.replace(
        "george brassens", "georges brassens").str.replace(
        "plateau cailloux", "saint paul").str.replace("(", "").str.replace(")", "").str.replace(
        "baudimont saint charles", "baudimont").str.replace(
        "baudimon saint charles", "baudimont").str.replace("w.a", "wolfgang amadeus ").str.replace(
        "vanh gogh", "van gogh").str.replace("scheitzer", "schweitzer").str.replace("camus, colombes",
                                                                                    "camus, bois colombes").str.replace(
        "le raincy", "le raincy").str.replace(
        ",vesinet", ",le vesinet").str.replace("sepr, france", ",sepr, lyon").str.replace("ampere site saxe",
                                                                                          "ampere").str.replace(
        "ampere saxe", "ampere").str.replace(
        "ampere bourse", "ampere").str.replace("eiffel bordeaux", "eiffel").str.replace("parentis en born",
                                                                                        "parentis").str.replace(
        "perigeux", "perigueux").str.replace(
        "petigueux", "perigueux").str.replace("des eucalyptus", "les eucalyptus").str.replace(r"liceo frances.*",
                                                                                              "frances de barcelona").str.replace(
        "francais de barcelone", "frances de barcelona").str.replace(
        r".*francais.*bon soleil", "francais bon soleil").str.replace(r"^\s*vieljeux.*", "leonce vieljeux").str.replace(
        r"^st ", "saint ").str.replace(r"^o ", "").str.replace(
        r"^ion ", "").str.replace(r"\s+", " ").str.replace(r"\s$", "").str.replace(r"^\s", "").str.replace(r"[0-9]+",
                                                                                                           "").str.replace(
        r"^ih ", "").str.replace(
        r"^et ", "").str.replace(
        r"^la salle ?l?y?o?n?$", "jean baptiste de la salle").str.replace(r"\s$", "").str.replace(r"^\s",
                                                                                                  "").str.replace(
        r"\si$ ", "").str.replace(r"^t ", "").str.replace(
        r"^ins ", "").str.replace(r"^ste ", "sainte ").str.replace(r"^balzac", "honore de balzac").str.replace(
        r"^buffon", "georges louis de buffon").str.replace(
        r"^diderot", "rene diderot").str.replace(
        r"^professionnel ", "").str.replace(r"^descartes", "rene descartes").str.replace(r"^champagnat",
                                                                                         "marcellin champagnat").str.replace(
        r"^itut", "institut").str.replace(
        r"damesaint", "dame saint").str.replace(r"^rodin", "auguste rodin").str.replace(r"secondare ", "").str.replace(
        r"^condorcet", "nicolas de condorcet").str.replace(
        r"^kirschleger", "frédéric kirschleger").str.replace(r"^hersnesaint", "ernest").str.replace(r".*fanon",
                                                                                                    "franz fanon").str.replace(
        r"^jb vatelot", "jean baptiste vatelot").str.replace(r"^legta ", "").str.replace(r"^ets ", "").str.replace(
        r"painlevee", "painleve").str.replace(
        r"moulin beziers", "moulin").str.replace(r"^pierre gilles de genne$", "pierre gilles de gennes").str.replace(
        r"^lumiere", "louis lumiere").str.replace(r"e. branly", "edouard branly").str.replace(r"^ampere",
                                                                                              "andre marie ampere").str.replace(
        r"vauvenargue", "vauvenargues").str.replace(r"^l.p. ", "").str.replace(
        r"^s felix", "felix").str.replace(r"therese avila", "therese d avila").str.replace(r"h.balzac",
                                                                                           "honore de balzac").str.replace(
        r"^de madame", "madame").str.replace(
        r"^n.$", "").str.replace(r"d\?angers$", "d angers").str.replace(r" prive", "").str.replace(r"technique",
                                                                                                   "").str.replace(
        r"l\?europe", "l europe").str.replace(
        r"^e\.? ", "").str.replace(
        r"^l\?$", "").str.replace(r"professionnelle guy$", "georges guy").str.replace(r"j.amyot",
                                                                                      "jacques amyot").str.replace(
        r"f.j. talma", "francois joseph talma").str.replace(r"^esj ", "").str.replace(r"sepr lyon ", "").str.replace(
        r"lt $", "").str.replace(r"^s ", "").str.replace(
        r"^institut ", "").str.replace(r"^municipal ", "").str.replace(r"^c.i.t. ", "").str.replace(r"^pro ",
                                                                                                    "").str.replace(
        r"^jb de la", "jean baptiste de la").str.replace(r"^edta", "").str.replace(r"^mangin",
                                                                                   "charles mangin").str.replace(
        r"marie, beaucamp", "marie").str.replace(r"w.\sa.$", "wolfgang amadeus").str.replace(r"^d et technique ",
                                                                                             "").str.replace(r"^l ",
                                                                                                             "").str.replace(
        r"^licp ", "").str.replace(r"jacques audiberti", "audiberti").str.replace(r"frederique et irene joliot curie",
                                                                                  "joliot curie").str.replace(
        r"albert eein", "albert einstein").str.replace(
        r"albert er", "albert 1er").str.replace(r"(.*)audiberti", "jacques audiberti").str.replace(
        r"charles de foucau(.*)", "charles de foucauld").str.replace(r"du mont blanc rene dayve",
                                                                     "mont blanc rene dayve").str.replace(
        r"ebtp", "claude nicolas ledoux ebtp").str.replace(r"ernesaint couteaux", "ernest couteaux").replace(
        r"eein", "albert einstein").str.replace(r"evariste gallois", "evariste galois").str.replace(
        r"frnacois er", "francois 1er").str.replace(r"francois premier", "francois 1er").str.replace(
        r"germaine tillon", "germaine tillion").str.replace(r"henry matisse", "henri matisse").str.replace(
        r"henry wallon", "henri wallon").str.replace(r"jules vernes", "jules verne").str.replace(
        r"la joliverie", "saint pierre la joliverie").str.replace(r"lecorbusier", "le corbusier").str.replace(
        r"mme de stael", "madame de stael").str.replace(r"nazareth", "nazareth haffreingue").str.replace(
        r"nicolas de condorcert schoeneck", "nicolas de condorcert").str.replace(r"nicolas de condorcert",
                                                                                 "condorcet").str.replace(
        r"ort lyon", "ort").str.replace(r"parc de vienis", "parc de vilgenis").str.replace(
        r"pierre mendes france tunis", "pierre mendes france").str.replace(r"institution (.*)", r"\1").str.replace(
        r"renee cassin", "rene cassin").str.replace(r"sacre c?ur", "sacre coeur").str.replace(
        r"saint joseph vannes", "saint joseph lasalle").str.replace(r'(.*) voie e et', r"\1").str.replace(
        r'jean moulin draguignan', 'jean moulin').str.replace(r'section d professionnel (.*)', r'\1').str.replace(
        r'arbez carmes', 'arbez carme').str.replace(r'francisque sarcey', 'nikola tesla').str.replace(
        r'marie madelaine fourcade', 'marie madeleine fourcade').str.replace(
        r'chaplin becquerel', 'charlie chaplin').str.replace(r'charles de gaule', 'charles de gaulle').str.replace(
        r'montjoux', 'jules haag').str.replace(r'^caousou', 'le caousou').str.replace(r'hernesaint hemingway',
                                                                                      'hernest hemingway').str.replace(
        r'prytanee militaire', 'prytanee national militaire').str.replace(r'notre dame des missions',
                                                                          'notre dame des missions saint pierre').str.replace(
        r'gabriel touchard', 'gabriel touchard washington').str.replace(r'sacre coeur de la perverie',
                                                                        'la perverie').str.replace(
        r'e.livet', 'livet').str.replace(r'legt andre maurois', 'andre maurois').str.replace(r'labriquerie',
                                                                                             'la briquerie').str.replace(
        r'costebnelle', 'costebelle').str.replace("^pierre rouge$", "saint joseph pierre rouge").str.replace(
        r'^vincensini$', 'paul vincensini').str.replace(
        r'o costebelle', 'costebelle').str.replace(r'paul le roland', 'paul le rolland').str.replace(r"^coubertin$",
                                                                                                     "pierre de coubertin").str.replace(
        r'maillard joubert', 'joubert emilien maillard').str.replace(r'jeanne d arc bastia',
                                                                     'jeanne d arc').str.replace(
        r'notre dame de la providece', 'notre dame de la providence').str.replace(r'de la croix blanche',
                                                                                  'la croix blanche').str.replace(
        r'les bourdonieres', 'les bourdonnieres').str.replace(r'leonard de vincie', 'leonard de vinci').str.replace(
        r'vaunargues', 'vauvenargues').str.replace(
        r'(.*)niepce(.*)balleur', 'niepce balleure').str.replace(r'alexis de toqueville',
                                                                 'alexis de tocqueville').str.replace(
        r'sainte marie la grande grange', 'sainte marie la grand grange').str.replace(r'saint famille saintonge',
                                                                                      'sainte famille saintonge').str.replace(
        r'joliot cuire', 'joliot curie').str.replace(r'jean baptiste de la salle(.*)',
                                                     'jean baptiste de la salle').str.replace(
        r'(.*)mirepoix(.*)', 'de mirepoix').str.replace(r'la salle les francs bourgeois',
                                                        'des francs bourgeois la salle').str.replace(
        r'^monteil$', 'alexis monteil').str.replace(r'', '').str.replace(r'prives saint pierre chanel',
                                                                         'saint pierre chanel').str.replace(
        r'^beau de rochas$', 'alphonse beau de rochas').str.replace(r'^duruy$', r'victor duruy').str.replace(
        r'de tois bassins', 'de trois bassins').str.replace(
        r'guilaume fichet', 'guillaume fichet').str.replace(r'^martiniere monplaisir$',
                                                            'la martiniere montplaisir').str.replace(
        r'blanche de castille(.*)', 'blanche de castille').str.replace(
        r'cned de rennes', 'alpha rennes').str.replace(r'ituion la croix blanche', 'la croix blanche').str.replace(
        r'^saint nicolas lassale$', 'la salle saint nicolas').str.replace(
        r'du sacree coeur', 'du sacre coeur').str.replace(r'charle de foucauld', 'charles de foucauld').str.replace(
        r'saint joseph la malassise', 'la malassise').str.replace(
        r'ogec la salle saint nicolas', 'la salle saint nicolas').str.replace(r'george de la tour',
                                                                              'georges de la tour').str.replace(
        r'aux lazaristes la salle', 'aux lazaristes').str.replace(r'jean ghenno', 'jean guehenno').str.replace(
        r'notre dame la merci', 'notre dame de la merci').str.replace(
        r'notre dame de sannois', 'notre dame').str.replace(r'fernand darchcourt', 'fernand darchicourt').str.replace(
        r'assomption rennes', 'assomption').str.replace(
        r'lp pierre lescot', 'pierre lescot').str.replace(r'', '').str.replace(r'frédéric kirschleger',
                                                                               'frederic kirschleger').str.replace(
        r'^carmejane', 'de digne carmejane').str.replace(r'elysee reclus', 'elisee reclus').str.replace(
        r'de bellepierre', 'bellepierre').str.replace(
        r'du parc des loges', 'parc des loges').str.replace(r'stanislas cannes', 'stanislas').str.replace(
        r'cite du parc imperial', 'du parc imperial').str.replace(
        r'renee descartes', 'rene descartes').str.replace(r' dorian', 'dorian').str.replace(r'jean baptiste kleber',
                                                                                            'kleber').str.replace(
        r'du gresivaudan (.*)', 'du gresivaudan').str.replace(r'saint jeanne elizabteh',
                                                              'sainte jeanne elisabeth').str.replace(
        r'jean sturm', 'gymnase jean sturm').str.replace(r'louise weiss acheres', 'louise weiss').str.replace(
        r'de lons le saunier edgar faure', 'edgar faure').str.replace(
        r'paul languevin', 'paul langevin').str.replace(r'jacques coeurs', 'jacques coeur').str.replace(
        r'francois rabelais', 'rabelais').str.replace(
        r'frederic bazille (.*)', 'frederic bazille').str.replace(r'^saint francois de s',
                                                                  'saint francois de sales').str.replace(
        r'les ctalins', 'les catalins').str.replace(r'notre dame du mur', 'notre dame du mur le porsmeur').str.replace(
        r'^notre dame de bonnes nouvelles$', 'notre dame de bonnes nouvelles dom sortais').str.replace(
        r'de chateauboeuf', 'chateauboeuf').str.replace(r'ferdinant ', 'ferdinand').str.replace(r'villa pia',
                                                                                                'saint louis villa pia').str.replace(
        r'henry loritz', 'henri loritz').str.replace(r'eifflel', 'eiffel').str.replace(r'charles de gaulles',
                                                                                       'charles de gaulle').str.replace(
        r'victor hugo caen', 'victor hugo').str.replace(
        r'bourdelle', 'antoine bourdelle').str.replace(' saint nicolas', 'saint nicolas').str.replace(r'cour peret',
                                                                                                      'cours peret bordeaux').str.replace(
        r'des pontonniers', 'les pontonniers').str.replace(r'jule froment', 'jules froment').str.replace(
        r'eugeune ionesco', 'eugene ionesco').str.replace(
        r'^folie saint james$', 'la folie saint james').str.replace(r'bertrand de born', 'bertran de born').str.replace(
        r'fred esteve', 'frederic esteve').str.replace(
        r'labruyere', 'la bruyere').str.replace(r'auguste mariette', 'mariette').str.replace(r'rene cassion',
                                                                                             'rene cassin').str.replace(
        r'latecoere', 'pierre latecoere').str.replace(r'jeanine manuel', 'jeannine manuel').str.replace(r'^talensac$',
                                                                                                        'talensac jeanne bernard').str.replace(
        r'saint michel de picpus de paris', 'saint michel de picpus').str.replace(r'marc seguin saint charles',
                                                                                  'marc seguin').str.replace(
        r'fustel de coulange', 'fustel de coulanges').str.replace(r'edisson', 'edison').str.replace(r'joliet curie',
                                                                                                    'joliot curie').str.replace(
        r'toseph', 'joseph').str.replace(r'loius armand', 'louis armand').str.replace(r'saint louis crest',
                                                                                      'saint louis').str.replace(
        r'ros glas', 'roz glas').str.replace(r'nicolas de condorcert belfort', 'nicolas de condorcet').str.replace(
        r'leonard de vinci hqe', 'leonard de vinci').str.replace(
        r'francois philibert dessaignes', 'philibert dessaignes').str.replace(r'la chataignerais',
                                                                              'la chataigneraie').str.replace(
        r' la chataigneraie', 'la chataigneraie').str.replace(
        r'deodat de severac toulouse', 'deodat de severac').str.replace(r'du val dargens',
                                                                        'du val d argens').str.replace(
        r'sainte marie d antony', 'sainte marie').str.replace(
        r'balzax', 'balzac').str.replace(r'andre malraux gaillon', 'andre malraux').str.replace(
        r'saint joseph bressuire', 'saint joseph').str.replace(
        r'notre dame, reze', 'notre dame').str.replace(r'saint joseph thones', 'saint joseph').str.replace(
        r'm.r.e. sully', 'sully').str.replace(
        r'notre dame le mans', 'notre dame').str.replace(r'j.b. corot', 'jean baptiste corot').str.replace(r'', '')

    print("Nettoyage terminé en {0}s.".format(time() - t0))
    return df


def clean_commune_lycee(df: pd.DataFrame, f: str, ff: str) -> pd.DataFrame:
    t0 = time()
    print("Nettoyage nom ville en fonction du lycee")
    df.loc[(df[f] == "") & (df[ff] == "albert claveille"), f] = "perigueux"
    df.loc[(df[f] == "saintes") & (df[ff] == "albert einstein"), f] = "sainte genevieve des bois"
    df.loc[(df[f] == "la seyne") & (df[ff] == "beaussier"), f] = "la seyne sur mer"
    df.loc[(df[f] == "la reunion") & (df[ff] == "bellepierre"), f] = "saint denis"
    df.loc[(df[f] == "le bois robert") & (df[ff] == "bois robert"), f] = "becon les granits"
    df.loc[(df[f] == "") & (df[ff] == "complexe sainte felicite"), f] = "cotonou"
    df.loc[(df[f] == "argelia") & (df[ff] == "cours soleil"), f] = "alger"
    df.loc[(df[f] == "") & (df[ff] == "du rempart"), f] = "marseille"
    df.loc[(df[f] == "akanda") & (df[ff] == "elise marie"), f] = "libreville"
    df.loc[(df[f] == "saintes") & (df[ff] == "elisee reclus"), f] = "sainte foy la grande"
    df.loc[(df[f] == "") & (df[ff] == "ernest couteaux"), f] = "saint amand les eaux"
    df.loc[(df[f] == "granollers") & (df[ff] == "escola pia granollers"), f] = "barcelona"
    df.loc[(df[f] == "") & (df[ff] == "eucalyptus"), f] = "nice"
    df.loc[(df[f] == "cesson") & (df[ff] == "frederic ozanam"), f] = "cesson sevigne"
    df.loc[(df[f] == "brives") & (df[ff] == "georges cabanis"), f] = "brive la gaillarde"
    df.loc[(df[f] == "labaule") & (df[ff] == "grand air"), f] = "la baule escoublac"
    df.loc[(df[f] == "joue les tours") & (df[ff] == "grandmont"), f] = "tours"
    df.loc[(df[f] == "avesnes") & (df[ff] == "jesse de forest"), f] = "avesnes sur helpe"
    df.loc[(df[f] == "saint sebestien") & (df[ff] == "la baugerie"), f] = "saint sebastien sur loire"
    df.loc[(df[f] == "") & (df[ff] == "la chataigneraie"), f] = "le mesnil esnard"
    df.loc[(df[f] == "") & (df[ff] == "la salle passy buzenval"), f] = "rueil malmaison"
    df.loc[(df[f] == "saintes") & (df[ff] == "le verger"), f] = "sainte marie"
    df.loc[(df[f] == "") & (df[ff] == "leonce vieljeux"), f] = "la rochelle"
    df.loc[(df[f] == "saint maximin") & (df[ff] == "maurice janeti"), f] = "saint maximin la sainte baume"
    df.loc[(df[f] == "") & (df[ff] == "mireille grenet"), f] = "compiegne"
    df.loc[(df[f] == "") & (df[ff] == "montebello"), f] = "lille"
    df.loc[(df[f] == "wattignies") & (df[ff] == "montebello"), f] = "lille"
    df.loc[(df[f] == "saint martin boulogne") & (df[ff] == "nazareth affreingue"), f] = "boulogne sur mer"
    df.loc[(df[f] == "saint martin") & (df[ff] == "nazareth affreingue"), f] = "boulogne sur mer"
    df.loc[(df[f] == "stiring wendel") & (df[ff] == "condorcet"), f] = "schoeneck"
    df.loc[(df[f] == "verneuil") & (df[ff] == "notre dame les oiseaux"), f] = "verneuil sur seine"
    df.loc[(df[f] == "massy palaiseau") & (df[ff] == "parc de vilgenis"), f] = "massy"
    df.loc[(df[f] == "la seyne") & (df[ff] == "paul langevin"), f] = "la seyne sur mer"
    df.loc[(df[f] == "") & (df[ff] == "raponda walker"), f] = "port gentil"
    df.loc[(df[f] == "ribeaupierre") & (df[ff] == "ribeaupierre"), f] = "ribeauville"
    df.loc[(df[f] == "") & (df[ff] == "rocroy saint vincent de paul"), f] = "paris"
    df.loc[(df[f] == "") & (df[ff] == "saint françois xavier"), f] = "vannes"
    df.loc[(df[f] == "montreal de l'aude") & (df[ff] == "saint joseph des carmes"), f] = "montreal"
    df.loc[(df[f] == "carnoles") & (df[ff] == "saint joseph carnoles"), f] = ""
    df.loc[(df[f] == "") & (df[ff] == "saint joseph lasalle"), f] = "vannes"
    df.loc[(df[f] == "caussade") & (df[ff] == "claude nougaro"), f] = "monteils"
    df.loc[(df[f] == "montgermont") & (df[ff] == "de la salle"), f] = "rennes"
    df.loc[(df[f] == "guemar") & (df[ff] == "camille see"), f] = "colmar"
    df.loc[(df[f] == "la riviere") & (df[ff] == "jean joly"), f] = "saint louis"
    df.loc[(df[f] == "sainte clotilde") & (df[ff] == "lislet geoffroy"), f] = "saint denis"
    df.loc[(df[f] == "villejuif") & (df[ff] == "de cachan"), f] = "cachan"
    df.loc[(df[f] == "saint quentin en yvelines") & (df[ff] == "montigny le bretonneux"), f] = ""
    df.loc[(df[f] == "saint sebastien") & (df[ff] == "la baugerie"), f] = "saint sebastien sur loire"
    df.loc[(df[f] == "ormesson sur marne") & (df[ff] == "champlain"), f] = "chennevieres sur marne"
    df.loc[(df[f] == "saint martin") & (df[ff] == "giraux sannier"), f] = "saint martin boulogne"
    df.loc[(df[f] == "woippy") & (df[ff] == "louis de cormontaigne"), f] = "metz"
    df.loc[(df[f] == "petite rosselle") & (df[ff] == "saint joseph la providence"), f] = "forbach"
    df.loc[(df[f] == "le vigan") & (df[ff] == "dhuoda"), f] = "nimes"
    df.loc[(df[f] == "sainte clotilde") & (df[ff] == "leconte de lisle"), f] = "saint denis"
    df.loc[(df[f] == "") & (df[ff] == ""), f] = ""
    df.loc[(df[f] == "") & (df[ff] == ""), f] = ""
    print("Nettoyage terminé en {0}s.".format(time() - t0))
    return df


def clean_lycee_with_commune(df: pd.DataFrame, f: str, ff: str) -> pd.DataFrame:
    t0 = time()
    df.loc[(df[f] == "angers") & (df[ff] == "renoir"), ff] = "auguste et jean renoir"
    df.loc[(df[f] == "asnieres sur seine") & (df[ff] == "renoir"), ff] = "auguste renoir"
    df.loc[(df[f] == "cagnes sur mer") & (df[ff] == "renoir"), ff] = "auguste renoir"
    df.loc[(df[f] == "vannes") & (df[ff] == "saint joseph vannes"), ff] = "saint joseph lasalle"
    df.loc[(df[f] == "bordeaux") & (df[ff] == "saint genes"), ff] = "saint genes la salle"
    df.loc[(df[f] == "nice") & (df[ff] == "cfa pauliani"), ff] = "antenne vauban du cfa ran"
    df.loc[(df[f] == "cachan") & (df[ff] == "gustave effeil"), ff] = "de cachan"
    df.loc[(df[f] == "montpellier") & (df[ff] == "mermoz"), ff] = "jean mermoz"
    df.loc[(df[f] == "peltre") & (df[ff] == "notre dame de peltre"), ff] = "notre dame"
    df.loc[(df[f] == "saint die des vosges") & (df[ff] == "beaumont"), ff] = "georges baumont"
    df.loc[(df[f] == "saint brieuc") & (df[ff] == "sacre c?ur"), ff] = "sacre coeur la salle"
    df.loc[(df[f] == "verdun") & (df[ff] == "marguerrite galland"), ff] = "jean auguste margueritte"
    df.loc[(df[f] == "sainte marie") & (df[ff] == "leverger"), ff] = "le verger"
    df.loc[(df[f] == "les avirons") & (df[ff] == "de saint exupery"), ff] = "antoine de saint exupery"
    df.loc[(df[f] == "nantes") & (df[ff] == "g. monge"), ff] = "gaspard monge la chauviniere"
    df.loc[(df[f] == "sevres") & (df[ff] == "vernant"), ff] = "jean pierre vernant"
    df.loc[(df[f] == "derval") & (df[ff] == "saint clair"), ff] = "saint clair blain derval"
    df.loc[(df[f] == "paris") & (df[ff] == "saint pierre fourrier"), ff] = "eugene napoleon saint pierre fourier"
    df.loc[(df[f] == "paris") & (df[ff] == "saint pierre fourier"), ff] = "eugene napoleon saint pierre fourier"
    df.loc[(df[f] == "saint ouen") & (df[ff] == "blanqui"), ff] = "auguste blanqui"
    df.loc[(df[f] == "arcachon") & (df[ff] == "grand air"), ff] = "de grand air"
    df.loc[(df[f] == "lyon") & (df[ff] == "saint juste"), ff] = "de saint juste"
    df.loc[(df[f] == "tourcoing") & (df[ff] == "sacre coeur"), ff] = "du sacre coeur"
    df.loc[(df[f] == "valenciennes") & (df[ff] == "antoine watteau"), ff] = "watteau"
    df.loc[(df[f] == "mantes la jolie") & (df[ff] == "antoine de saint exupery"), ff] = "saint exupery"
    df.loc[(df[f] == "sete") & (df[ff] == "joliot curie"), ff] = "irene et frederic joliot curie"
    df.loc[(df[f] == "saint brieuc") & (df[ff] == "francois rabelais"), ff] = "rabelais"
    df.loc[(df[f] == "la verpilliere") & (df[ff] == "sainte marie lyon"), ff] = "sainte marie"
    df.loc[(df[f] == "montauban") & (df[ff] == "michelet"), ff] = "jules michelet"
    df.loc[(df[f] == "albi") & (df[ff] == "rascol"), ff] = "louis rascol"
    df.loc[(df[f] == "meudon") & (df[ff] == "francois rabelais"), ff] = "rabelais"
    df.loc[(df[f] == "luneville") & (df[ff] == "bichat"), ff] = "ernest bichat"
    df.loc[(df[f] == "fameck") & (df[ff] == "saint exupery"), ff] = "antoine de saint exupery"
    df.loc[(df[f] == "gardanne") & (df[ff] == "m.m. fourcade"), ff] = "fourcade l etoile"
    df.loc[(df[f] == "lorient") & (df[ff] == "colbert"), ff] = "jean baptiste colbert"
    df.loc[(df[f] == "guyancourt") & (df[ff] == "villaroy"), ff] = "de villaroy"
    df.loc[(df[f] == "belfort") & (df[ff] == "gustave courbet"), ff] = "courbet"
    df.loc[(df[f] == "sarcelles") & (df[ff] == "la tourelle"), ff] = "de la tourelle"
    df.loc[(df[f] == "angers") & (df[ff] == "joseph wresi itec"), ff] = "joseph wresinski"
    df.loc[(df[f] == "angers") & (df[ff] == "saint serge"), ff] = "joseph wresinski"
    df.loc[(df[f] == "nice") & (df[ff] == "calmette"), ff] = "albert calmette"
    df.loc[(df[f] == "colomiers") & (df[ff] == "ort"), ff] = "ort maurice grynfogel"
    df.loc[(df[f] == "andrezieux boutheon") & (df[ff] == "francois mauriac"), ff] = "francois mauriac forez"
    df.loc[(df[f] == "brioude") & (df[ff] == "lafayette"), ff] = "la fayette"
    df.loc[(df[f] == "mulhouse") & (df[ff] == "lavoisier"), ff] = "laurent de lavoisier"
    df.loc[(df[f] == "tours") & (df[ff] == "vaucanson"), ff] = "jacques de vaucanson"
    df.loc[(df[f] == "la celle saint cloud") & (df[ff] == "pierre"), ff] = "pierre corneille"
    df.loc[(df[f] == "villemomble") & (df[ff] == "george clemenceau"), ff] = "clemenceau"
    df.loc[(df[f] == "valenciennes") & (df[ff] == "wallon"), ff] = "henri wallon"
    df.loc[(df[f] == "nantes") & (df[ff] == "gaspard monge"), ff] = "gaspard monge la chauviniere"
    df.loc[(df[f] == "la roche sur yon") & (df[ff] == "alfred kastler guitton"), ff] = "rosa parks"
    df.loc[(df[f] == "dax") & (df[ff] == "borda"), ff] = "de borda"
    df.loc[(df[f] == "quimperle") & (df[ff] == "de kerneuzec"), ff] = "kerneuzec"
    df.loc[(df[f] == "sarcelles") & (df[ff] == "la tourelles"), ff] = "de la tourelle"
    df.loc[(df[f] == "") & (df[ff] == ""), ff] = ""
    df.loc[(df[f] == "") & (df[ff] == ""), ff] = ""
    df.loc[(df[f] == "") & (df[ff] == ""), ff] = ""
    df.loc[(df[f] == "") & (df[ff] == ""), ff] = ""
    print("Nettoyage terminé en {0}s.".format(time() - t0))
    return df


def clean_nat(df: pd.DataFrame, f: str) -> pd.DataFrame:
    print("Nettoyage nom nationalité")
    df.loc[~df[f].isna(), f] = df.loc[~df[f].isna(), f].str.lower().apply(unidecode).str.strip().str.replace("-",
                                                                                                             " ").str.replace(
        "'", " ")
    df.loc[~df[f].isna(), f] = df.loc[~df[f].isna(), f].apply(
        lambda x: x + "ne" if len(x) > 3 and x[-2:] == "en" else x)
    df.loc[~df[f].isna(), f] = df.loc[~df[f].isna(), f].apply(
        lambda x: x + "nne" if len(x) > 3 and x[-2:] == "ie" else x)
    df.loc[~df[f].isna(), f] = df.loc[~df[f].isna(), f].apply(
        lambda x: x + "enne" if len(x) > 3 and x[-2:] in ["ay", "ti"] else x)
    df.loc[~df[f].isna(), f] = df.loc[~df[f].isna(), f].apply(
        lambda x: x + "e" if len(x) > 4 and x[-3:] in ["ain", "ais", "nol", "ois", "hiz", "akh", "var", "and"] else x)
    df.loc[df[f] == "france", f] = 'francaise'
    df.loc[df[f] == "cameroun", f] = 'camerounaise'
    df.loc[df[f] == "cameroune", f] = 'camerounaise'
    df.loc[df[f] == "cameounaise", f] = 'camerounaise'
    df.loc[df[f] == "portugal", f] = 'portugaise'
    df.loc[df[f] == "portuguaise", f] = 'portugaise'
    df.loc[df[f] == "roumanienne", f] = 'roumaine'
    df.loc[df[f] == "kazakhstan", f] = 'kazakhe'
    df.loc[df[f] == "moneguasque", f] = 'monegasque'
    df.loc[df[f] == "senegal", f] = 'senegalaise'
    df.loc[df[f] == "britanique", f] = 'britannique'
    df.loc[df[f] == "anglaise", f] = 'britannique'
    df.loc[df[f] == "royaume uni", f] = 'britannique'
    df.loc[df[f] == "maroc", f] = 'marocaine'
    df.loc[df[f] == "russie", f] = 'russe'
    df.loc[df[f] == "francise", f] = 'francaise'
    df.loc[df[f] == "cote d ivoire", f] = 'ivoirienne'
    df.loc[df[f] == "fancaise", f] = 'francaise'
    df.loc[df[f] == "mali", f] = 'malienne'
    df.loc[df[f] == "madagascar", f] = 'malgache'
    df.loc[df[f] == "franccaise", f] = 'francaise'
    df.loc[df[f] == "fran", f] = 'francaise'
    df.loc[df[f] == "fra", f] = 'francaise'
    df.loc[df[f] == "fr", f] = 'francaise'
    df.loc[df[f] == "f", f] = 'francaise'
    df.loc[df[f] == "francaie", f] = 'francaise'
    df.loc[df[f] == "frnacaise", f] = 'francaise'
    df.loc[df[f] == "francaisz", f] = 'francaise'
    df.loc[df[f] == "francaaise", f] = 'francaise'
    df.loc[df[f] == "francaice", f] = 'francaise'
    df.loc[df[f] == "francaiqe", f] = 'francaise'
    df.loc[df[f] == "francoise", f] = 'francaise'
    df.loc[df[f] == "rouenne", f] = 'francaise'
    df.loc[df[f] == "paris 17eme", f] = 'francaise'
    df.loc[df[f] == "martenique", f] = 'francaise'
    df.loc[df[f] == "fran,caise", f] = 'francaise'
    df.loc[df[f] == "fraicaise", f] = 'francaise'
    df.loc[df[f] == "francaises", f] = 'francaise'
    df.loc[df[f] == "mayotte", f] = 'francaise'
    df.loc[df[f] == "paris", f] = 'francaise'
    df.loc[df[f] == "marseille", f] = 'francaise'
    df.loc[df[f] == "cavaillon", f] = 'francaise'
    df.loc[df[f] == "francaise2008", f] = 'francaise'
    df.loc[df[f] == "francaienne", f] = 'francaise'
    df.loc[df[f] == "frabcaise", f] = 'francaise'
    df.loc[df[f] == "morocaine", f] = 'marocaine'
    df.loc[df[f] == "autriche", f] = 'autrichienne'
    df.loc[df[f] == "bosnienne", f] = 'bosniaque'
    df.loc[df[f] == "gabon", f] = 'gabonaise'
    df.loc[df[f] == "congo", f] = 'congolaise'
    df.loc[df[f] == "la republique democratique du congo", f] = 'congolaise'
    df.loc[df[f] == "tanger", f] = 'marocaine'
    df.loc[df[f] == "maorcaine", f] = 'marocaine'
    df.loc[df[f] == "benin", f] = 'beninoise'
    df.loc[df[f] == "belgique", f] = 'belge'
    df.loc[df[f] == "chine", f] = 'chinoise'
    df.loc[df[f] == "china", f] = 'chinoise'
    df.loc[df[f] == "america", f] = 'americaine'
    df.loc[df[f] == "american", f] = 'americaine'
    df.loc[df[f] == "peruviens", f] = 'peruvienne'
    df.loc[df[f] == "andorran", f] = 'andorraise'
    df.loc[df[f] == "turc", f] = 'turque'
    df.loc[df[f] == "turquie", f] = 'turque'
    df.loc[df[f] == "venezuela", f] = 'venezuelienne'
    df.loc[df[f] == "spanish", f] = 'espagnole'
    df.loc[df[f] == "philippines", f] = 'philippine'
    df.loc[df[f] == "sri lankaise", f] = 'srilankaise'
    df.loc[df[f] == "dominiquaise", f] = 'dominicaine'
    df.loc[df[f] == "guinee conakry", f] = 'guineenne'
    df.loc[df[f] == "algerienne ( en cours de naturalisation)", f] = 'algerienne'
    df.loc[df[f] == "americaine (etatsnotemptyunis)", f] = 'americaine'
    df.loc[df[f] == "etatsnotemptyunis", f] = 'americaine'
    df.loc[df[f] == "francaise et usa", f] = 'franco-americaine'
    df.loc[df[f] == "franconotemptyamericaine", f] = 'franco-americaine'
    df.loc[df[f] == "gabonaisear", f] = 'gabonaise'
    df.loc[df[f] == "indiennenne", f] = 'indienne'
    df.loc[df[f] == "refugie russe", f] = 'russe'
    df.loc[df[f] == "ttunisienne", f] = 'tunisienne'
    df.loc[df[f] == "turcnotemptyfrancaise", f] = 'franco-turque'
    df.loc[df[f] == "1987", f] = np.nan
    df.loc[df[f] == "80200", f] = np.nan
    df.loc[df[f] == "6 rue de la grange aux nonins", f] = np.nan
    print(df[f].unique())
    print("Nettoyage terminé")
    return df


def clean_section(df: pd.DataFrame) -> pd.DataFrame:
    t0 = time()
    print("Nettoyage section bac")
    df.loc[:, 'section_bac'] = np.nan
    df.loc[~df['Section - BAC'].isna(), 'Section - BAC'] = df.loc[
        ~df['Section - BAC'].isna(), 'Section - BAC'].str.lower().apply(unidecode)
    df.loc[df.section_bac.isna() & (df['Section - BAC'].str.contains('pro')), 'section_bac'] = "PRO"
    df.loc[df.section_bac.isna() & (df['Section - BAC'].str.contains('dut')), 'section_bac'] = "UNIVERSITE"
    df.loc[df['Section - BAC'].isin(
        ['s', 's.i.', 'soptionsi', 's-si (isn)', 'physique-chimie', 'mathematiques', 'physique - chimie', 'maths', 'ts',
         't s', 's,', "ssi", "s si", "s-si", "s, si", "s - si",
         "si", "s : si", 's,si', "s , si", "s(si)", "s.si", "s.si.", 's/si', 's_si', r's.i', "s.i.", 't s si', "c", 'd',
         'serie s2', 'serie d', ' s', 'tssi', 'spe maths', ' d',
         'term s', 's-si ', 's2', 'math/langues', 'si ', 'isn', 'pc', 'f', 'cientifico', 's.s.i', 'f2', 'f3',
         'ciencias', 'phisique', 'baccalaureat s', 'info-math',
         'mathematique', 'sm', 'ciencia fisica ', 'bac s', 'techniques-mathematiques', 's- si',
         'cientifico tecnologico', 'ciencias y tecnologia', 'ciencies i tecnologia',
         'bachillerato cientifico', 'ciencias ', 'baccalaureat americain - major maths & chemistry', 'math technique ',
         'math']) | df['Section - BAC'].str.contains(
        'terminale? s', regex=True) | df['Section - BAC'].str.contains('scien') | df['Section - BAC'].str.contains(
        'sientifique') | df['Section - BAC'].str.contains('svt') | (
                   df['Section - BAC'].str[:2] == 's '), 'section_bac'] = 'S'
    df.loc[df.section_bac.isna() & (
            df['Section - BAC'].isin(['es', 'ses', 'e.s', 'e.s.', 'sh', 'les', 'b', 's.e.s.', 'ss', "1es"]) | (
            df['Section - BAC'].str.contains('eco')
            & ~df['Section - BAC'].apply(
        lambda x: "ecol" in str(x))) | (df['Section - BAC'].str[:2] == 'es')), 'section_bac'] = 'ES'
    df.loc[df.section_bac.isna() & (
            df['Section - BAC'].str.contains('litt') | df['Section - BAC'].str.contains('lettre') | df[
        'Section - BAC'].str.contains('latin') |
            df['Section - BAC'].isin(
                ['l - cinema audiovisuel', 'l', 'a1', 'a2', 'section artistique'])), 'section_bac'] = 'L'
    df.loc[df.section_bac.isna() & (
            df['Section - BAC'].str.contains('sti ') | df['Section - BAC'].str.contains('get') | df[
        'Section - BAC'].str.contains('tsi2d') |
            df['Section - BAC'].str.contains('sin ') | df['Section - BAC'].str.contains('numerisation') | df[
                'Section - BAC'].str.contains('sti2') |
            df['Section - BAC'].str.contains('sti-') | df['Section - BAC'].str.contains('stid') | df[
                'Section - BAC'].str.contains('stdi') |
            df['Section - BAC'].str.contains('st2d') | df['Section - BAC'].str.contains('transition') | df[
                'Section - BAC'].str.contains('s.t.i') |
            df['Section - BAC'].str.contains(' sti') | df['Section - BAC'].isin(
        ['sti', 'sin', 'stie', 'tsti', 'stied', 'sti1d', 'ti'])), 'section_bac'] = 'STI2D'
    df.loc[df.section_bac.isna() & (
            df['Section - BAC'].str.contains('stmg') | df['Section - BAC'].str.contains('stg') | df[
        'Section - BAC'].str.contains('stt') |
            df['Section - BAC'].isin(['gsi', 'cfe', 'mercatique', 'cgrh', 'gmnf', 'smtg', 'h', 'sig',
                                      'terminale mercatique'])), 'section_bac'] = 'STMG'
    df.loc[df.section_bac.isna() & df['Section - BAC'].str.contains('stl'), 'section_bac'] = 'STL'
    df.loc[df.section_bac.isna() & (
            df['Section - BAC'].str.contains('st2s') | df['Section - BAC'].str.contains('stss') | df[
        'Section - BAC'].str.contains('sante') |
            df['Section - BAC'].str.contains('techniques sociales')), 'section_bac'] = 'ST2S'
    df.loc[df.section_bac.isna() & df['Section - BAC'].str.contains('stav'), 'section_bac'] = 'STAV'
    df.loc[df.section_bac.isna() & df['Section - BAC'].isin(['std2a']), 'section_bac'] = 'STD2A'
    df.loc[df.section_bac.isna() & df['Section - BAC'].str.contains('hote'), 'section_bac'] = 'STHR'
    df.loc[df.section_bac.isna() & df['Section - BAC'].str.contains('tmd'), 'section_bac'] = 'TMD'
    df.loc[df.section_bac.isna() & df['Section - BAC'].str.contains("techno"), 'section_bac'] = 'ST'
    df.loc[df.section_bac.isna() & (
            df['Section - BAC'].str.contains('pro') | df['Section - BAC'].str.contains('compta') | df[
        'Section - BAC'].str.contains('design') | (df[
                                                       'Section - BAC'].str.contains(
        'technicien') | df['Section - BAC'].str.contains('sen') | df['Section - BAC'].str.contains('elee?c',
                                                                                                   regex=True) | df[
                                                       'Section - BAC'].str.contains(
        'commerce') & ~df['Section - BAC'].apply(lambda x: "ecol" in str(x)))
            | df['Section - BAC'].str.contains('artisanat') | df['Section - BAC'].str.contains('geometre') |
            df['Section - BAC'].str.contains('vente') | df['Section - BAC'].str.contains('assistant') | df[
                'Section - BAC'].str.contains('transport') | df[
                'Section - BAC'].str.contains('boulangerie') | df['Section - BAC'].str.contains('amenagement') | df[
                'Section - BAC'].str.contains(
        'reparation') | df['Section - BAC'].str.contains('archi') | df['Section - BAC'].str.contains('micro') | df[
                'Section - BAC'].str.contains('mecanique') | df[
                'Section - BAC'].str.contains('metier') | df['Section - BAC'].str.contains('foret') | df[
                'Section - BAC'].str.contains('cgea') | df[
                'Section - BAC'].str.contains('communication') | df['Section - BAC'].str.contains('secretariat') |
            df['Section - BAC'].str.contains(
                'restauration') | df['Section - BAC'].str.contains('maintenance') | df[
                'Section - BAC'].str.contains('cuisine') | df['Section - BAC'].str.contains('energie') | df[
                'Section - BAC'].str.contains('industrie') | df['Section - BAC'].isin(
        ['securite prevention', 'accompagnement soins et service a la personne', 'cgea vigne et vins', 'tbee',
         'tebee', 'teb ee', 'tbaa', 'ebenisterie', 'carrosserie',
         'infographie', 'mrim', 'couverture', 'edpi', 'sen', 's.e.n', 's.e.n.', 's.e.n t.r', 'csem', 'tp', 'tisec',
         'snb', 'audiovisuel', 'informatique', 'lcq', 'mva',
         'm.a.e.m.c', 'hr', 'saac', 'bp com', 'pns', 'iee', 'ie e', 'tbpre', 'microtechniques', 'tma', 'm.e.i',
         'tmsec', 'tfca', 'tm', 'assp', 'bi', 'bet', 'bac teb',
         'fpir', 'm.s.m.a', 'bma ebeniste', 'bma ebenisterie', 'cgem', 'ga', 'cvp', 's/n', 'logisitique', 'eeec',
         'ga', 'ga ', 'sthr', 'sn', 'sn risc', 'mt', 'tu',
         'chaudronerie', 'ssiht', 'tr', 't.a.i', 'bgb', 'logistique', 'gestion administration', 'gestion',
         'systeme numerique', 'systeme numerique ', 'systemes numeriques',
         'gestion administraive', 'gestion administation', "gestion des systemes d'informations",
         'gestion administration '])), 'section_bac'] = 'PRO'
    df.loc[df.section_bac.isna() & (
            df['Section - BAC'].str.contains('eur') | df['Section - BAC'].str.contains('EUR') | df[
        'Section - BAC'].str.contains('abi')), 'section_bac'] = 'EURO'
    df.loc[df.section_bac.isna() & (
            df['Section - BAC'].str.contains('inter') | df['Section - BAC'].str.contains('niveau iv') | df[
        'Section - BAC'].str.contains('INTER') |
            df['Section - BAC'].str.contains('ib')), 'section_bac'] = 'INTERNATIONAL'
    df.loc[df.section_bac.isna() & (
            df['Section - BAC'].str.contains('daeu') | df['Section - BAC'].str.contains('d.a.e.u') | df[
        'Section - BAC'].str.contains('csr eu') | df[
                'Section - BAC'].str.contains('deau') | df['Section - BAC'].str.contains('univ') | df[
                'Section - BAC'].str.contains('bts') | df['Section - BAC'].str.contains('cess')
            | df['Section - BAC'].str.contains('master') | df['Section - BAC'].str.contains('dut') | df[
                'Section - BAC'].str.contains('preparatoire')
            | df['Section - BAC'].str.contains('ecole de') | df['Section - BAC'].str.contains('medecine')
            | df['Section - BAC'].str.contains('licence') | df['Section - BAC'].str.contains(
        'ecole')), 'section_bac'] = 'UNIVERSITE'
    df.loc[df.section_bac.isna() & (df['Section - BAC'].isin(
        ['generale', 'general', 'terminal', 'juin 2009', "6 juin", "2012/2013", 'bordeaux', 2010, '2012', '2013',
         '2014', '2011/2012', '2019', '/', 'autre', 'e', 'technique', 2009,
         2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2006, 2008, 'juillet', 'section_bac', 'francophone',
         'terminal', 'tdeggen', 'tborgo', 'mei', 'msc', 'tedpi',
         'daeu a', 'terminale', 'ttci', 'a4', 'tge', 'scaa', 'aucune', 'ser3', 'grh', 'bb', ' juin 2003', 'level a',
         'terminal ', 'a', 'normale', '1', 'tge', 'dnl', 'angais', 'se',
         'allemand', 'a-levels', 'vietnamien', 'pedagogique', 'nglai', 'm', 's, l.', 'formation', 'general ', '2', '-',
         'it', 'high school', 'no aplica', 'trtr', 'mpc', 'ex',
         'loem', '6 a levels =  math + math avancees + physique + chimie + biologie + francais', 'tecnologico', 'mai',
         'equivalent : niveau 4', 'technique ',
         'terminale (bac)', 'salarie',
         'grado superior desarollo multiplataforma']) | df['Section - BAC'].str.contains('0') | df[
                                        'Section - BAC'].str.contains('autre bac')), 'section_bac'] = 'UNKNOWN'
    print(df.loc[df.section_bac == "UNKNOWN", "Section - BAC"].unique())
    print("Nettoyage effectué en {0}s.".format(time() - t0))
    return df


def clean_lv(df: pd.DataFrame, field: str) -> pd.DataFrame:
    t0 = time()
    print("Nettoyage langue vivante")
    df.loc[:, field] = df.loc[:, field].str.lower()
    df.loc[~df[field].isna(), field] = df.loc[~df[field].isna(), field].apply(unidecode).str.strip(' ')
    df.loc[~df[field].isna() & (
            df[field].str.contains('ang') | df[field].str.contains('anf') | df[field].str.contains('eng') | df[
        field].str.contains('anl') | df[field].str.contains('glais')
            | df[field].str.contains('analg') | df[field].str.contains('ingl') | df[field].str.contains('agnl')),
           field] = 'anglais'
    df.loc[~df[field].isna() & (
            df[field].str.contains('esp') | df[field].isin(
        ['epagnol', 'expagnole', 'esoagnol', 'expagnol', 'esagnol', 'spanish', 'esapgnol', 'spagnole', 'eapagnol',
         'epsagnol',
         'spagnol', 'expognol', 'eqpagnol'])), field] = 'espagnol'
    df.loc[~df[field].isna() & (df[field].str.contains('all') | df[field].str.contains('ale') | df[field].str.contains(
        'germ')), field] = 'allemand'
    df.loc[~df[field].isna() & df[field].str.contains('it'), field] = 'italien'
    df.loc[~df[field].isna() & df[field].str.contains('portu'), field] = 'portugais'
    df.loc[~df[field].isna() & (df[field].str.contains('fra') | df[field].str.contains('fren') | df[field].isin(
        ["fr", "fancais", 'frnacais'])), field] = 'francais'
    df.loc[~df[field].isna() & df[field].str.contains('hol'), field] = 'neerlandais'
    df.loc[~df[field].isna() & df[field].str.contains('arab'), field] = 'arabe'
    df.loc[~df[field].isna() & df[field].str.contains('vinet'), field] = 'vietnamien'
    df.loc[~df[field].isna() & df[field].str.contains('hebr'), field] = 'hebreu'
    df.loc[~df[field].isna() & df[field].str.contains('ruso'), field] = 'russe'
    df.loc[~df[field].isna() & df[field].str.contains('jap'), field] = 'japonais'
    df.loc[~df[field].isna() & df[field].str.contains('grec'), field] = 'grec'
    df.loc[~df[field].isna() & df[field].str.contains('creol'), field] = 'creole'
    df.loc[~df[field].isna() & (df[field].str.contains('manda') | df[field].str.contains('chn')), field] = 'chinois'
    df.loc[df[field].isin(
        ['/', 'aucun', 'neant', 'n/a', 'none', '0', '-', '', 'e', 'ns', 'rien', 'sans objet', 'dispense',
         'non pratique', 'pas de lv2', 'sans', 'dispencer', 'aucune', 'aucunne',
         '', 'bb', 'ipiab', 'baccalaureat technologique - serie e -', 'sciences physiques',
         'casablanca nouaceur , maroc', 'baccalaureat mathematiques', 's.o', 'science '
                                                                             'experimental,',
         'science de la nature et de la vie', 'la bergerie', 'electronique', "technicien d'assistance informatique",
         'l1, informatique de gestion',
         'hamdani  said tizi ozou algerie', 'biologie chimie', 'sciences humaines', 'egico',
         'option - genie industriel', 'subias', 'software', 'bepc',
         'option science physique metion(tres bien', 'l2, informatique de gestion', 'bejaia', '2016', 'mathematique',
         'lycee yazouren said azazega', 'physique chimie', 'polaco',
         'cpge mp spe', '11,65', 'physik', '3333', '4ans', 'adr', 'science physique', '2',
         'sciences de la vie et management', 'bachelor', 'x', 'geographie', 'en cour', 'ar',
         'descartes u.f.r.', 'science economique', '2eme annee  bac sciences physiques', 'jff', '16,5',
         'option: science mathematique, b', '18.5', 'systeme et reseaux',
         'porte de restauration', 'bac+3 in computer science & finance', 'bbb', 'reseau', '14,43', 'dd',
         'chimie industrielle', 'xxxxxxxxxxxxx', 'electrotechnique', 'non',
         'informatique industriel', 'classe prepa 3il 2', '12.10', 'zarma', 'neans', '19', '13', '12.12',
         'computer application', 'section europeenne',
         'tazmalt', 'fort', "sciences de l'informatique", 'bien', 'informatique', 'ninguno', '2015', 'abeche', 'bac',
         'brevet de technicien en informatique', 'excellent',
         'biologie', '10.73', 'lycee hamadi mohamed aghribs', 'maintenance industriel bts',
         'sciences de la vie et de la terre', 'sciences experimentales', 'prepa a la upec',
         'cpge mp sup', 'lycee jemmel', '14,12', '11', '63/100', 'technik', 'niveau bac', '1 - age', '2 - no news',
         'rethorique', 'fdsfds', '770', 'sciences de la vie',
         'option internationale americaine', 'mathematik', '12', 'renate caswara', '11,930', '1111', '12ans', 'svt',
         '1', "lycee d'oyack douala-bassa", 'd c i',
         'mathematiques & sciences physiques', 'science mathematique serie a', 'echec', 'oui', 'sciences mathematiques',
         'bac scientifique', 'mathematiques &sciences physiques',
         '8', 'economie', '1ere annee  bac. science maths', 'lycee gimnazija bezigrad ib program', 'rh', '15',
         'lycee er-razi, settat, maroc', 'l1', 'lycee tizi-ouzou algerie',
         '10.74', '17.5', 'non concernee', 'math', '11,5', '13.90', '14,37 de moyenne', 'lycee college saint etienne',
         '..', 'developpement', 'victor hugo', 'genie electrique',
         'crefi', 'licence 1 en informatique', 'high school diploma', 'aaa', 'maintenance', '14,53', 'cccdd', 'arpajon',
         'tres bien', 'madagascar', 'lyhana', '17,16',
         'marrakech', 'xxxxxxxxxx', 'f3', 'bounoura', 'UNKNOWN', 'ingenieur industriel en informatique', 'moyen',
         'automatisme', 'pc', 'paris', 'cepd', 'france',
         'classe prepa 3il 1', 'master', '10', 'mouloud mammeri', 'lycee ley-wendou', 'mathemathiques',
         'dessin assiste par ordinateur', "n'djamena (tchad)", 'deug', '18',
         '11.46', 'commerce']), field] = np.nan
    print("Nettoyage effectué en {0}s.".format(time() - t0))
    return df


def clean_annee(df: pd.DataFrame, field: str) -> pd.DataFrame:
    df.loc[~df[field].isna(), field] = df.loc[~df[field].isna(), field].apply(lambda x: x.strip())
    for y in range(2000, 2020):
        df.loc[df[field].isin(
            ['-'.join([str(y), str(y + 1)]), '/'.join([str(y), str(y + 1)]), ' / '.join([str(y), str(y + 1)]),
             ' - '.join([str(y), str(y + 1)]), ' '.join([str(y), str(y + 1)]),
             '_'.join([str(y), str(y + 1)])]), field] = y
    df.loc[df[field].isin(
        ['nan', 'S', 'STG', 'Terminale', 'terminale', 'STI', 'Oui', 'terminal', 'T', 'S', 'SI', '3', 'STL', 'i', '1',
         'Tle', 'Term', 'SEN', '2', '0000', 'Terminal',
         'ES']), field] = np.nan
    df.loc[df[field].isin(['? 2012', '20012', '2102', '2012s']), field] = 2012
    df.loc[df[field].isin(['2009-2011', '211']), field] = 2011
    df.loc[df[field].isin(['2012-13', '2012/13', '2013S']), field] = 2013
    df.loc[df[field].isin(['2012-2014', '2013/14']), field] = 2014
    df.loc[df[field].isin(['2012-2015', '1015']), field] = 2015
    df.loc[df[field].isin(['2014/2017', 'en cours 2017']), field] = 2017
    df.loc[df[field].isin(['2017/18', '1018']), field] = 2018
    df.loc[~df[field].isna(), field] = df.loc[~df[field].isna(), field].apply(
        lambda x: int(x) if str(x).isnumeric() and float(str(x)) > 1940 else "")
    return df


def clean_filiere(df: pd.DataFrame, field: str) -> pd.DataFrame:
    df.loc[df[field] == 'Epitech 1 PGE', field] = "1-PGE"
    df.loc[df[field] == 'PSO Section 1 PGE', field] = "1-PSO"
    df.loc[df[field] == 'PSO Section 2 PGE', field] = "2-PSO"
    df.loc[df[field] == 'Epitech 2 PGE', field] = "2-PGE"
    df.loc[df[field] == 'Epitech 2 PGT', field] = "2-PGT"
    df.loc[df[field] == 'Pré-MSc Pro-ed', field] = "3-MSC0ed"
    df.loc[df[field] == 'Pré-MSc Pro', field] = "3-MSC0"
    df.loc[df[field] == 'Epitech 3 PGE', field] = "3-PGE"
    df.loc[df[field] == 'Epitech 3s PGE', field] = "3-PGEs"
    df.loc[df[field] == 'Tech3si PGE', field] = "3-PGEsi"
    df.loc[df[field] == 'Epitech 3-ed PGT', field] = "3-PGTed"
    df.loc[df[field] == 'Epitech 3 PGT', field] = "3-PGT"
    df.loc[df[field] == 'Epitech 4 PGE', field] = "4-PGE"
    df.loc[df[field] == 'Epitech 4-ed PGT', field] = "4-PGTed"
    df.loc[df[field] == 'Epitech 4 PGT', field] = "4-PGT"
    df.loc[df[field] == 'MSc Pro 1-ed', field] = "4-MSC1ed"
    df.loc[df[field] == 'MSc Pro 1', field] = "4-MSC1"
    df.loc[df[field] == 'Epitech 5 PGT', field] = "5-PGT"
    df.loc[df[field] == 'Epitech 5 PGE', field] = "5-PGE"
    df.loc[df[field] == 'MSc Pro 2', field] = "5-MSC2"
    df.loc[:, 'numero_annee_cursus'] = np.nan
    df.loc[~df[field].isna(), 'numero_annee_cursus'] = df.loc[~df[field].isna(), field].apply(lambda x: x.split("-")[0])
    return df


def clean_gender(df: pd.DataFrame, field: str) -> pd.DataFrame:
    df.loc[df[field] == 'Monsieur', field] = "M"
    df.loc[df[field] == 'Madame', field] = "F"
    df.loc[df[field] == 'Mademoiselle', field] = "F"
    df.loc[~df[field].isin(["M", "F"]), field] = np.nan
    return df


def clean_pays(df: pd.DataFrame, field: str) -> pd.DataFrame:
    df.loc[df[field] == "USA", field] = "Etats Unis"
    df.loc[df[field] == "Etats-Unis", field] = "Etats Unis"
    df.loc[df[field] == "Angleterre", field] = "Royaume Uni"
    df.loc[df[field] == "Royaume-Uni", field] = "Royame Uni"
    df.loc[df[field] == "BELGIQUE", field] = "Belgique"
    df.loc[df[field] == "Bangladesh", field] = "Bengladesh"
    df.loc[df[field] == "Corée Du Sud", field] = "Coree"
    df.loc[df[field] == "Corée du sud", field] = "Coree"
    df.loc[df[field] == "Pays-Bas", field] = "Pays Bas"
    df.loc[df[field] == "Pays-bas", field] = "Pays Bas"
    df.loc[df[field] == "suisse", field] = "Suisse"
    df.loc[df[field] == "LUXEMBOURG", field] = "Luxembourg"
    return df


def clean_fonction(df: pd.DataFrame, field: str) -> pd.DataFrame:
    df.loc[df[field].isin(['Développeur Back']), field] = "Développeur / Ingénieur logiciel"
    df.loc[df[field].isin(['Développeur logiciel']), field] = "Développeur / Ingénieur logiciel"
    df.loc[df[field].isin(['CPO']), field] = 'Chef de projet / MOA / MOE'
    df.loc[df[field].isin(['Tech lead']), field] = 'Chef de projet / MOA / MOE'
    df.loc[df[field].isin(['Développeur Mobile']), field] = "Développeur / Ingénieur logiciel"
    df.loc[df[field].isin(['Dev Ops']), field] = "Développeur / Ingénieur logiciel"
    df.loc[df[field].isin(['Développeur full stack']), field] = "Développeur / Ingénieur logiciel"
    df.loc[df[field].isin(['Developer Realite virtuelle']), field] = "Développeur / Ingénieur logiciel"
    df.loc[df[field].isin(['Ingénieur de développement']), field] = "Développeur / Ingénieur logiciel"
    df.loc[df[field].isin(['Développeur Front']), field] = "Développeur / Ingénieur logiciel"
    df.loc[df[field].isin(['Business Analyst / Project Manager']), field] = 'Chef de projet / MOA / MOE'
    df.loc[df[field].isin(['Ingénieur Cybersécurité']), field] = "Développeur / Ingénieur logiciel"
    df.loc[df[field].isin(['Lead développeur']), field] = "Développeur / Ingénieur logiciel"
    df.loc[df[field].isin(['Gameplay Programmer ']), field] = "Développeur / Ingénieur logiciel"
    df.loc[df[field].isin(['Product Designer for Apple Maps']), field] = 'Chef de projet / MOA / MOE'
    df.loc[df[field].isin(['Site Reliability Engineer']), field] = 'Chef de projet / MOA / MOE'
    df.loc[df[field].isin(['Développeur Jeux Vidéo']), field] = "Développeur / Ingénieur logiciel"
    df.loc[df[field].isin(['Développeur Concepteur Logiciel']), field] = "Développeur / Ingénieur logiciel"
    df.loc[df[field].isin(['Programmeur Gameplay']), field] = "Développeur / Ingénieur logiciel"
    df.loc[df[field].isin(['Ingénieur logiciel']), field] = "Développeur / Ingénieur logiciel"
    df.loc[df[field].isin(['Développeur Réalité Virtuelle']), field] = "Développeur / Ingénieur logiciel"
    df.loc[df[field].isin(['Ingénieur Cyber-Sécurité']), field] = "Développeur / Ingénieur logiciel"
    df.loc[df[field].isin(['Auditeur IT']), field] = "Expert / Consultant technique"
    df.loc[df[field].isin(['Développeur / Informaticien ']), field] = "Développeur / Ingénieur logiciel"
    df.loc[df[field].isin(['Product Owner']), field] = 'Chef de projet / MOA / MOE'
    df.loc[df[field].isin(['Product Buider']), field] = 'Chef de projet / MOA / MOE'
    df.loc[df[field].isin(['Ingénieur jeu vidéo']), field] = "Développeur / Ingénieur logiciel"
    df.loc[df[field].isin(['Développeur IOT']), field] = "Développeur / Ingénieur logiciel"
    df.loc[df[field].isin(['Développeur Cloud']), field] = "Développeur / Ingénieur logiciel"
    df.loc[df[field].isin(['Développeur C++']), field] = "Développeur / Ingénieur logiciel"
    df.loc[df[field].isin(['Expert technique']), field] = "Expert / Consultant technique"
    df.loc[df[field].isin(['Software Engineering Leader']), field] = "Développeur / Ingénieur logiciel"
    df.loc[df[field].isin(['Développeur']), field] = "Développeur / Ingénieur logiciel"
    df.loc[df[field].isin(['Développeur & Chef de projet technique']), field] = "Développeur / Ingénieur logiciel"
    df.loc[df[field].isin(['Lead Développeur Full Stack']), field] = "Développeur / Ingénieur logiciel"
    df.loc[df[field].isin(['Data Engineer']), field] = 'Ingénieur Data / Data scientist'
    df.loc[df[field].isin(['Ingénieur cybersecurité']), field] = 'Ingénieur Data / Data scientist'
    df.loc[df[field].isin(['Président']), field] = 'CEO / Directeur exécutif'
    df.loc[df[field].isin(['Développeur embarque']), field] = "Développeur / Ingénieur logiciel"
    df.loc[df[field].isin(['Ingénieur machine learning']), field] = 'Ingénieur Data / Data scientist'
    df.loc[df[field].isin(['Head of Security']), field] = 'analyste sécurité'
    df.loc[df[field].isin(['Ingénieur Logiciel']), field] = "Développeur / Ingénieur logiciel"
    df.loc[df[field].isin(['Développeur Logiciel']), field] = "Développeur / Ingénieur logiciel"
    df.loc[df[field].isin(['Ingenieur video / optimisation']), field] = "Développeur / Ingénieur logiciel"
    df.loc[df[field].isin(['Ingénieur systèmes de production']), field] = "Développeur / Ingénieur logiciel"
    df.loc[df[field].isin(['DEVELOPEUR C']), field] = "Développeur / Ingénieur logiciel"
    df.loc[df[field].isin(['Quantitative Développeur ']), field] = "Développeur / Ingénieur logiciel"
    df.loc[df[field].isin(['Développeur Gameplay']), field] = "Développeur / Ingénieur logiciel"
    df.loc[df[field].isin(['Dev Software']), field] = "Développeur / Ingénieur logiciel"
    df.loc[df[field].isin(['Product Designer']), field] = 'Chef de projet / MOA / MOE'
    df.loc[df[field].isin(['Head of QA']), field] = 'QA'
    df.loc[df[field].isin(['Lead Dev']), field] = "Développeur / Ingénieur logiciel"
    df.loc[df[field].isin(['Développeur jeux vidéo']), field] = "Développeur / Ingénieur logiciel"
    df.loc[df[field].isin(['Ingénieur systèmes embarqués']), field] = "Développeur / Ingénieur logiciel"
    df.loc[df[field].isin(['Ingénieure en développement logiciels']), field] = "Développeur / Ingénieur logiciel"
    df.loc[df[field].isin(['Développeur/Concepteur ']), field] = "Développeur / Ingénieur logiciel"
    df.loc[df[field].isin(['Ingénieur QA']), field] = 'QA'
    df.loc[df[field].isin(['Développeur réseau']), field] = "Développeur / Ingénieur logiciel"
    df.loc[df[field].isin(['Lead Tech']), field] = 'Chef de projet / MOA / MOE'
    df.loc[df[field].isin(['Tech Lead']), field] = 'Chef de projet / MOA / MOE'
    df.loc[df[field].isin(['Développeur Python']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Developpeur Full stack']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Software Engineer']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(["Ingénieur d'Applications"]), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Consultant']), field] = 'Expert / Consultant technique'
    df.loc[df[field].isin(['Ingénieur Développement Logiciel']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Lead developper Front-end']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Développeur JAVA']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Chef de projet développement']), field] = 'Chef de projet / MOA / MOE'
    df.loc[df[field].isin(['Lead developer ']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['developpeur informatique']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Développeur Web']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(["Ingénieur d'études"]), field] = 'Ingénieur études et développement'
    df.loc[df[field].isin(['software engineer']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Développeur informatique']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Ingenieur datacenter']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Ingénieur études et développement 1']), field] = 'Ingénieur études et développement'
    df.loc[df[field].isin(["Développeur en charge de l'expoitation"]), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Consultant ']), field] = 'Expert / Consultant technique'
    df.loc[df[field].isin(['Développer']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Ingénieur technique ']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Administrateur systèmes et réseaux']), field] = 'Administrateur Système'
    df.loc[df[field].isin(['Ingénieur Consultant']), field] = 'Expert / Consultant technique'
    df.loc[df[field].isin(['Entrepreneur']), field] = 'CEO / Directeur exécutif'
    df.loc[df[field].isin(['Programmeur Moteur']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Analyste consultant ']), field] = 'Expert / Consultant technique'
    df.loc[df[field].isin(['Ingénieure Systèmes']), field] = 'Administrateur Système'
    df.loc[df[field].isin(['Développeur fullstack']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Consultant .Net']), field] = 'Expert / Consultant technique'
    df.loc[df[field].isin(['dev iOS']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Fullstack Developper']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Ingénieur développeur']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(["Ingénieur d'étude"]), field] = "Ingénieur de recherche ou d'études"
    df.loc[df[field].isin(['Software engineer, web & mobile solutions']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['ios developer']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Ingénieur développement logiciel']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Cto']), field] = 'CTO / Directeur technique'
    df.loc[df[field].isin(['Software Engineer and Research assistant']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['BackEnd developpeur']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Développeur iOS']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Mobile Developer']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['développeur']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[
        df[field].isin(['Ingénieur Support technique vidéosurveillance']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Ingénieur Informatique']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Développeur Unity3D / Chef de projet JR']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Developper Full Stack']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Product Manager']), field] = 'Chef de projet / MOA / MOE'
    df.loc[df[field].isin(['Développeur web']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Ingénieur export']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Lead Développeur Back-end']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Ingénieur étude et développement ']), field] = "Ingénieur de recherche ou d'études"
    df.loc[df[field].isin(['Lead iOS developer']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Développeur backend']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Ingénieur développeur logiciel']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Développeur Android']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Développeur fullstack JS']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Développeur ']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Cybersecurity consultant']), field] = 'Expert / Consultant technique'
    df.loc[df[field].isin(['Ingénieur']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Software engineer']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Developpeur.euse generaliste console']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Ingénieur en développement logiciel']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Développeur / Ingénieur logiciel']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Developpeur']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Lead developer mobile']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Ingénieur Recherche & Développement']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Développeur Full-Stack']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Data Scientist']), field] = 'Ingénieur Data / Data scientist'
    df.loc[df[field].isin(['Research Engineer Computer Vision']), field] = "Ingénieur de recherche ou d'études"
    df.loc[df[field].isin(['Développeur Front-End']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Ingénieur data']), field] = 'Ingénieur Data / Data scientist'
    df.loc[df[field].isin(['Développeur backend php']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Devops']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['développer full-stack php']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Online Developer']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Ingenieur informatique junior']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Consultant Technique']), field] = 'Expert / Consultant technique'
    df.loc[df[field].isin(['Consultant iOS']), field] = 'Expert / Consultant technique'
    df.loc[df[field].isin(['Ingénieur débutant']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Analyste développeur']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Security Researcher']), field] = "Ingénieur de recherche ou d'études"
    df.loc[df[field].isin(['Ingénieur informatique']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Développeur Mobile iOS']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['junior iOS Developer']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Ingénieur en développement']), field] = "Ingénieur de recherche ou d'études"
    df.loc[df[field].isin(['Software Engineering']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Fullstack Rails developer']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['software engineer full stack']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['CEO']), field] = 'CEO / Directeur exécutif'
    df.loc[df[field].isin(["Consultant en technologies de l'information"]), field] = "Consultant / Expert technique"
    df.loc[df[field].isin(['Accompagnateur Pédagogique Epitech']), field] = 'Enseignant / Accompagnateur pédagogique'
    df.loc[df[field].isin(['Développeur R&D']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Analyste developper']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Fullstack developer']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['FullStack Dev']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Développeur Full Stack']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Data Processing Developper']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Ingénieur développeur C++']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['SOFTWARE ENGINEER']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['software ingé']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Consultant Junior en technologies Microsoft']), field] = 'Ingénieur Data / Data scientist'
    df.loc[df[field].isin(['Associate Software Engineer']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Consultant développement web']), field] = 'Expert / Consultant technique'
    df.loc[df[field].isin(['Développeur web et mobile']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Développeur FULL STACK']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Consultant & Formateur']), field] = 'Expert / Consultant technique'
    df.loc[df[field].isin(['developeur mobile']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Développeur Unity 3D']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Ingénieur étude et développement']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Software Developer Analyst']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Lead Web Developper']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Gameplay Programmer']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Lead Developpeur Web ']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Data scientist']), field] = 'Ingénieur Data / Data scientist'
    df.loc[df[field].isin(['Developpeur full stack']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Développeur C#']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Développeur web Javascript']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Developpeur back-end']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Directeur Général']), field] = 'CEO / Directeur exécutif'
    df.loc[df[field].isin(['Développeur embarqué']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Consultant OpenText']), field] = 'Expert / Consultant technique'
    df.loc[df[field].isin(['BI Consultant']), field] = 'Expert / Consultant technique'
    df.loc[df[field].isin(['Reférent technique ']), field] = 'Expert / Consultant technique'
    df.loc[df[field].isin(['Développeur FullStack']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Accompagnateur Pédagogique EPITECH']), field] = 'Enseignant / Accompagnateur pédagogique'
    df.loc[df[field].isin(['Consultant Android']), field] = 'Expert / Consultant technique'
    df.loc[df[field].isin(['Consultant Solutions']), field] = 'Expert / Consultant technique'
    df.loc[df[field].isin(['Ingénieur en informatique']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Ingenieur front-end']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Développeur back-end']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Analyste programmeur']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Directrice générale']), field] = 'CEO / Directeur exécutif'
    df.loc[df[field].isin(['Directeur général']), field] = 'CEO / Directeur exécutif'
    df.loc[df[field].isin(['Unity Developer']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Consultant IT']), field] = 'Expert / Consultant technique'
    df.loc[df[field].isin(['Ingénieur développement']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Ingénieur Data / Data scientist']), field] = 'Ingénieur Data / Data scientist'
    df.loc[df[field].isin(['Devops Engineer ']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Front-End technical leader']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Software developer']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Developer Full-stack']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['developpeur unity c#']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Ingénieur développeur junior']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Développeur Web Full Stack']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Architecte logiciel']), field] = 'Architecte'
    df.loc[df[field].isin(['Back-end developer']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Ingénieur logiciel/web']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(
        ['Ingénieur de production des systèmes d’information']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Développeur Logiciel/Web']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Consultant test']), field] = "QA"
    df.loc[df[field].isin(['Directeur pédagogique adjoint']), field] = 'Enseignant / Accompagnateur pédagogique'
    df.loc[df[field].isin(['Consultant - Code management & Quality']), field] = "QA"
    df.loc[df[field].isin(['Accompagnateur pedagogique']), field] = 'Enseignant / Accompagnateur pédagogique'
    df.loc[df[field].isin(['Développeur fullstack javascript']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Consultant JavaScript']), field] = 'Expert / Consultant technique'
    df.loc[df[field].isin(['Lead iOS']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Accompagnateur Pédagogique']), field] = 'Enseignant / Accompagnateur pédagogique'
    df.loc[df[field].isin(['Ingénieur Etude et Développement']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Dev']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(["Responsables des systèmes d'informations"]), field] = "Administrateur Système"
    df.loc[df[field].isin(['Développeur en intelligence artificielle']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Lead Android Developer']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Développeur backend ']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['consultant BI']), field] = 'Expert / Consultant technique'
    df.loc[df[field].isin(['Lead Developpeur']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Ingénieur Etude et Developpement']), field] = "Ingénieur de recherche ou d'études"
    df.loc[df[field].isin(['Solution developer']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Ingénieur Developpeur']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Ingénieur Développement .NET']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['iOS Developer']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Chef de projet']), field] = 'Chef de projet / MOA / MOE'
    df.loc[df[field].isin(['Developeur Golang et C++']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Dévelopeur android']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Consultant BI']), field] = 'Expert / Consultant technique'
    df.loc[df[field].isin(['Ingénieur qualité']), field] = "QA"
    df.loc[df[field].isin(['Développeur multimédia (.NET)']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['consultant']), field] = 'Expert / Consultant technique'
    df.loc[df[field].isin(['Responsable Développeur']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Software Developer']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Développeur Backend Juinior']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['CTO']), field] = 'CTO / Directeur technique'
    df.loc[df[field].isin(['Ingenieur System DevOps']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Ingénieur Junior']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Lead developer']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Développeur mobile freelance']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Delta1 Software Developer']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Computer Vision Engineer']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Développeur outils']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Consultant NTIC']), field] = 'Expert / Consultant technique'
    df.loc[df[field].isin(['Dévelopeur']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Développeur Fullstack']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(["Ingénieur sécurité informatique / tests d'intrusion"]), field] = 'analyste sécurité'
    df.loc[df[field].isin(['Lead dev IOS']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Lead Developer']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Full Stack Developper']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Consultant SAP']), field] = 'Expert / Consultant technique'
    df.loc[df[field].isin(['Lead Developer ']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Ingénieur étude et developpement']), field] = "Ingénieur de recherche ou d'études"
    df.loc[df[field].isin(['Annalyste développeur']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Web developper']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Analyste Programmeur']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Developpeur ']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Xamarin Consultant']), field] = 'Expert / Consultant technique'
    df.loc[df[field].isin(['Ingénieur Junior Etudes et Développement']), field] = "Ingénieur de recherche ou d'études"
    df.loc[df[field].isin(['Ingenieur Etude Et Developpement']), field] = "Ingénieur de recherche ou d'études"
    df.loc[df[field].isin(['DevOps']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Developpeur consultant ']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Consultant Sécurité']), field] = 'Expert / Consultant technique'
    df.loc[df[field].isin(['Devops engineer']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Développeur Commando']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Dev Logiciel Embarqué']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(["Ingénieur d'étude et développement"]), field] = "Ingénieur de recherche ou d'études"
    df.loc[df[field].isin(["Ingenieur d'etude"]), field] = "Ingénieur de recherche ou d'études"
    df.loc[df[field].isin(['Ingénieur de développement logiciel']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Ingénieur Solutions']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Administrateur Sytème et réseau']), field] = 'Administrateur Système'
    df.loc[df[field].isin(['Full Stack Developer']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Dév iOS']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['ingénieur logiciel']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Ingénieur backend ']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['développeur full stack']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Software Engineer .Net']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Développeur Full Stack Web']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Ingenieur']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Android developer']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Développeur android']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Developpeur informatique']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Consultant nouvelles technologies']), field] = 'Expert / Consultant technique'
    df.loc[df[field].isin(['Consultant informatique']), field] = 'Expert / Consultant technique'
    df.loc[df[field].isin(['Ingénieur Étude et Développement']), field] = "Ingénieur de recherche ou d'études"
    df.loc[df[field].isin(['Ingénieur réseaux']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Directrice de projet CRM  (MOE/MOA)']), field] = 'Chef de projet / MOA / MOE'
    df.loc[df[field].isin(['Directeur technique']), field] = 'CTO / Directeur technique'
    df.loc[df[field].isin(['Ingénieur Concepteur Développeur ']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Responsable Pédagogique']), field] = 'Enseignant / Accompagnateur pédagogique'
    df.loc[df[field].isin(['Gameplay Developer']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Développeur .NET Full Stack']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Senior Developer']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Consultant IT / Web developer']), field] = 'Expert / Consultant technique'
    df.loc[df[field].isin(['Ingénieur consultant ']), field] = 'Expert / Consultant technique'
    df.loc[df[field].isin(['Administrateur SI']), field] = 'Administrateur Système'
    df.loc[
        df[field].isin(['Développeur Application hybride (IOS - Android)']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Développeur Ruby on Rails']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Technical support analyst']), field] = 'Expert / Consultant technique'
    df.loc[df[field].isin(['Développeur logciel, web et mobile']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Ingénieur expert ']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['DevOps Engineer']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Assistant Pédagogique Epitech']), field] = 'Enseignant / Accompagnateur pédagogique'
    df.loc[df[field].isin(["Ingénieur d'affaire"]), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Business Developper']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Développeur full stack JS']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Responsable pédagogique']), field] = 'Enseignant / Accompagnateur pédagogique'
    df.loc[df[field].isin(['Gameplay/UI Programmer Junior']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Développeur Front-end']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['iOS developer']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Accompagnateur pédagogique']), field] = 'Enseignant / Accompagnateur pédagogique'
    df.loc[df[field].isin(['Developpeur Web']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Ingénieur Infrastructure']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Consultant Cloud']), field] = 'Expert / Consultant technique'
    df.loc[df[field].isin(['UI Software Developer']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Developer']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Ingénieur d’etudes et développement']), field] = "Ingénieur de recherche ou d'études"
    df.loc[df[field].isin(['Business Manager']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Ingénieur Etudes et Développements']), field] = "Ingénieur de recherche ou d'études"
    df.loc[df[field].isin(['Ingénieur Systeme et réseaux']), field] = 'Administrateur Système'
    df.loc[df[field].isin(['Developpeur fullstack']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Ingénieur junior']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Developpeur web junior']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(
        ["Développeur web au sein d'une entreprise de chatbots"]), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Ingénieur d’applIcation']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Data Analyst']), field] = 'Ingénieur Data / Data scientist'
    df.loc[df[field].isin(['Software engineer / Project manager']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Développeur full-stack']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Ingénieur Support Logiciel']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['ingenieur qa']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Ingenieur Avant Vente']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Ingénieur Etude et developement']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['solution developer']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Responsable Système Information']), field] = "Administrateur Système"
    df.loc[df[field].isin(
        ["Développeur fullstack d'application web et mobile"]), field] = 'Développeur / Ingénieur logiciel'
    df.loc[
        df[field].isin(['Entre Backend developer et solution developer']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(["ingénieur d'études développeur"]), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Development Engineer']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Analyst developer']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Chef de projet technique']), field] = 'Chef de projet / MOA / MOE'
    df.loc[df[field].isin(['Consultant développeur Javascript']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(["Ingénieur d'études Web & Développement"]), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(["Développeur d'application web"]), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Développeur nodeJS']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['NodeJS developer']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Développeur Logiciels']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Ingenieur full stack']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Développeur Outils']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Consultant MOE']), field] = 'Chef de projet / MOA / MOE'
    df.loc[df[field].isin(['IT Consultant']), field] = 'Expert / Consultant technique'
    df.loc[df[field].isin(['Junior Data Scientist Consultant']), field] = 'Expert / Consultant technique'
    df.loc[df[field].isin(['Développeur web / mobile']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['R&D Developer']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Lead software engineer']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['FrontEnd Developper']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['DevOps/Data Protection Officer']), field] = 'Ingénieur Data / Data scientist'
    df.loc[df[field].isin(["Responsable Evolution d'Architecture"]), field] = 'Architecte'
    df.loc[df[field].isin([
        "Ingénieurs et cadres d'études, recherche et développement en informatique "]), field] = "Ingénieur de recherche ou d'études"
    df.loc[df[field].isin(['Spécialiste développeur']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Ingénieur Etudes et Développement']), field] = "Ingénieur de recherche ou d'études"
    df.loc[df[field].isin(['Developeur']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['developpeur']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Consultant et développeur junior']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Software support engineer']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Développeur Web Full-Stack']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['DEVELOPPEUR FRONTEND']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Développeuse junior']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Consultant Junior en CyberSécurité']), field] = 'Expert / Consultant technique'
    df.loc[df[field].isin(['Android Software Engineer']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Fullstack Software Engineer']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Chef de projet ']), field] = 'Chef de projet / MOA / MOE'
    df.loc[df[field].isin(['Développeur mobile, Cadre consultant']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Ingénieur logiciel / Chargé de projet']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(["Consultant sécurité / Tests d'intrusion "]), field] = 'Expert / Consultant technique'
    df.loc[df[field].isin(['Ingénieur réseau et sécurité']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Développeur plugin / Scrum Master']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Entrepreneur']), field] = 'CEO / Directeur exécutif'
    df.loc[df[field].isin(['Consultant Développeur']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Devops Cloud ']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['développeur']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Senior front end developer']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Développeur fullstack JS']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Freelance developer full-stack']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['Software Developer']), field] = 'Développeur / Ingénieur logiciel'
    df.loc[df[field].isin(['ingénieur réseau / administrateur système et réseaux']), field] = 'Administrateur Système'
    return df


def clean_secteur(df: pd.DataFrame, field: str) -> pd.DataFrame:
    df.loc[~df[field].isna(), field] = \
        df.loc[~df[field].isna(), field].apply(lambda x: unidecode(str(x).lower()))
    df.loc[df[field].apply(lambda x: ";" in str(x)), field] = \
        df.loc[df[field].apply(lambda x: ";" in str(x)), field].apply(lambda x: str(x).split(";")[0])
    df.loc[df[field].apply(lambda x: "," in str(x)), field] = \
        df.loc[df[field].apply(lambda x: "," in str(x)), field].apply(lambda x: str(x).split(",")[0])
    df.loc[df[field].apply(lambda x: "/" in str(x)), field] = \
        df.loc[df[field].apply(lambda x: "/" in str(x)), field].apply(lambda x: str(x).split("/")[0])
    df[field] = df[field].apply(lambda x: str(x).strip())
    df.loc[df[field].isin(["telecommunications"]), field] = "telecom"
    df.loc[df[field].isin(["logiciel (libre", "logiciel (commercial",
                           "editeur logiciel", "logiciels",
                           "editeur logiciel"]), field] = "logiciel"
    df.loc[df[field].isin(
        ["multimedia (son", "multimedia(commercial"]), field] = "multimedia"
    df.loc[df[field].isin(["autre : edition"]), field] = "edition"
    df.loc[df[field].isin(["autre : web"]), field] = "web"
    df.loc[df[field].isin(["services bancaires", "banque", "assurance", 'finances', "finance (banque",
                           "autre : finance", "finance", "secteur bancaire",
                           'assurrance']),
           field] = "banque / assurance / finance"
    df.loc[df[field].isin(["agence de communication", "autre : agence de communication", "media",
                           'communication']), field] = "communication / publicite"
    df.loc[df[field].isin(["e-marketing ", 'targeting publicitaire', 'marketing interactif',
                           'social marketing']), field] = "marketing"
    df.loc[df[field].isin(["surveillance de l'environnement", "ecomobilite"]), field] = "environnement"
    df.loc[df[field].isin(
        ["autre : jeux videos", "jeux videos", "virtualisation", "jeu video",
         "jeux video"]), field] = "JV / Virtualisation"
    df.loc[df[field].isin(["autre : sport"]), field] = "sport"
    df.loc[df[field].isin(
        ["ingenieur r&d", 'r & d, enseignement', 'recherche & innovation', 'recherche', 'r & d']), field] = "R et D"
    df.loc[df[field].isin(["conseil", 'consultant', "conseil "]), field] = "conseil"
    df.loc[df[field].isin(['e-commerce']), field] = "commerce"
    df.loc[df[field].isin(['militaire', "industrie et defense"]), field] = "defense"
    df.loc[df[field].isin(['application mobiles', 'mobile']), field] = "mobile"
    df.loc[df[field].isin(['mode', 'cosmetiques', 'cosmetique', 'mode beaute luxe']), field] = "mode"
    df.loc[df[field].isin(['industrie']), field] = "industrie"
    df.loc[df[field].isin(['service public']), field] = "service public"
    df.loc[df[field].isin(['autre : education', 'fomation', 'formation']), field] = "education"
    df.loc[df[field].isin(['cloud']), field] = "hebergement"
    df.loc[df[field].isin(['asministration systeme', "sysadmin"]), field] = "adm systeme"
    df.loc[df[field].isin(['batiment', "btp"]), field] = "btp"
    df.loc[df[field].isin(['chomage']), field] = "sans emploi"
    df.loc[df[field].isin(['voyage']), field] = "tourisme"
    df.loc[df[field].isin(['securite']), field] = "securite informatique"
    df.loc[df[field].isin(['ia', 'robotique']), field] = "IA / robotique"
    df.loc[df[field].isin(
        ['sts', 'service public', 'nc', 'autre', 'neant', 'boulangerie', 'autopartage']), field] = "unknown"
    df.loc[df[field].isin(['aero']), field] = 'aeronotique'
    df.loc[df[field].isin(['domotique']), field] = 'iot'
    df.loc[df[field].isin(['spacial']), field] = 'aeronotique'
    df.loc[df[field].isin(['asset management']), field] = 'banque / assurance / finance'
    df.loc[df[field].isin([r"cloud [?]"]), field] = 'cloud computing'
    df.loc[df[field].isin(['industrie & defense']), field] = 'defense'
    df.loc[df[field].isin(['conseil en informatique']), field] = 'conseil'
    df.loc[df[field].isin(['courtage']), field] = 'banque / assurance / finance'
    df.loc[df[field].isin(['btp']), field] = 'batiment'
    df.loc[df[field].isin(['trading']), field] = 'banque / assurance / finance'
    df.loc[df[field].isin(['art']), field] = 'culture'
    df.loc[df[field].isin(['paie']), field] = 'admnistration'
    df.loc[df[field].isin(['vente']), field] = 'commerce'
    df.loc[df[field].isin(['hebergement']), field] = 'cloud computing'
    df.loc[df[field].isin(['hotel']), field] = 'hotellerie'
    df.loc[df[field].isin(['energie et transport']), field] = 'energie'
    df.loc[df[field].isin(['publicite']), field] = 'communication / publicite'
    df.loc[df[field].isin(['pub']), field] = 'communication / publicite'
    df.loc[df[field].isin(['cyber-securite']), field] = 'securite informatique'
    df.loc[df[field].isin(['cybersecurite']), field] = 'securite informatique'
    df.loc[df[field].isin(['advertising']), field] = 'communication / publicite'
    df.loc[df[field].isin(['crm']), field] = 'web'
    df.loc[df[field].isin(['trading & blockchain']), field] = 'banque / assurance / finance'
    df.loc[df[field].isin(['logiciel']), field] = 'developpement logiciel'
    df.loc[df[field].isin(['edition de logiciels applicatifs']), field] = 'developpement logiciel'
    df.loc[df[field].isin(['customer relationship management']), field] = 'relation client'
    df.loc[df[field].isin(['reparation informatique']), field] = 'hardware'
    df.loc[df[field].isin(["aviation d'affaire"]), field] = 'aeronautique'
    df.loc[df[field].isin(['grossiste informatique']), field] = 'retail'
    df.loc[df[field].isin(['recrutement', 'employee engagement software']), field] = 'ressources humaines'
    df.loc[df[field].isin(['publicite et etudes de marche']), field] = 'communication / publicite'
    return df
