import os
import re
try:
    from PIL import Image
except ImportError:
    import Image
import pytesseract
from spellchecker import SpellChecker
import nltk
from nltk import word_tokenize
nltk.download('punkt')
import pandas as pd
from typing import TextIO, List, Dict, Tuple, Set
import plotly.io as pio
pytesseract.pytesseract.tesseract_cmd = r'insert_path_to_tesseract'
from spellchecker import SpellChecker
spell = SpellChecker(language="fr")  # loads default word frequency list
path_data = os.getcwd()


#%%


original_documents = []
original_dates = []
i=0
for name in os.listdir(os.path.join(path_data,"img","img_original")) : 
    if not os.path.isfile(os.path.join(path_data,"ocr", "ocr_clean",os.path.splitext(name)[0]+".txt")) :
        try : 
            f = open(os.path.join(path_data,"ocr","ocr_clean",os.path.splitext(name)[0]+".txt"), "x")
            f.write(pytesseract.image_to_string(Image.open(os.path.join(path_data,"img","img_original",name))))
            f.close()
            print(name," : OCR fini")
        except : 
            print(name," : error")
    else :
        print(name," : déjà OCRisé")
        
        
#%%
        

original_documents = []
original_dates = []
i=0
for name in os.listdir(os.path.join(path_data,"img","img_clean")) : 
    if not os.path.isfile(os.path.join(path_data,"ocr", "ocr_clean_1",os.path.splitext(name)[0]+".txt")) :
        try : 
            f = open(os.path.join(path_data,"ocr","ocr_clean_1",os.path.splitext(name)[0]+".txt"), "x")
            f.write(pytesseract.image_to_string(Image.open(os.path.join(path_data,"img","img_clean",name))))
            f.close()
            print(name," : OCR fini")
        except : 
            print(name," : error")
    else :
        print(name," : déjà OCRisé")
        
#%%
        
        
original_documents = []
original_dates = []
i=0
for name in os.listdir(os.path.join(path_data,"img","img_clean")) : 
    if not os.path.isfile(os.path.join(path_data,"ocr", "ocr_clean_fr",os.path.splitext(name)[0]+".txt")) :
        try : 
            f = open(os.path.join(path_data,"ocr","ocr_clean_fr",os.path.splitext(name)[0]+".txt"), "x")
            f.write(pytesseract.image_to_string(Image.open(os.path.join(path_data,"img","img_clean",name),lang="fr")))
            f.close()
            print(name," : OCR fini")
        except : 
            print(name," : error")
    else :
        print(name," : déjà OCRisé")

#%%
        

original_documents = []
original_dates = []
i=0
for name in os.listdir(os.path.join(path_data,"img","img_original")) : 
    if not os.path.isfile(os.path.join(path_data,"ocr", "ocr_original_fr",os.path.splitext(name)[0]+".txt")) :
        try : 
            f = open(os.path.join(path_data,"ocr","ocr_original_fr",os.path.splitext(name)[0]+".txt"), "x")
            f.write(pytesseract.image_to_string(Image.open(os.path.join(path_data,"img","img_original",name))))
            f.close()
            print(name," : OCR fini")
        except : 
            print(name," : error")
    else :
        print(name," : déjà OCRisé")
        
#%%
document = []
date = []
year = []
month = []
day = []
page = []
for name in os.listdir(os.path.join(path_data,"ocr","ocr_clean")) :
    print(name)
    f = open(os.path.join(path_data,"ocr","ocr_clean",name), "r")
    document.append(str(f.read()))
    date.append(re.split("img_|\.",name)[1])
    year.append(re.split("_",name)[1])
    
    m = re.split("_",name)[2]
    if len(m) == 1 :
        m = "0"+m
    month.append(m)
    
    d = re.split("_",name)[3]
    if len(d) == 1 :
        d = "0" + d
    day.append(d)

df = pd.DataFrame({"date":date,"year":year,"month":month,"day":day,"text":document})
df = df.groupby(["year","month",'day'])["text"].apply(lambda x : " ".join(x)).reset_index()
df.loc[:,"date"]=df.year+"-"+df.month+"-"+df.day
df = df.sort_values(by="date")
df.to_csv(os.path.join(path_data,"ocr_original.csv"), encoding="utf-8", sep=";", index=False) 
    
#%%
document = []
date = []
year = []
month = []
day = []
page = []
for name in os.listdir(os.path.join(path_data,"ocr","ocr_clean_1")) :
    print(name)
    f = open(os.path.join(path_data,"ocr","ocr_clean_1",name), "r")
    document.append(str(f.read()))
    date.append(re.split("img_|\.",name)[1])
    year.append(re.split("_",name)[1])
    
    m = re.split("_",name)[2]
    if len(m) == 1 :
        m = "0"+m
    month.append(m)
    
    d = re.split("_",name)[3]
    if len(d) == 1 :
        d = "0" + d
    day.append(d)

df_clean = pd.DataFrame({"date":date,"year":year,"month":month,"day":day,"text":document})
df_clean = df_clean.groupby(["year","month",'day'])["text"].apply(lambda x : " ".join(x)).reset_index()
df_clean.loc[:,"date"]=df_clean.year+"-"+df_clean.month+"-"+df_clean.day
df_clean = df_clean.sort_values(by="date")
df_clean.to_csv(os.path.join(path_data,"ocr_clean.csv"), encoding="utf-8", sep=";", index=False) 

#%%
document = []
date = []
year = []
month = []
day = []
page = []
for name in os.listdir(os.path.join(path_data,"ocr","ocr_clean_fr")) :
    print(name)
    f = open(os.path.join(path_data,"ocr","ocr_clean_fr",name), "r")
    document.append(str(f.read()))
    date.append(re.split("img_|\.",name)[1])
    year.append(re.split("_",name)[1])
    
    m = re.split("_",name)[2]
    if len(m) == 1 :
        m = "0"+m
    month.append(m)
    
    d = re.split("_",name)[3]
    if len(d) == 1 :
        d = "0" + d
    day.append(d)

df_clean_fr = pd.DataFrame({"date":date,"year":year,"month":month,"day":day,"text":document})
df_clean_fr = df_clean_fr.groupby(["year","month",'day'])["text"].apply(lambda x : " ".join(x)).reset_index()
df_clean_fr.loc[:,"date"]=df_clean_fr.year+"-"+df_clean_fr.month+"-"+df_clean_fr.day
df_clean_fr = df_clean_fr.sort_values(by="date")
df_clean_fr.to_csv(os.path.join(path_data,"ocr_clean_fr.csv"), encoding="utf-8", sep=";", index=False) 
#%%
document = []
date = []
year = []
month = []
day = []
page = []
for name in os.listdir(os.path.join(path_data,"ocr","ocr_original_fr")) :
    print(name)
    f = open(os.path.join(path_data,"ocr","ocr_original_fr",name), "r")
    document.append(str(f.read()))
    date.append(re.split("img_|\.",name)[1])
    year.append(re.split("_",name)[1])
    
    m = re.split("_",name)[2]
    if len(m) == 1 :
        m = "0"+m
    month.append(m)
    
    d = re.split("_",name)[3]
    if len(d) == 1 :
        d = "0" + d
    day.append(d)

df_fr = pd.DataFrame({"date":date,"year":year,"month":month,"day":day,"text":document})
df_fr = df_fr.groupby(["year","month",'day'])["text"].apply(lambda x : " ".join(x)).reset_index()
df_fr.loc[:,"date"]=df_fr.year+"-"+df_fr.month+"-"+df_fr.day
df_fr = df_fr.sort_values(by="date")
df_fr.to_csv(os.path.join(path_data,"ocr_original_fr.csv"), encoding="utf-8", sep=";", index=False) 

#%%
document = []
date = []
year = []
month = []
day = []
page = []
for name in os.listdir(os.path.join(path_data, "\ocr_sorted")) :
    if re.split("\.",name)[0] in df.date.values : 
        print(name)
        f = open(os.path.join(path_data, "ocr_sorted", name), "r", encoding="utf-8")
        document.append(str(f.read()))
        date.append(re.split("img_|\.", name)[0])
        year.append(re.split("-", name)[0])

        m = re.split("-", name)[1]
        if len(m) == 1 :
            m = "0"+m
        month.append(m)

        d = re.split("-", name)[2]
        if len(d) == 1 :
            d = "0" + d
        day.append(d)
df_bnf = pd.DataFrame({"date":date,"year":year,"month":month,"day":day,"text":document})
df_bnf.to_csv(os.path.join(path_data,"ocr_bnf.csv"), encoding="utf-8", sep=";", index=False) 
#%%
metrics_ocr_original = []
for i in range(df.shape[0]) :
    text = " ".join(re.findall("[a-zâêûîôäëüïöùàçéè\-]+",df.text[i].lower()))
    bag_of_words: List[str] = word_tokenize(text.lower(), language="french")
    bag_of_words = [w for w in bag_of_words if 1 <= len(w) < 22]
    s = pd.Series(bag_of_words)
    metrics_ocr_original.append(len(spell.known(s.values)) / len(s.unique()))

metrics_ocr_original_fr = []

for i in range(df_fr.shape[0]) :
    text = " ".join(re.findall("[a-zâêûîôäëüïöùàçéè\-]+",df.text[i].lower()))
    bag_of_words: List[str] = word_tokenize(text.lower(), language="french")
    bag_of_words = [w for w in bag_of_words if 1 <= len(w) < 22]
    s = pd.Series(bag_of_words)
    metrics_ocr_original_fr.append(len(spell.known(s.values)) / len(s.unique()))

metrics_ocr_clean = []
for i in range(df_clean.shape[0]) :
    text = " ".join(re.findall("[a-zâêûîôäëüïöùàçéè\-]+",df_clean.text[i].lower()))
    bag_of_words: List[str] = word_tokenize(text.lower(), language="french")
    bag_of_words = [w for w in bag_of_words if 1 <= len(w) < 22]
    s = pd.Series(bag_of_words)
    metrics_ocr_clean.append(len(spell.known(s.values)) / len(s.unique()))
    
metrics_ocr_clean_fr = []
for i in range(df_clean.shape[0]) :
    text = " ".join(re.findall("[a-zâêûîôäëüïöùàçéè\-]+",df_clean_fr.text[i].lower()))
    bag_of_words: List[str] = word_tokenize(text.lower(), language="french")
    bag_of_words = [w for w in bag_of_words if 1 <= len(w) < 22]
    s = pd.Series(bag_of_words)
    metrics_ocr_clean_fr.append(len(spell.known(s.values)) / len(s.unique()))

metrics_ocr_original_fr_1 = []
len_ocr_original_fr_1 = []
for i in range(df_clean.shape[0]) :
    text = " ".join(re.findall("[a-zâêûîôäëüïöùàçéè\-]+",df_fr.text[i].lower()))
    text = re.sub("([a-z])- ",r"\1",text)
    text = re.sub("\-"," ",text)
    bag_of_words: List[str] = word_tokenize(text.lower(), language="french")
    bag_of_words = [w for w in bag_of_words if 1 <= len(w) < 22]
    s = pd.Series(bag_of_words)
    len_ocr_original_fr_1.append(len(s))
    metrics_ocr_original_fr_1.append(len(spell.known(s.values)) / len(s.unique()))   
    
    
metrics_ocr_clean_fr_1 = []
len_ocr_clean_fr_1 = []
for i in range(df_clean.shape[0]) :
    text = " ".join(re.findall("[a-zâêûîôäëüïöùàçéè\-]+",df_clean_fr.text[i].lower()))
    text = re.sub("([a-z])- ",r"\1",text)
    text = re.sub("\-"," ",text)
    bag_of_words: List[str] = word_tokenize(text.lower(), language="french")
    bag_of_words = [w for w in bag_of_words if 1 <= len(w) < 22]
    s = pd.Series(bag_of_words)
    len_ocr_clean_fr_1.append(len(s))
    metrics_ocr_clean_fr_1.append(len(spell.known(s.values)) / len(s.unique()))   
    
metrics_ocr_bnf = []
len_ocr_bnf = []
for i in range(df_bnf.shape[0]) :
    text = " ".join(re.findall("[a-zâêûîôäëüïöùàçéè\-]+",df_bnf.text[i].lower()))
    #text = re.sub("([a-z])- ",r"\1",text)
    text = re.sub("([a-z])- ",r"\1",text)
    text = re.sub("\-"," ",text)
    bag_of_words: List[str] = word_tokenize(text.lower(), language="french")
    bag_of_words = [w for w in bag_of_words if 1 <= len(w) < 22]
    s = pd.Series(bag_of_words)
    len_ocr_bnf.append(len(s))
    metrics_ocr_bnf.append(len(spell.known(s.values)) / len(s.unique()))   
    
#%%
import plotly.graph_objects as go

import pandas as pd

fig = go.Figure()


fig.add_trace(go.Scatter(
    name="OCR - original",
    mode="markers+lines", x=df["date"], y=metrics_ocr_original
))

fig.add_trace(go.Scatter(
    name="OCR - clean",
    mode="markers+lines", x=df["date"], y=metrics_ocr_clean
))


fig.show()

#%%
import plotly.graph_objects as go

import pandas as pd

fig = go.Figure()


fig.add_trace(go.Scatter(
    name="OCR - original",
    mode="markers+lines", x=df["date"], y=metrics_ocr_original
))

fig.add_trace(go.Scatter(
    name="OCR - clean",
    mode="markers+lines", x=df["date"], y=metrics_ocr_clean
))


fig.add_trace(go.Scatter(
    name="OCR - clean - fr",
    mode="markers+lines", x=df["date"], y=metrics_ocr_clean_fr
))


fig.show()
    #%%
import plotly.graph_objects as go

import pandas as pd

fig = go.Figure()



fig.add_trace(go.Scatter(
    name="OCR - clean - fr",
    mode="markers+lines", x=df["date"], y=metrics_ocr_clean_fr
))

fig.add_trace(go.Scatter(
    name="OCR - bnf",
    mode="markers+lines", x=df["date"], y=metrics_ocr_bnf
))


fig.show()

#%%

import plotly.graph_objects as go

import pandas as pd

fig = go.Figure()



fig.add_trace(go.Scatter(
    name="OCR - clean - fr - 1",
    mode="markers+lines", x=df["date"], y=metrics_ocr_clean_fr_1
))

fig.add_trace(go.Scatter(
    name="OCR - bnf",
    mode="markers+lines", x=df["date"], y=metrics_ocr_bnf
))


fig.show()

#%%
for i in range(df.shape[0]) :
    text = " ".join(re.findall("[a-zâêûîôäëüïöùàçéè\-]+",df.text[i].lower()))
    bag_of_words: List[str] = word_tokenize(text.lower(), language="french")
    bag_of_words = [w for w in bag_of_words if 1 <= len(w) < 22]
    s = pd.Series(bag_of_words)
    metrics_ocr_original.append(len(spell.known(s.values)) / len(s.unique()))
#%%
metrics_ocr_bnf_M = []  
list_mr = []  
    
for i in range(df_clean_fr.shape[0]) :
    text = " ".join(re.findall("[\.A-Za-zâêûîôäëüïöùàçéè\-]+",df_bnf.text[i]))
    text = re.sub("([a-z])- ",r"\1",text)
    text = re.sub("\-"," ",text)
    list_mr.append(re.findall("[M]+\. ([A-Z]+[a-zâêûîôäëüïöùàçéè]+(?:\s[A-Z]+[a-zâêûîôäëüïöùàçéè]+)?)",text))
    text = re.sub("[M]+\. ([A-Z]+[a-zâêûîôäëüïöùàçéè]+(?:\s[A-Z]+[a-zâêûîôäëüïöùàçéè]+)?)"," ",text)
    text = text.lower()
    text = re.sub("\."," ",text)
    bag_of_words: List[str] = word_tokenize(text.lower(), language="french")
    bag_of_words = [w for w in bag_of_words if 1 <= len(w) < 22]
    s = pd.Series(bag_of_words)
    metrics_ocr_bnf_M.append(len(spell.known(s.values)) / len(s.unique()))
    























