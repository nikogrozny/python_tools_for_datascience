import os
import pandas as pd
import re
import numpy as np
try:
    from PIL import Image
except ImportError:
    import Image
import plotly.io as pio

import plotly.graph_objects as go


import pytesseract
from spellchecker import SpellChecker
spell = SpellChecker(language="fr")
import nltk
from nltk import word_tokenize
nltk.download('punkt')
import pandas as pd
from typing import TextIO, List, Dict, Tuple, Set
import plotly.io as pio
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
from nltk.tokenize import sent_tokenize

path_data = r'C:\Users\Aurelien Pellet\Desktop\Aurelien\epitech\methodo_histoire_nlp\ocr_evaluation'
#%%
df_original = pd.read_csv(os.path.join(path_data,"ocr_original.csv"),sep=";")
df_original_fr = pd.read_csv(os.path.join(path_data,"ocr_original_fr.csv"),sep=";")
df_clean = pd.read_csv(os.path.join(path_data,"ocr_clean.csv"),sep=";")
df_clean_fr = pd.read_csv(os.path.join(path_data,"ocr_clean_fr.csv"),sep=";")
df_bnf = pd.read_csv(os.path.join(path_data,"ocr_bnf.csv"),sep=";")
date = df_original.date
#%%

def clean_text(df: pd.DataFrame) -> (pd.DataFrame , List) :
    metrics : List = []
    for i in range(df.shape[0]) :
        text : str  = " ".join(re.findall("[a-zâêûîôäëüïöùàçéè]+",df.text[i].lower()))
        bag_of_words: List[str] = word_tokenize(text.lower(), language="french")
        bag_of_words = [w for w in bag_of_words if 1 <= len(w) < 22]
        s : pd.Series = pd.Series(bag_of_words)
        metrics.append(len(spell.known(s.values)) / len(s.unique()))
    return df , metrics

def clean_text_1(df: pd.DataFrame) -> (List , List) :
    metrics : List = []
    len_ocr : List = []
    for i in range(df.shape[0]) :
        text : str  = " ".join(re.findall("[a-zâêûîôäëüïöùàçéè\-]+",df.text[i].lower()))
        text = re.sub("([a-z])- ",r"\1",text)
        text = re.sub("\-"," ",text)
        bag_of_words: List[str] = word_tokenize(text.lower(), language="french")
        bag_of_words = [w for w in bag_of_words if 1 <= len(w) < 22]
        s : pd.Series = pd.Series(bag_of_words)
        metrics.append(len(spell.known(s.values)) / len(s.unique()))
        len_ocr.append(len(s))
    return metrics , len_ocr
    
def clean_text_2(df : pd.DataFrame) ->( List , List) :
    metrics : List = []
    name : List = []
    for i in range(df_clean_fr.shape[0]) :
        text = " ".join(re.findall("[\.A-Za-zâêûîôäëüïöùàçéè\-]+",df_bnf.text[i]))
        text = re.sub("([a-z])- ",r"\1",text)
        text = re.sub("\-"," ",text)
        name.append(re.findall("[M]+\. ([A-Z]+[a-zâêûîôäëüïöùàçéè]+(?:\s[A-Z]+[a-zâêûîôäëüïöùàçéè]+)?)",text))
        text = re.sub("[M]+\. ([A-Z]+[a-zâêûîôäëüïöùàçéè]+(?:\s[A-Z]+[a-zâêûîôäëüïöùàçéè]+)?)"," ",text)
        text = text.lower()
        text = re.sub("\."," ",text)
        bag_of_words: List[str] = word_tokenize(text.lower(), language="french")
        bag_of_words = [w for w in bag_of_words if 1 <= len(w) < 22]
        s = pd.Series(bag_of_words)
        metrics.append(len(spell.known(s.values)) / len(s.unique()))
    return metrics , name
    
def name_list(name : List) -> pd.DataFrame :
    l = [list(pd.Series(name[i]).value_counts().index) for i in range(len(name))]
    dft = pd.concat([pd.Series(x) for x in l], axis=1)
    dft.columns = df_bnf.date
    dft.to_csv(os.path.join(path_data,"liste_monsieurs.csv"), encoding="latin-1", sep=";", index=False) 
#%%
metrics_original = clean_text(df_original)[1]
metrics_original_fr = clean_text(df_original_fr)[1]
metrics_clean = clean_text(df_clean)[1]
metrics_clean_fr = clean_text(df_clean_fr)[1]
metrics_clean_fr_1 , len_clean_fr_1 = clean_text_1(df_clean_fr)
metrics_bnf_1 , len_bnf_1 = clean_text_1(df_bnf)
metrics_bnf_2 , name = clean_text_2(df_bnf) 
#%%
def metrics_evaluation_1() :
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        name="OCR - original",
        mode="markers+lines", x=date, y=metrics_original
    ))
    fig.add_trace(go.Scatter(
        name="OCR - clean",
        mode="markers+lines", x=date, y=metrics_clean
    ))
    
    fig.add_trace(go.Scatter(
        name="OCR - clean - fr",
        mode="markers+lines", x=date, y=metrics_clean_fr
    ))
    fig.show()
#%%
def metrics_evaluation_2() : 
    fig = go.Figure()   
    fig.add_trace(go.Scatter(
        name="OCR - original",
        mode="markers+lines", x=date, y=metrics_clean_fr_1
    ))
    fig.add_trace(go.Scatter(
        name="OCR - clean",
        mode="markers+lines", x=date, y=metrics_bnf_1
    ))    
    fig.show()
    
    
def len_ocr() : 
    fig = go.Figure()   
    fig.add_trace(go.Scatter(
        name="#tokens ocr tessaract",
        mode="markers+lines", x=date, y=len_clean_fr_1
    ))
    fig.add_trace(go.Scatter(
        name="#tokens ocr bnf",
        mode="markers+lines", x=date, y=len_bnf_1
    ))    
    pio.write_image(fig,r"C:\Users\Aurelien Pellet\Desktop\Aurelien\epitech\methodo_histoire_nlp\ocr_evaluation\ocr_size.png")
    fig.show()
    
def metrics_evaluation_3() : 
    fig = go.Figure()   
    #fig.add_trace(go.Scatter(
    #    name="OCR - clean - 1",
    #    mode="markers+lines", x=date, y=metrics_clean_fr_1
    #))
    fig.add_trace(go.Scatter(
        name="OCR - bnf - 1",
        mode="markers+lines", x=date, y=metrics_bnf_1,marker_color="crimson"
    ))    
    #fig.add_trace(go.Scatter(
    #    name="OCR - bnf - 2",
    #    mode="markers+lines", x=date, y=metrics_bnf_2
    #))    
    
    
    fig.update_xaxes(
        tickangle = 90,
        title_text = "Date",
        title_font = {"size": 20},
        title_standoff = 25)
    
    #fig.update_yaxes(tickvals=[0.1,0.5,0.9],title_text = "Poids",
    #                 title_font = {"size": 20},title_standoff = 25)

    fig.update_layout(go.Layout(autosize=False,width=1000,height=700),
                      title={
            'text': "",
            'y':0.85,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
                     legend=dict(yanchor="top",y=1,xanchor="left",x=0.9),template="plotly_white") 
    fig.write_html(os.path.join(r"C:\Users\Aurelien Pellet\Desktop\Aurelien\epitech\methodo_histoire_nlp\ocr_evaluation\fig7.png"))
    
    pio.write_image(fig,r"C:\Users\Aurelien Pellet\Desktop\Aurelien\epitech\methodo_histoire_nlp\ocr_evaluation\fig7.png")
    fig.show()
metrics_evaluation_3()
#%%
    
    
#%%
n_vote = []
reg = "VOT[EÉ]"
for i in range(len(df_bnf.text)) :
    print(df_bnf.date[i])
    print(re.findall(reg,df_bnf.text[i]))
    n_vote.append(len(re.findall(reg,df_bnf.text[i])))



fig = go.Figure()


fig.add_trace(go.Scatter(
    name="#vote",
    mode="markers+lines", x=date, y=n_vote
))


pio.write_image(fig,r"C:\Users\Aurelien Pellet\Desktop\Aurelien\epitech\methodo_histoire_nlp\ocr_evaluation\fig8.png")
fig.show()

fig = go.Figure()

n_vote_scale = np.divide(n_vote,np.max(n_vote))
n_vote_inverted = [1-n_vote_scale[i] for i in range(len(n_vote_scale))]

fig.add_trace(go.Scatter(
    name="#vote",
    mode="markers+lines", x=date, y=n_vote_inverted
))

fig.add_trace(go.Scatter(
    name="bnf",
    mode="markers+lines", x=date, y=metrics_bnf_1
))

pio.write_image(fig,r"C:\Users\Aurelien Pellet\Desktop\Aurelien\epitech\methodo_histoire_nlp\ocr_evaluation\fig9.png")
fig.show()

#%%
n_bureau = []
#reg = "bureau\."
reg = "[0-9]+.{0,4}\sbureau\."
for i in range(len(df_bnf.text)) :
    print(df_bnf.date[i])
    print(re.findall(reg,df_bnf.text[i]))
    n_bureau.append(len(re.findall(reg,df_bnf.text[i])))


fig = go.Figure()


fig.add_trace(go.Scatter(
    name="#vote",
    mode="markers+lines", x=date, y=n_bureau
))


#pio.write_image(fig,r"C:\Users\Aurelien Pellet\Desktop\Aurelien\epitech\methodo_histoire_nlp\ocr_evaluation\fig7.png")
fig.show()

#%%

fig = go.Figure()
n = [n+p for n , p in zip(n_vote,n_bureau)]
n_scale = np.divide(n,np.max(n))
n_inverted = [1-n_scale[i] for i in range(len(n_scale))]

fig.add_trace(go.Scatter(
    name="#vote",
    mode="markers+lines", x=date, y=n_inverted
))

fig.add_trace(go.Scatter(
    name="bnf",
    mode="markers+lines", x=date, y=metrics_bnf_1
))

#pio.write_image(fig,r"C:\Users\Aurelien Pellet\Desktop\Aurelien\epitech\methodo_histoire_nlp\ocr_evaluation\fig7.png")
fig.show()


#%%
for i , text in enumerate(df_bnf.text) :
    print(df_bnf.date[i])
    print(re.findall("[Aa]{0,1}nn.{0,6}ver",text))
    
    
    
#%%
metrics : List = []
name : List = []
for i in range(df_clean_fr.shape[0]) :
    text = " ".join(re.findall("[\.A-Za-zâêûîôäëüïöùàçéè\-]+",df_bnf.text[i]))
    text = re.sub("([a-z])- ",r"\1",text)
    text = re.sub("\-"," ",text)
    name.append(re.findall("[M]+\. ([A-Z]+[a-zâêûîôäëüïöùàçéè]+(?:\s[A-Z]+[a-zâêûîôäëüïöùàçéè]+)?)",text))
    text = re.sub("[M]+\. ([A-Z]+[a-zâêûîôäëüïöùàçéè]+(?:\s[A-Z]+[a-zâêûîôäëüïöùàçéè]+)?)"," ",text)
    text = text.lower()
    text = re.sub("\."," ",text)
    bag_of_words: List[str] = word_tokenize(text.lower(), language="french")
    bag_of_words = [w for w in bag_of_words if 1 <= len(w) < 22]
    s = pd.Series(bag_of_words)
l= s.apply(lambda x : len(x)==1).sum()

#%%
metrics : List = []
name : List = []
for i in range(df_clean_fr.shape[0]) :
    text = " ".join(re.findall("[\.A-Za-zâêûîôäëüïöùàçéè\-]+",df_clean_fr.text[i]))
    text = re.sub("([a-z])- ",r"\1",text)
    text = re.sub("\-"," ",text)
    name.append(re.findall("[M]+\. ([A-Z]+[a-zâêûîôäëüïöùàçéè]+(?:\s[A-Z]+[a-zâêûîôäëüïöùàçéè]+)?)",text))
    text = re.sub("[M]+\. ([A-Z]+[a-zâêûîôäëüïöùàçéè]+(?:\s[A-Z]+[a-zâêûîôäëüïöùàçéè]+)?)"," ",text)
    text = text.lower()
    text = re.sub("\."," ",text)
    bag_of_words: List[str] = word_tokenize(text.lower(), language="french")
    bag_of_words = [w for w in bag_of_words if 1 <= len(w) < 22]
    s = pd.Series(bag_of_words)
ll= s.apply(lambda x : len(x)==1).sum()     
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        