import spacy
from spacy import displacy
import pickle


nlp = spacy.load("en_core_web_sm")

tokenized = nlp("Fidel Castro is the President of Cuba")

displacy.serve(tokenized, style='ent')
#if Windows open localhost:5000 else recomended by console
