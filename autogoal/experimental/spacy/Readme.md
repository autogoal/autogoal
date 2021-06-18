======================================================================
Comments

This project includes two new python files to contrib to Autogoal Project
Files: __init__.py to iniciate a library using _manual.py
In _manual.py two classes included SpacyTagger and SpacyNER
SpacyTagger includes a self_tokenizer for tokenization (original by authors of autogoal) and a couple or functions from spacy 3 APi
SpacyNER is a Named Entity Recognizer using Spacy 3 where the user can add custom entities

======================================================================
Requirements

Usage requirements for NER project

Install Spacy 3 run commands
$ pip install -U pip wheel
$ pip install -U spacy
$ python -m spacy download en_core_web_sm / es_core_news_sm #choose a language

Install Pickle
$pip install pickle

Download pretrained data sets for spacy /* At least */

Dataset 		Domain
---------------------------
wikigold        Wikipedia 
SEC-filings     Finances

======================================================================
Usage

For the SPacyNER class run run inherited from AlgorithmBase
Parameters specified in comments of the method run in class SpacyNER
To test there is a Test folder with two python files to test
The folder ./data contains a pretrained corpus for spacy named wikigold.conll.txt with domain Wikipedia
To change the pretrained corpus copy from ./recognition-data-sets-master "name".conll.txt to ./data
and change run.bat and test_add_ent to load a different data set
The file entity recognition data sets master contains several pretained corpuses


======================================================================
