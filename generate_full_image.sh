#!/bin/bash
contribs=(sklearn nltk gensim regex spacy streamlit telegram keras transformers wikipedia)
docker build . -t autogoal/autogoal:all-contribs -f dockerfiles/development/dockerfile --build-arg extras="common $contribs remote" --no-cache