#!/bin/bash
contribs=(sklearn nltk gensim regex keras spacy streamlit telegram transformers wikipedia)

for contrib in "${contribs[@]}"
do
  make docker-contrib CONTRIB="$contrib"
done