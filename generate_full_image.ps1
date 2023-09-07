$contribs = @("sklearn", "nltk", "gensim", "regex", "keras", "spacy", "streamlit", "telegram", "transformers", "wikipedia")
docker build . -t autogoal/autogoal:all-contribs -f dockerfiles/development/dockerfile --build-arg extras="common $contribs remote" --no-cache
