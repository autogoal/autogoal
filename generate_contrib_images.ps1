$contribs = @("sklearn", "nltk", "gensim", "regex", "keras", "spacy", "streamlit", "telegram", "transformers", "wikipedia")

foreach ($contrib in $contribs) {
  make docker-contrib CONTRIB="$contrib"
}