#!/bin/bash

set +e #otherwise the script will exit on error
containsElement () {
  local e match="$1"
  shift
  for e; do [[ "$e" == "$match" ]] && return 0; done
  return 1
}

declare -a contribs=("sklearn" "nltk")

for word in "$@"; 
    do containsElement "$word" "${contribs[@]}";
    if [ $? == 0 ]; then
        echo "Installing contrib" "'$word'.";
        poetry config virtualenvs.create false
        poetry install --with="$word"
        echo ""
    else
        echo "AutoGOAL do not support" "'$word'" "so far."
    fi
done