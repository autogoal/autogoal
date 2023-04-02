#!/bin/bash
set -e

# Split the first argument into an array of words
contribs=("$@")
poetry config virtualenvs.create false
for arg in "${contribs[@]}"
do
    echo "Trying to install autogoal_$arg"
    case $arg in
        core)
            cd /home/coder/autogoal/autogoal && poetry install
            cd /home/coder/autogoal && pip install -e autogoal
        ;;
        remote)
            cd /home/coder/autogoal/autogoal-remote && poetry install
            cd /home/coder/autogoal && pip install -e autogoal-remote
        ;;
        common)
            cd /home/coder/autogoal/autogoal-contrib/autogoal_contrib && poetry install
            cd /home/coder/autogoal/autogoal-contrib && pip install -e autogoal_contrib
        ;;
        *)
            cd "/home/coder/autogoal/autogoal-contrib/autogoal_$arg" && poetry install
            cd "/home/coder/autogoal/autogoal-contrib" && pip install -e "autogoal_$arg"
        ;;
    esac
done

