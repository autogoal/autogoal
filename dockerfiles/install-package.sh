#!/bin/bash
set -e
set -x  # Enable printing of commands

# Split the first argument into an array of words
contribs=("$@")
poetry config virtualenvs.create false
for arg in "${contribs[@]}"
do
    echo "Trying to install autogoal_$arg"
    case $arg in
        core)
            cd /home/coder/autogoal/autogoal && sudo poetry install
            cd /home/coder/autogoal && pip install -e autogoal
        ;;
        remote)
            cd /home/coder/autogoal/autogoal-remote && sudo poetry install
            cd /home/coder/autogoal && sudo pip install -e autogoal-remote
        ;;
        common)
            cd /home/coder/autogoal/autogoal-contrib/autogoal_contrib && sudo poetry install
            cd /home/coder/autogoal/autogoal-contrib && sudo pip install -e autogoal_contrib
        ;;
        *)
            cd "/home/coder/autogoal/autogoal-contrib/autogoal_$arg" && sudo poetry install
            cd "/home/coder/autogoal/autogoal-contrib" && sudo pip install -e "autogoal_$arg"
        ;;
    esac
done

