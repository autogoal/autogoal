#!/bin/bash
arg=$1

poetry config virtualenvs.create false
case $arg in
    core)
        echo "Installing autogoal-core"
        cd /home/coder/autogoal/autogoal && poetry install
        cd /home/coder/autogoal && pip install -e autogoal
    ;;
    remote)
        echo "Installing autogoal-remote"
        cd /home/coder/autogoal/autogoal-remote && poetry install
        cd /home/coder/autogoal && pip install -e autogoal-remote
    ;;
    common)
        echo "Installing autogoal-contrib-common"
        cd /home/coder/autogoal/autogoal-contrib/common && poetry install
        cd /home/coder/autogoal/autogoal-contrib && pip install -e common
    ;;
    sklearn | nltk)
        echo "Installing autogoal-$arg"
        cd "/home/coder/autogoal/autogoal-contrib/$arg" && poetry install
        cd /home/coder/autogoal/autogoal-contrib && pip install -e "$arg"
    ;;
    *)
        echo "Invalid argument"
    ;;
esac