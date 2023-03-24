#!/bin/bash
set -e

arg=$1 # argument must be either core, remote, common, or a contrib identifier

echo "Trying to install autogoal-$arg"
poetry config virtualenvs.create false
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
        cd /home/coder/autogoal/autogoal-contrib/common && poetry install
        cd /home/coder/autogoal/autogoal-contrib && pip install -e autogoal-contrib-common
    ;;
    sklearn | nltk)
        cd "/home/coder/autogoal/autogoal-contrib/autogoal-$arg" && poetry install
        cd "/home/coder/autogoal/autogoal-contrib" && pip install -e "autogoal-$arg"
    ;;
    *)
        echo "No supported AutoGOAL package with name 'autogoal-$arg'"
    ;;
esac