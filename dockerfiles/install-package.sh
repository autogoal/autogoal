#!/bin/bash
set -e

arg=$1 # argument must be either core, remote, common, or a contrib identifier

echo "Trying to install autogoal_$arg"
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
        cd /home/coder/autogoal/autogoal-contrib/autogoal_contrib && poetry install
        cd /home/coder/autogoal/autogoal-contrib && pip install -e autogoal_contrib
    ;;
    sklearn | nltk)
        cd "/home/coder/autogoal/autogoal-contrib/autogoal_$arg" && poetry install
        cd "/home/coder/autogoal/autogoal-contrib" && pip install -e "autogoal_$arg"
    ;;
    *)
        echo "No supported AutoGOAL package with name 'autogoal_$arg'"
    ;;
esac