#!/bin/bash
arg=$1

case $arg in
    remote)
        echo "Installing autogoal-$arg"
        ln -s /home/coder/autogoal/autogoal-"$arg" /usr/local/lib/python3.9/site-packages/autogoal-"$arg"
        cd autogoal-contrib/"$arg" && poetry install
    ;;
    sklearn | nltk)
        echo "Installing autogoal-$arg"
        ln -s /home/coder/autogoal/autogoal-"$arg" /usr/local/lib/python3.9/site-packages/autogoal-"$arg"
        cd autogoal-contrib/"$arg" && poetry install
    ;;
    *)
        echo "Invalid argument"
    ;;
esac