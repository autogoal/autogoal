#!/bin/bash
arg=$1

case $arg in
    remote)
        echo "Installing autogoal-$arg"
        echo "linking /home/coder/autogoal/autogoal-remote to /usr/local/lib/python3.9/site-packages/autogoal_remote"
        ln -s /home/coder/autogoal/autogoal-remote /usr/local/lib/python3.9/site-packages/autogoal_remote
        cd /home/coder/autogoal/autogoal-remote/ && poetry install
    ;;
    sklearn | nltk)
        echo "Installing autogoal-$arg"
        echo "linking /home/coder/autogoal/autogoal-contrib/$arg to /usr/local/lib/python3.9/site-packages/autogoal_$arg"
        ln -s /home/coder/autogoal/autogoal-contrib/$arg /usr/local/lib/python3.9/site-packages/autogoal_$arg
        cd /home/coder/autogoal/autogoal-contrib/$arg && poetry install
    ;;
    *)
        echo "Invalid argument"
    ;;
esac