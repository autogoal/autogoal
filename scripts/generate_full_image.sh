#!/bin/bash

# Initialize our own variables
push_image=0

# Parse command-line options
while getopts ":p" opt; do
  case ${opt} in
    p)
      push_image=1
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      ;;
  esac
done


contribs="$(cd autogoal-contrib/ && ls -d autogoal_* | grep -v 'autogoal_contrib' | sed 's/autogoal_//')"
docker build . -t autogoal/autogoal:full-latest -f dockerfiles/development/dockerfile --build-arg extras="common $contribs remote" --no-cache

if [ "$push_image" -eq 1 ]; then
    docker push autogoal/autogoal:full-latest
    docker rmi autogoal/autogoal:full-latest
  fi