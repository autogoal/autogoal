#!/bin/bash

# Initialize our own variables
push_images=0

# Parse command-line options
while getopts ":p" opt; do
  case ${opt} in
    p)
      push_images=1
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      ;;
  esac
done

contribs="$(cd autogoal-contrib/ && ls -d autogoal_* | grep -v 'autogoal_contrib' | sed 's/autogoal_//')"

for contrib in $contribs
do
  docker build . -t autogoal/autogoal:$contrib -f dockerfiles/development/dockerfile --build-arg extras="common $contrib remote" --no-cache
  if [ "$push_images" -eq 1 ]; then
    docker push autogoal/autogoal:$contrib
  fi
done
