#!/bin/bash
contribs="$(cd autogoal-contrib/ && ls -d autogoal_* | grep -v 'autogoal_contrib' | sed 's/autogoal_//')"
docker build . -t autogoal/autogoal:all-contribs-latest -f dockerfiles/development/dockerfile --build-arg extras="common $contribs remote" --no-cache2