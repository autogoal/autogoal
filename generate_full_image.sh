#!/bin/bash
contribs="common remote $(cd autogoal-contrib/ && ls -d autogoal_* | grep -v 'autogoal_contrib' | sed 's/autogoal_//')"
docker build . -t autogoal/autogoal:all-contribs -f dockerfiles/development/dockerfile --build-arg extras="common $contribs remote" --no-cache