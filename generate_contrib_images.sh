#!/bin/bash
contribs="$(cd autogoal-contrib/ && ls -d autogoal_* | grep -v 'autogoal_contrib' | sed 's/autogoal_//')"
for contrib in "${contribs[@]}"
do
  make docker-contrib CONTRIB="$contrib"
done