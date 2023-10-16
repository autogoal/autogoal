#!/bin/bash

for dir in autogoal-contrib/autogoal_*; do 
    if [[ $dir =~ autogoal-contrib/autogoal_(.*) ]]; then 
        contrib=${BASH_REMATCH[1]}; 
        bash scripts/run_specific_contrib_tests.sh "$contrib"; 
    fi; 
done