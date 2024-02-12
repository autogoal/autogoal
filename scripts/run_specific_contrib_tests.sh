#!/bin/bash

contrib=$1
dir="autogoal-contrib/autogoal_${contrib}/autogoal_${contrib}/"

echo "${dir}tests"

if [ ! -d "${dir}tests" ]; then
    echo "No tests to run for autogoal_${contrib}"
    exit 0
fi

if python -c "import pkgutil; exit(not pkgutil.find_loader('autogoal_${contrib}'))"; then
    echo "Running tests for autogoal_${contrib}"
    pytest ${dir}tests
else
    echo "autogoal_${contrib} is not installed. Skipping tests..."
fi
