#! /bin/sh

set -e

# Change ownership of data folder to current user
sudo chown -R coder:coder /home/coder/.autogoal

exec $@
