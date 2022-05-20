# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# (!) Logo modified by AutoGOAL contributors
# ==============================================================================

# Change ownership of datasets folder
sudo chown -R coder:coder /home/coder/.autogoal

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/usr/local/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/usr/local/etc/profile.d/conda.sh" ]; then
        . "/usr/local/etc/profile.d/conda.sh"
    else
        export PATH="/usr/local/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

export PS1="\[\e[31m\]autogoal-docker\[\e[m\] \[\e[33m\]\w\[\e[m\] > "
export TERM=xterm-256color
alias grep="grep --color=auto"
alias ls="ls --color=auto"
alias poetry="sudo poetry"
alias autogoal="python3 -m autogoal"

echo -e "\e[1;31m"
cat<<TF
     ^         _         ____  ___    ^    _     
    / \  _   _| |_ ___  / ___|/ _ \  / \  | |    
   / _ \| | | | __/ _ \| |_ _| | | |/ _ \ | |    
  / ___ \ |_| | || (_) | |_| | |_| / ___ \| |___ 
 /_/   \_\__,_|\__\___/ \____|\___/_/   \_\_____|

TF
echo -e "\e[0;33m"

if [[ $EUID -eq 0 ]]; then
  cat <<WARN
WARNING: You are running this container as root, which can cause new files in
mounted volumes to be created as the root user on your host machine.

To avoid this, run the container by specifying your user's userid:

$ docker run -u \$(id -u):\$(id -g) args...
WARN
else
  cat <<EXPL
You are running this container as user with ID $(id -u) and group $(id -g),
which should map to the ID and group for your user on the Docker host. Great!
EXPL
fi

# Turn off colors
echo -e "\e[m"

conda activate autogoal