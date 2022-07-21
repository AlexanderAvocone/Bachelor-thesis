#!/bin/bash
echo "Hostname: $(hostname)"
echo "Arguments: $@"


# setup environment


export SOURCING_DIR=$PWD

#patch to bachelor enviroment
cd /work/aavocone/anaconda3/envs/Bachelor
__conda_setup="$('/work/aavocone/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/work/aavocone/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/work/aavocone/anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="/work/aavocone/anaconda3/bin:$PATH"
    fi
fi
unset __conda_setup

conda activate Bachelor

cd $SOURCING_DIR


# start job                                                                     
python3 /work/aavocone/Code/$1
python3 /work/aavocone/Code/$2


