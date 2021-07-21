#!/bin/bash

if ! [[ -d "venv" ]]
then
    echo "##################################################"
    echo "# 🚀 Create virtualenv and install requirements  #"
    echo "##################################################"
    echo

    python -m venv venv/

    # Enable the virtualenv
    source venv/Scripts/activate

    # Install the dependencies if needed
    python -m pip install -r requirements.txt
else
    # Enable the virtualenv
    source venv/Scripts/activate

    echo "##################################################"
    echo "# 🚀 Install requirements                        #"
    echo "##################################################"
    echo

    # Install the dependencies if needed
    python -m pip install -q -r requirements.txt
fi
