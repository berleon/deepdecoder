#! /usr/bin/env bash

REPOS=".."
PIP_ARGS=" --user "

set -e


if [ "$1" == "clone" ]; then
    cd $REPOS
    git clone  git@github.com:berleon/pybeesgrid.git
    git clone  git@github.com:berleon/deepdecoder.git
    git clone  git@github.com:berleon/beras.git
fi

if [ "$1" == "--help" ]; then
    echo "$0 [clone]       helper script to setup development enviroment"
    echo "Change the REPOS variable to the path, where your repositories will be stored."
    echo "Current value is REPOS=`realpath $REPOS`"
    echo "Arguments"
    echo "      clone       Downloads all repositories from github"
fi

DEEPDECODER=$(realpath "$REPOS/deepdecoder")
PYBEESGRID=$(realpath "$REPOS/pybeesgrid")
BERAS=$(realpath "$REPOS/beras")


cd $BERAS
pip install -e . $PIP_ARGS

cd $PYBEESGRID
mkdir -p build
cd build
cmake .. && make -j `nproc`
make create_python_pkg
cd $PYBEESGRID/build/python/package

pip install --upgrade --no-deps . $PIP_ARGS
cd $DEEPDECODER
pip install -e . $PIP_ARGS

