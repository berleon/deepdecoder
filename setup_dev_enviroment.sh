#! /usr/bin/env bash

REPOS=".."
PIP_ARGS=""
SUDO="sudo "

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

$SUDO pip install git+https://github.com/EderSantana/seya.git@8567d2715706b94d72da4d6a3864aae094be3951#egg=seya $PIP_ARGS

cd $BERAS
$SUDO pip install -e . $PIP_ARGS

cd $PYBEESGRID
mkdir -p build
cd build
cmake .. && make -j `nproc`
make create_python_pkg
cd $PYBEESGRID/build/python/package
$SUDO pip install --upgrade --no-deps . $PIP_ARGS

cd $DEEPDECODER
$SUDO pip install -e . $PIP_ARGS

