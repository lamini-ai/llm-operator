#!/bin/bash

# Safely execute this bash script
# e exit on first failure
# x all executed commands are printed to the terminal
# u unset variables are errors
# a export all variables to the environment
# E any trap on ERR is inherited by shell functions
# -o pipefail | produces a failure code if any stage fails
set -Eeuoxa pipefail

# Get the directory of this script
LOCAL_DIRECTORY="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# build
$LOCAL_DIRECTORY/scripts/build.sh

NETWORK=""

while getopts l flag; do
    case "${flag}" in
        l) NETWORK=--network='host';;
    esac
done

docker run $NETWORK -v ~/.powerml:/root/.powerml -v $LOCAL_DIRECTORY/data:/app/llm_operator/data -v $LOCAL_DIRECTORY/models:/app/llm_operator/models -it --rm --entrypoint /app/llm_operator/scripts/start-customer-support.sh llm_operator:latest "$@"