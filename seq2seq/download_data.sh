#!/bin/bash

FILE="dataset.tar.bz2"

if [[ ! -d "dataset" ]]; then
    id="1ywYJEEaFnXIWW5xBwp0cNuPinDwQjCxe"
    curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${id}" > /dev/null
    curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${id}" -o ${FILE}
    rm ./cookie
    tar -xvf ${FILE} && rm ${FILE}
fi
