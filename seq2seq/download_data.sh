#!/bin/bash

FILE="dataset.tar.bz2"

if [[ ! -d "dataset" ]]; then
    id="1bwURjAyQT6OkGRd_f9mwkWg9FABa_c6S"
    cert="https://docs.google.com/uc?export=download&id=${id}"
    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate ${cert} -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=${id}" -O ${FILE}
    rm -rf /tmp/cookies.txt
    tar -xvf ${FILE} && rm ${FILE}
fi
