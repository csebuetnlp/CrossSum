#!/bin/bash

FILE="CrossSum_v1.0.tar.bz2"

if [[ ! -d "dataset" ]]; then
    id="11yCJxK5necOyZBxcJ6jncdCFgNxrsl4m"
    cert="https://docs.google.com/uc?export=download&id=${id}"
    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate ${cert} -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=${id}" -O ${FILE}
    rm -rf /tmp/cookies.txt
    tar -xvf ${FILE} && rm ${FILE}
fi
