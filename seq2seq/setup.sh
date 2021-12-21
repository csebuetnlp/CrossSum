#!/bin/bash

# install dependecies
pip install -r ../LaSE/requirements.txt --upgrade-strategy only-if-needed --user
python -m unidic download # for japanese segmentation
pip install --upgrade ../LaSE
python -m nltk.downloader punkt

# install transformers and requirements
pip install --upgrade -r requirements.txt
pip install --upgrade transformers/