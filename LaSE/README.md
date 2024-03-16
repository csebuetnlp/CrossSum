# Language-agnostic Summary Evaluation (LaSE)

## Overview

The standard evaluation metric for abstractive summarization, namely ROUGE, assesses the quality of a summary using the lexical overlap between the summary and reference text. However, when doing cross-lingual summarisation, it may very well be the case that we don't have the reference summary in the target language, but in the source language. In this case, we can't use ROUGE since the source and target languages can be of different scripts. To alleviate this problem and as an alternative to ROUGE, we propose `LaSE`, which can effectively compute summary quality in both of the above cases. `LaSE` is computed using three components:

* Meaning similarity
* Language confidence
* Length Penalty

To know how each of these components are evaluated, refer to the paper.

## Installation

```bash
pip3 install -r requirements.txt
python3 -m unidic download
pip3 install --upgrade ./
```

## Usage

```python
from LaSE import LaSEScorer 
scorer = LaSEScorer()

ref_text = """reference text"""
pred_text = """prediction text"""
ref_lang = "reference language name" # see the list of language names below

score = scorer.score(
    ref_text,
    pred_text,
    target_lang=ref_lang # language name of the reference text
)

print(score)
>>> LaSEResult(ms=0.89, lc=0.92, lp=0.98, LaSE=0.802424)


# with list of sentences
list_of_references = ["reference1", "reference2", ...]
list_of_predictions = ["predictions1", "predictions2", ...]

scores = scorer.batched_score(
    list_of_references,
    list_of_predictions,
    target_lang=ref_lang,
    batch_size=32
)

>>> [LaSEResult(ms=0.89, lc=0.92, lp=0.98, LaSE=0.802424), LaSEResult(ms=0.89, lc=0.92, lp=0.98, LaSE=0.802424), ...] 

```

* Available language names: `oromo`, `french`, `amharic`, `arabic`, `azerbaijani`, `bengali`, `burmese`, `chinese_simplified`, `chinese_traditional`, `welsh`, `english`, `kirundi`, `gujarati`, `hausa`, `hindi`, `igbo`, `indonesian`, `japanese`, `korean`, `kyrgyz`, `marathi`, `spanish`, `scottish_gaelic`, `nepali`, `pashto`, `persian`, `pidgin`, `portuguese`, `punjabi`, `russian`, `serbian_cyrillic`, `serbian_latin`, `sinhala`, `somali`, `swahili`, `tamil`, `telugu`, `thai`, `tigrinya`, `turkish`, `ukrainian`, `urdu`, `uzbek`, `vietnamese`, `yoruba`
  

* ***Note: If the reference language name is not provided or recognized, language confidence will be set to 1.0.***
