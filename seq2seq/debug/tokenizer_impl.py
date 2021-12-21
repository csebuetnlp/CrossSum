#%%
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import re
import json

LANG_IDS = ['kirundi', 'indonesian', 'ukrainian', 'spanish', 'arabic', 'kyrgyz', 'thai', 'azerbaijani', 'uzbek', 'igbo', 'french', 'serbian_latin', 'vietnamese', 'marathi', 'pidgin', 'turkish', 'tigrinya', 'punjabi', 'swahili', 'somali', 'nepali', 'hindi', 'telugu', 'persian', 'scottish_gaelic', 'yoruba', 'welsh', 'gujarati', 'serbian_cyrillic', 'korean', 'english', 'sinhala', 'tamil', 'burmese', 'pashto', 'amharic', 'russian', 'japanese', 'urdu', 'portuguese', 'chinese_simplified', 'oromo', 'bengali', 'hausa', 'chinese_traditional']

tokenizer = AutoTokenizer.from_pretrained("google/mt5-base", use_fast=False)
unused_tokens = [k for k in tokenizer.get_vocab() if re.search(r"<0x.*?>", k)]
extra_tokens = [k for k in tokenizer.get_vocab() if re.search(r"<extra_id_.*?>", k)]

unused_langid_map = {}
extra_langid_map = {}

for i, lang_id in enumerate(LANG_IDS):
    unused_langid_map[lang_id] = unused_tokens[i]
    extra_langid_map[lang_id] = extra_tokens[i]

with open("unused_tokens_langid_map.json", "w") as f:
    json.dump(unused_langid_map, f, indent=4, ensure_ascii=False)


with open("extra_tokens_langid_map.json", "w") as f:
    json.dump(extra_langid_map, f, indent=4, ensure_ascii=False)
