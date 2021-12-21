import torch
import logging
import numpy as np
from rouge_score import rouge_scorer
from .utils import load_langid_model, LANG2ISO, FASTTEXT_LANGS
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class LaSEScorer(object):

    def __init__(self, device=None, cache_dir=None):
        if not device:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.labse_model = SentenceTransformer('LaBSE', device=device, cache_folder=cache_dir)
        self.langid_model = load_langid_model(cache_dir)

    def _score_ms(self, target, prediction):
        """Computes meaning similarity score"""

        target_emb = self.labse_model.encode(target, show_progress_bar=False)
        prediction_emb = self.labse_model.encode(prediction, show_progress_bar=False)

        return target_emb.dot(prediction_emb)

    def _score_lc(self, prediction, target_lang):
        """Computes language confidence score"""
        
        target_lang_code = LANG2ISO.get(target_lang, None)

        if not target_lang_code or target_lang_code not in FASTTEXT_LANGS:
            logger.info(f"{target_lang} not reconginzed. language confidence set to 1.0")
            return 1.0

        langs, scores = self.langid_model.predict(prediction, k=176, threshold=-1.0)
        idx = langs.index(f"__label__{target_lang_code}")

        return 1.0 if idx == 0 else scores[idx]

    def _score_lp(self, target, prediction, target_lang, alpha):
        """Computes length penalty score"""
        tokenizer = rouge_scorer.RougeScorer(None, lang=target_lang)._tokenizer
        target_token_count = len(tokenizer(target))
        prediction_token_count = len(tokenizer(prediction))

        if prediction_token_count <= target_token_count + alpha:
            score = 1.0
        else:
            score = np.exp(1 - (prediction_token_count / (target_token_count + alpha)))

        return score


    def score(
        self, 
        target, 
        prediction,
        target_lang=None,
        alpha=6
    ):
        return (
            self._score_ms(target, prediction)
            * self._score_lc(prediction, target_lang)
            * self._score_lp(target, prediction, target_lang, alpha)
        )
    