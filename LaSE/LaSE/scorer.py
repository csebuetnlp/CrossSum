import torch
import logging
import numpy as np
from rouge_score import rouge_scorer
from .utils import load_langid_model, LANG2ISO, FASTTEXT_LANGS
from sentence_transformers import SentenceTransformer
from collections import namedtuple

logger = logging.getLogger(__name__)
LaSEResult = namedtuple("LaSEResult", ("ms", "lc", "lp", "LaSE"))

class LaSEScorer(object):

    def __init__(self, device=None, cache_dir=None):
        if not device:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.labse_model = SentenceTransformer('LaBSE', device=device, cache_folder=cache_dir)
        self.langid_model = load_langid_model(cache_dir)

    def _score_ms(self, targets, predictions, batch_size):
        """Computes batched meaning similarity score"""

        embeddings = self.labse_model.encode(targets + predictions, batch_size=batch_size, show_progress_bar=False)
        return (embeddings[:len(targets)] * embeddings[len(targets):]).sum(axis=1)

    def _score_lc(self, predictions, target_lang):
        """Computes batched language confidence score"""
        target_lang_code = LANG2ISO.get(target_lang, None)

        if not target_lang_code or target_lang_code not in FASTTEXT_LANGS:
            logger.info(f"{target_lang} not reconginzed. language confidence set to 1.0")
            return [1.0] * len(predictions)
        
        all_langs, all_scores = self.langid_model.predict(predictions, k=176, threshold=-1.0)
        columns = np.asarray([langs.index(f"__label__{target_lang_code}") for langs in all_langs])
        rows, all_scores = range(len(all_langs)), np.asarray(all_scores)

        scores = all_scores[rows, columns]
        scores[columns == 0] = 1.0

        return scores

    def _score_lp(self, targets, predictions, target_lang, alpha):
        """Computes batched length penalty score"""
        tokenizer = rouge_scorer.RougeScorer(None, lang=target_lang)._tokenizer
        token_counts = np.asarray([len(tokenizer(s)) for s in targets + predictions])
        target_token_counts = token_counts[:len(targets)]
        prediction_token_counts = token_counts[len(targets):]

        fractions = 1 - (prediction_token_counts / (target_token_counts + alpha))
        return np.exp(fractions * (fractions <= 0.))

    def batched_score(self, targets, predictions, target_lang=None, batch_size=32, alpha=6):
        assert len(targets) == len(predictions)
        batch_size = min(batch_size, len(targets))

        ms_scores = self._score_ms(targets, predictions, batch_size)
        lc_scores = self._score_lc(predictions, target_lang)
        lp_scores = self._score_lp(targets, predictions, target_lang, alpha)
        
        return [
            LaSEResult(ms, lc, lp, ms * lc * lp)
            for ms, lc, lp in zip(ms_scores, lc_scores, lp_scores)
        ]
    
    def score(self, target, prediction, target_lang=None, alpha=6):
        return self.batched_score(
            [target], [prediction], target_lang, 1, alpha
        )[0]