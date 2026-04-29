"""
Reimplementation of the (missing-from-repo) cosine_similarity helper used in
eval/evaluation.ipynb. Returns the pairwise cosine similarity matrix of a
list of texts, computed on a TF-IDF representation.
"""

from typing import List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as _cs


def cosine_sim(texts: List[str]) -> np.ndarray:
    vec = TfidfVectorizer()
    mat = vec.fit_transform(texts)
    return _cs(mat)
