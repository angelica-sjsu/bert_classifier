#!/usr/bin/env python
# coding: utf-8
from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm


# create embeddings
def sentence_embedder(sentences, dest, phase):
    """
    Creates feature maps(embeddings) for each reviews.
    :params sentences: list of reviews to extract feature maps
    :params dest: folder to save the features maps
    :return None
    """
    model = SentenceTransformer('bert-base-nli-mean-tokens')
    index = 0
    for s in tqdm(sentences):
        embedding = model.encode([s])
        padded_ind = str(index).zfill(6)
        np.save(f'{dest}/{padded_ind}_{phase}_emb.bert', embedding)
        index += 1

