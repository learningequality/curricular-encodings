import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text
import numpy as np
from collections import defaultdict
from pandas import DataFrame
import os
import pickle
# import labse
# import cmlm

from functools import lru_cache

embedding_cache = defaultdict(dict)
abbrev_to_model = {
    "useml3": "universal-sentence-encoder-multilingual-large/3",
    "use4": "universal-sentence-encoder/4",
    "useq3": "universal-sentence-encoder-qa/3",
    "l1": "LaBSE/1",
    "usel5": "universal-sentence-encoder-large/5",
    "usem3": "universal-sentence-encoder-multilingual/3",
    # "usecmb1": "universal-sentence-encoder-cmlm/multilingual-base/1",
    # "usecmbb1": "universal-sentence-encoder-cmlm/multilingual-base-br/1",
}
DEFAULT_MODEL = "useml3"
DEFAULT_CHUNKSIZE = 1000

embeddings = {}


def save_embeddings():
    print("Saving embeddings... ", end="")
    with open("embeddings.pickle", "wb") as f:
        pickle.dump(embeddings, f)
    with open("embedding_cache.pickle", "wb") as f:
        pickle.dump(embedding_cache, f)
    print("Done!")


def load_embeddings():
    print("Loading embeddings... ", end="")
    if os.path.isfile("embeddings.pickle") and os.path.isfile("embedding_cache.pickle"):
        with open("embeddings.pickle", "rb") as f:
            embeddings.update(pickle.load(f))
        with open("embedding_cache.pickle", "rb") as f:
            embedding_cache.update(pickle.load(f))
        print("Done!")
    else:
        print("Pickle file doesn't exist; skipping load.")


@lru_cache
def get_embedder(model):
    if "/" not in model:
        return get_embedder(abbrev_to_model[model])
    
    if model == "LaBSE/1":
        return labse.encode

    # if model == "universal-sentence-encoder-cmlm/multilingual-base/1":
    #     return cmlm.encode

    # if model == "universal-sentence-encoder-cmlm/multilingual-base-br/1":
    #     return cmlm.encode_br

    return hub.load(f"https://tfhub.dev/google/{model}")


@lru_cache
def get_embedder_layer(model, trainable=False):

    if "/" not in model:
        return get_embedder(abbrev_to_model[model])

    return hub.KerasLayer(
        f"https://tfhub.dev/google/{model}",
        output_shape=[512],
        input_shape=[],
        dtype=tf.string,
        trainable=False,
        name=model_to_abbrev(model),
    )


def model_to_abbrev(model):
    if "/" not in model:
        return model
    name, number = model.rsplit("/", 1)
    initials = [word[0] for word in name.replace("/", "-").split("-")]
    abbrev = "".join(initials) + number
    abbrev_to_model[abbrev] = model
    return abbrev


def embed(text, model=DEFAULT_MODEL, chunksize=DEFAULT_CHUNKSIZE, maxlength=200):

    abbrev = model_to_abbrev(model)
    
    if isinstance(text, str):
        text = [text]

    to_embed = []
    for msg in text:
        if msg not in embedding_cache[abbrev]:
            to_embed.append(msg)

    if len(to_embed) > 0:
        embedder = get_embedder(model)

    for i in range(0, len(to_embed), chunksize):
        chunk = to_embed[i:i+chunksize]
        print(f"Completed {i} of {len(to_embed)} embeddings...")
        count = sum([len(e[:maxlength]) for e in chunk]) 
        print(f"Next chunk has {count} characters.")
        embedded = embedder([t[:maxlength] for t in chunk]).numpy()
        for msg, embedding in zip(chunk, embedded):
            embedding_cache[abbrev][msg] = embedding

    return np.array([embedding_cache[abbrev][msg] for msg in text])


def calculate_embeddings(data, fields=["title", "description"], model=DEFAULT_MODEL, chunksize=DEFAULT_CHUNKSIZE):

    for field in fields:
        strings = data.column(field)
        embeddings[model_embedding_key(model, field)] = DataFrame(embed(strings, model=model, chunksize=chunksize), index=data.indices())


def model_embedding_key(model, field):
    abbrev = model_to_abbrev(model)
    return field + "_" + abbrev
