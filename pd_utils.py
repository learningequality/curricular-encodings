import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize

from embedding_utils import embeddings


def get_descendants(df, index, content_only=False):
	row = df.loc[index]
	filt = f'(lft > {row.lft}) & (lft < {row.rght}) & (channel_int == "{row.channel_int}")'
	if content_only:
		filt += ' & (kind_int == 0)'
	return df.query(filt)


def get_ancestors(df, index):
	row = df.loc[index]
	filt = pd.eval('(df.lft < row.lft) & (df.rght > row.rght) & (df.channel_int == row.channel_int)')
	return df[filt]


def get_mean_descendant_embedding(df, index, key, content_only=False):
	return embeddings[key].loc[get_descendants(df, index, content_only=content_only).index].mean()


def get_mean_ancestor_embedding(df, index, key):
	return embeddings[key].loc[get_ancestors(df, index).index].mean()


def normalize_embedding(key):
	embeddings[key] = pd.DataFrame(normalize(embeddings[key], norm='l2'), index=embeddings[key].index)