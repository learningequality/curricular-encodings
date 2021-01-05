import json
import numpy as np
import pandas as pd
import itertools
from tqdm import tqdm

from numpy.random import choice

from embedding_utils import embeddings


def grouper(n, iterable):
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk


def node_grouper(n, iterable):
    for chunk in grouper(n, iterable):
        yield NodeList(chunk)


class Node(object):

    index = None
    parent = None
    children = None
    fields = None
    group = None

    def __init__(self, fields):
        self.fields = fields
        self.children = []

    def __getattr__(self, name):
        if name not in self.fields:
            raise AttributeError
        return self.fields[name]

    def __getitem__(self, name):
        return self.fields[name]

    def __dir__(self):
        return super().__dir__() + self.fields.keys()

    def __repr__(self):
        data = {attr: getattr(self, attr)() for attr in ["breadcrumbs", "studio_url"]}
        data.update(self.fields)
        return json.dumps(data, indent=2)

    def breadcrumbs(self, separator=" > "):
        return separator.join(self.get_ancestors().column("title"))

    def get_ancestors(self, include_self=False, _top=True):
        ancestors = (self.parent.get_ancestors(include_self=True, _top=False) if self.parent else []) + ([self] if include_self else [])
        if _top:
            ancestors = NodeList(ancestors)
        return ancestors

    def get_embedding(self, key):
        assert key in embeddings
        return embeddings[key].loc[self.index]

    def set_embedding(self, key, embedding):
        embeddings[key].loc[self.index] = embedding

    # def get_ranked_matches

    def studio_url(self):
        if self.kind == "topic":
            return f"https://studio.learningequality.org/channels/{self.channel_id}/edit/{self.id[:7]}"
        else:
            return f"https://studio.learningequality.org/channels/{self.channel_id}/edit/{self.parent.id[:7]}/{self.id[:7]}"


class NodeList(list):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._by_index = {n.index: n for n in self}
        self._by_id = {n.id: n for n in self}

    def by_indices(self, indices):
        return NodeList([self._by_index[index] for index in indices])

    def by_index(self, index):
        return self._by_index[index]

    def by_ids(self, ids):
        return NodeList([self._by_id[id] for id in ids])

    def by_content_ids(self, content_ids):
        content_ids = set(content_ids)
        return self.filter(lambda x: x.content_id in content_ids)

    def filter(self, fn):
        return NodeList([node for node in self if fn(node)])

    def exclude(self, fn):
        return NodeList([node for node in self if not fn(node)])

    def indices(self):
        return self.column("index")

    def column(self, key):
        return [getattr(n, key) for n in self]

    def embeddings(self, key):
        assert key in embeddings
        return embeddings[key].loc[self.indices()]

    def dot(self, other, self_key, other_key=None):
        if other_key is None:
            other_key = self_key
        return self.embeddings(self_key).dot(other.embeddings(other_key).T)

    def ranked_matches(self, other, self_key, other_key=None):
        scores = self.dot(other, self_key, other_key)
        return [other.by_indices(scores.loc[index].sort_values(ascending=False).index) for index in scores.index]

    def rankings_by_content_id_to_parent(self, other, group, self_key, other_key=None, chunk_size=1000):
        # self = all topics; other = test/holdout content nodes

        other_grouped = other.filter(lambda x: x.group == group)

        dataframe = other.as_dataframe()

        rankings = []

        # chunk to keep the scores calculation from generating an OOM error
        for chunk in tqdm(node_grouper(chunk_size, other_grouped), total=len(other_grouped) / chunk_size):

            chunk_indices = set(chunk.indices())

            parent_nodes = self.filter(lambda x: chunk_indices.intersection(x.children.indices()))

            scores = parent_nodes.dot(other_grouped, self_key, other_key)

            # for each of the content nodes being tested
            for node in chunk:

                ranked_indices = scores.loc[node.parent.index].sort_values(ascending=False).index
                content_ids = list(dataframe.loc[ranked_indices].content_id.unique())
                position = content_ids.index(node.content_id)

                percentile = position / len(content_ids)
                rankings.append([percentile, position, len(content_ids) - position - 1, node.channel_id, node.lang_id, node.kind])

        return pd.DataFrame(rankings, columns=["percentile", "before", "after", "channel", "language", "kind"], index=other_grouped.indices())

    def ranked_matches_for_embedding(self, embedding, key, count=10, deduped=True, kind=None, language=None):
        scores = embeddings[key].loc[self.indices()].dot(embedding.T)
        matches = self.by_indices(scores.sort_values(ascending=False).index)
        results = []
        content_ids = set()
        for match in matches:
            if kind and kind != match.kind:
                continue
            if language and language != match.lang_id:
                continue
            if deduped and match.content_id in content_ids:
                continue
            content_ids.add(match.content_id)
            results.append(match)
            if len(results) == count:
                break
        return results

    def assign_groups(self, groups, inclusion_condition=lambda x: x.kind != "topic"):
        
        for node in self.exclude(inclusion_condition):
            node.group = ""

        assert "" not in groups
        names, probs = zip(*list(groups.items()))
        assert sum(probs) < 1
        names = [""] + list(names)
        probs = [1 - sum(probs)] + list(probs)

        for node in self.filter(inclusion_condition):
            node.group = choice(names, 1, p=probs)[0]

    def initialize_empty_embedding(self, key, length=512):
        embeddings[key] = pd.DataFrame(np.zeros((len(self), length), dtype=np.float32), index=self.indices())

    def as_dataframe(self):
        channels = []
        kinds = []
        dicts = []
        for node in self:
            d = {
                "group": node.group or "",
            }
            d.update(node.fields)
            if d["channel_id"] not in channels:
                channels.append(d["channel_id"])
            if d["kind"] not in kinds:
                kinds.append(d["kind"])
            d["channel_int"] = channels.index(d["channel_id"])
            d["kind_int"] = kinds.index(d["kind"])
            d["parent_index"] = node.parent.index if node.parent else -1
            dicts.append(d)
        df = pd.DataFrame(dicts, index=self.indices())
       
        int32_cols = ["lft", "rght", "parent_index"]
        df[int32_cols] = df[int32_cols].astype(np.int32)
        int16_cols = ["level", "kind_int", "channel_int"]
        df[int16_cols] = df[int16_cols].astype(np.int16)

        return df

