import json
import numpy as np
import pandas as pd
import random

from embedding_utils import embeddings

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
        return json.dumps(self.fields, indent=2)

    def breadcrumbs(self, separator=" > "):
        prefix = (self.parent.breadcrumbs(separator=separator) + separator) if self.parent else ""
        return prefix + self.title

    def embedding(self, key):
        assert key in embeddings
        return embeddings[key].loc[self.index]        


class NodeSet(dict):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._by_index = {n.index: n for n in self.values()}
        self._by_id = {n.id: n for n in self.values()}

    def by_index(self, indices):
        return [self._by_index[index] for index in indices]

    def filter(self, fn):
        return NodeSet({key: val for key, val in self.items() if fn(val)})

    def indices(self):
        return self.column("index")

    def column(self, key):
        return [getattr(n, key) for n in self.values()]

    def embeddings(self, key):
        assert key in embeddings
        return embeddings[key].loc[self.indices()]

    def dot(self, other, self_key, other_key=None):
        if other_key is None:
            other_key = self_key
        return self.embeddings(self_key).dot(other.embeddings(other_key).T)

    def ranked_matches(self, other, self_key, other_key=None):
        scores = self.dot(other, self_key, other_key)
        return [other.by_index(scores.loc[index].sort_values(ascending=False).index) for index in scores.index]

    def rankings_by_content_id_to_parent(self, other, self_key, other_key=None):
        # self = all topics; other = test/holdout content nodes
        ranked_matches = self.ranked_matches(other, self_key, other_key)
        ranked_dict = {index: matches for index, matches in zip(self.indices(), ranked_matches)}

        befores = []
        afters = []
        percentiles = []
        rankings = []

        # for each of the content nodes being tested
        for node in other.values():
            before = set()
            after = set()
            found = False
            matches = ranked_dict[node.parent.index]
            for match in matches:
                # skip if parent is different language
                # if match.lang_id != node.lang_id:
                #     continue
                if not found:
                    if match.content_id == node.content_id:
                        found = True
                    else:
                        before.add(match.content_id)
                else:                
                    if match.content_id != node.content_id and match.content_id not in before:
                        after.add(match.content_id)

            percentile = (len(before) + 1) / (len(before) + len(after) + 1)
            rankings.append([percentile, len(before), len(after)])

        return pd.DataFrame(rankings, columns=["percentile", "before", "after"], index=other.indices())

    def rankings_of_group(data, group, field="em_title"):
        
        befores = []
        afters = []
        percents = []
        
        grouped = [n for n in data.values() if n["group"] == group]
        
        for node in grouped:
            before = set()
            after = set()
            found = False
            matches = ranked_matches(node["parent"], grouped)
            for match in matches:
                if not found:
                    if match["content_id"] == node["content_id"]:
                        found = True
                    else:
                        before.add(match["content_id"])
                else:                
                    if match["content_id"] != node["content_id"]:
                        after.add(match["content_id"])

            befores.append(len(before))
            afters.append(len(after))
            percents.append((len(before) + 1) / (len(before) + len(after) + 1))

        return np.mean(befores), np.mean(afters), np.mean(percents)

    def assign_groups(self, holdout_fraction, test_fraction):
        # holdout and test sets
        # only assigning to content nodes, not topics
        
        for node in self.values():
            node.group = ""
            if node.kind != "topic":
                rand = random.random()
                if rand <= holdout_fraction:
                    node.group = "holdout"
                elif rand <= (holdout_fraction + test_fraction):
                    node.group = "test"