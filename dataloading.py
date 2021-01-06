import pandas as pd
import numpy as np
import json
import requests
from le_utils.constants import languages as le_utils_languages
from le_utils.constants import content_kinds as le_utils_content_kinds

DEFAULT_FILENAME = "metadatadump.json"

def load_data(filename=DEFAULT_FILENAME):

    print("Loading data... ", end="")

    from classes import Node, NodeList

    with open(filename) as f:
        data = {key: Node(fields) for key, fields in json.load(f).items()}

    for i, node in enumerate(data.values()):
        node.index = i
        # remove language subcodes
        if node["lang_id"]:
            node.fields["lang_id"] = node["lang_id"].split("-")[0]
        parent_id = node.fields["parent_id"]
        if parent_id:
            node.parent = data[parent_id]
            node.parent.children.append(node)

    for node in data.values():
        node.children = NodeList(node.children)

    print("Done!")

    return NodeList(data.values())


def load_languages():
    return list({l.primary_code: l for l in le_utils_languages.LANGUAGELIST}.keys())


def load_content_kinds():
    return list({k.id: k for k in le_utils_content_kinds.KINDLIST}.keys())

languages = load_languages()
language_lookup = pd.Series(range(len(languages)), index=languages, dtype=np.uint16)
language_lookup[None] = language_lookup["und"]

content_kinds = load_content_kinds()
kind_lookup = pd.Series(range(len(content_kinds)), index=content_kinds, dtype=np.uint8)
