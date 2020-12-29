import json
from classes import Node, NodeSet

DEFAULT_FILENAME = "metadatadump.json"

def load_data(filename=DEFAULT_FILENAME):
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

    return NodeSet(data)

if __name__ == "__main__":
    data = load_data()