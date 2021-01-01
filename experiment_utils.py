from dataloading import *
from embedding_utils import *
from classes import *

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.ion()

# MODEL = "use4"
# MODEL = "useml3"
# MODEL = "usel5"
MODEL = "usem3"
# MODEL = "l1"

CHUNK_SIZE = 10000 if MODEL == "use4" else 1000

data = load_data()#.filter(lambda x: x.index < 5000)


load_embeddings()
calculate_embeddings(data, fields=["title", "description"], model=MODEL)
save_embeddings()

topics = data.filter(lambda x: x.kind == "topic")
content = data.filter(lambda x: x.kind != "topic")

try:
    with open("group_assignments.json") as f:
        print("Loading group assignments... ", end="")
        assignments = json.load(f)
    for node in data:
        node.group = assignments[node.id]
except FileNotFoundError:
    print("Generating and saving group assignments... ", end="")
    content.assign_groups({"validation": 0.01, "testing": 0.01})
    assignments = {node.id: node.group for node in data}
    with open("group_assignments.json", "w") as f:
        json.dump(assignments, f)
print("Done!")

validation = content.filter(lambda x: x.group == "validation")
testing = content.filter(lambda x: x.group == "testing")
excluded_content_ids = set(content.filter(lambda x: x.group != "").column("content_id"))
excluded = content.filter(lambda x: x.group in ["", "excluded"] and x.content_id in excluded_content_ids)
for node in excluded:
    node.group = "excluded"


def measure_performance(topic_key, content_key=None, topics=topics, content=content, group="testing"):
    rankings = topics.rankings_by_content_id_to_parent(content, group, self_key=topic_key, other_key=content_key)
    return rankings.percentile.mean(), rankings.percentile.median()



# def bottom_up(node, source_key, reps, include_content=True, include_duplicate_siblings=True, self_weight=0.1):
#     excluded_titles = [] if include_duplicate_siblings else [n.title for n in node.children if n.group]
#     kids = [n for n in node.children if not n.group and n.title not in excluded_titles]
#     if not include_content:
#         kids = [n for n in kids if n.kind=="topic"]
#     rep = node.get_embedding(source_key).copy()
#     if kids:
#         rep *= self_weight
#         children_weight = (1 - self_weight) / len(kids)
#         for kid in kids:
#             rep += bottom_up(kid, source_key, reps, include_content=include_content, self_weight=self_weight) * children_weight
#     reps[node.index] = rep
#     return rep


# def bottom_up(node, source_key, reps, include_content=True, include_duplicate_siblings=True, self_weight=0.1):
#     excluded_titles = [] if include_duplicate_siblings else [n.title for n in node.children if n.group]
#     kids = [n for n in node.children if not n.group and n.title not in excluded_titles]
#     if not include_content:
#         kids = [n for n in kids if n.kind=="topic"]
#     rep = node.get_embedding(source_key).copy()
#     if kids:
#         rep *= self_weight
#         children_weight = (1 - self_weight) / len(kids)
#         for kid in kids:
#             rep += bottom_up(kid, source_key, reps, include_content=include_content, self_weight=self_weight) * children_weight
#     reps[node.index] = rep
#     return rep


# reps = {}
# rootnodes = topics.filter(lambda x: x.parent is None)
# for node in rootnodes:
#     bottom_up(node, f"title_{MODEL}", reps)#, include_content=False)
# embeddings[f"bu_title_{MODEL}"] = pd.DataFrame.from_dict(reps, orient="index")


# # rankings = topics.rankings_by_content_id_to_parent(validation, self_key=f"bu_title_{MODEL}", other_key=f"title_{MODEL}")
# # print(rankings.mean())

# def top_down(node, source_key, reps, self_weight=0.85):
#     rep = node.get_embedding(source_key).copy()
#     if node.parent:
#         rep = self_weight * rep + (1 - self_weight) * reps[node.parent.index]
#     reps[node.index] = rep
#     kids = [n for n in node.children if not n.group and n.kind == "topic"]
#     for kid in kids:
#         top_down(kid, source_key, self_weight=self_weight, reps=reps)

# reps = {}
# rootnodes = topics.filter(lambda x: x.parent is None)
# for node in rootnodes:
#     top_down(node, f"bu_title_{MODEL}", reps)
# embeddings[f"td_bu_title_{MODEL}"] = pd.DataFrame.from_dict(reps, orient="index")


# rankings = topics.rankings_by_content_id_to_parent(validation, self_key=f"td_bu_title_{MODEL}", other_key=f"title_{MODEL}")
# print("Across entire validation set:")
# print("Means:")
# print(rankings.mean())
# print("Medians:")
# print(rankings.median())

# rankings = topics.rankings_by_content_id_to_parent(validation.filter(lambda x: x.lang_id=="en"), self_key=f"td_bu_title_{MODEL}", other_key=f"title_{MODEL}")
# print("Across English validation set:")
# print("Means:")
# print(rankings.mean())
# print("Medians:")
# print(rankings.median())

# rankings = topics.rankings_by_content_id_to_parent(validation.filter(lambda x: x.lang_id!="en"), self_key=f"td_bu_title_{MODEL}", other_key=f"title_{MODEL}")
# print("Across non-English validation set:")
# print("Means:")
# print(rankings.mean())
# print("Medians:")
# print(rankings.median())


# rankings = topics.rankings_by_content_id_to_parent(validation, f"title_{MODEL}")
# print(rankings.mean())



