from dataloading import *
from stats import *
from embedding_utils import *
from classes import *

# MODEL = "use4"
MODEL = "useml3"
# MODEL = "usel5"
# MODEL = "usem3"
# MODEL = "l1"

CHUNK_SIZE = 10000 if MODEL == "use4" else 1000

data = load_data()#.filter(lambda x: x.index < 5000)

calculate_embeddings(data, fields=["title"], model=MODEL)

topics = data.filter(lambda x: x.kind == "topic")
content = data.filter(lambda x: x.kind != "topic")

content.assign_groups(0.001, 0)
holdout = content.filter(lambda x: x.group == "holdout")


def bottom_up(node, source_key, reps, include_content=True, self_weight=0.1):
    kids = [n for n in node.children if not n.group]
    if not include_content:
        kids = [n for n in kids if n.kind=="topic"]
    rep = node.embedding(source_key).copy()
    if kids:
        rep *= self_weight
        children_weight = (1 - self_weight) / len(kids)
        for kid in kids:
            rep += bottom_up(kid, source_key, reps, include_content=include_content, self_weight=self_weight) * children_weight
    reps[node.index] = rep
    return rep


reps = {}
rootnodes = topics.filter(lambda x: x.parent is None)
for node in rootnodes.values():
    bottom_up(node, f"title_{MODEL}", reps)#, include_content=False)
embeddings[f"bu_title_{MODEL}"] = pd.DataFrame.from_dict(reps, orient="index")


# rankings = topics.rankings_by_content_id_to_parent(holdout, self_key=f"bu_title_{MODEL}", other_key=f"title_{MODEL}")
# print(rankings.mean())

def top_down(node, source_key, reps, self_weight=0.85):
    rep = node.embedding(source_key).copy()
    if node.parent:
        rep = self_weight * rep + (1 - self_weight) * reps[node.parent.index]
    reps[node.index] = rep
    kids = [n for n in node.children if not n.group and n.kind == "topic"]
    for kid in kids:
        top_down(kid, source_key, self_weight=self_weight, reps=reps)

reps = {}
rootnodes = topics.filter(lambda x: x.parent is None)
for node in rootnodes.values():
    top_down(node, f"bu_title_{MODEL}", reps)
embeddings[f"td_bu_title_{MODEL}"] = pd.DataFrame.from_dict(reps, orient="index")


rankings = topics.rankings_by_content_id_to_parent(holdout, self_key=f"td_bu_title_{MODEL}", other_key=f"title_{MODEL}")
print("Across entire holdout set:")
print("Means:")
print(rankings.mean())
print("Medians:")
print(rankings.median())

rankings = topics.rankings_by_content_id_to_parent(holdout.filter(lambda x: x.lang_id=="en"), self_key=f"td_bu_title_{MODEL}", other_key=f"title_{MODEL}")
print("Across English holdout set:")
print("Means:")
print(rankings.mean())
print("Medians:")
print(rankings.median())

rankings = topics.rankings_by_content_id_to_parent(holdout.filter(lambda x: x.lang_id!="en"), self_key=f"td_bu_title_{MODEL}", other_key=f"title_{MODEL}")
print("Across non-English holdout set:")
print("Means:")
print(rankings.mean())
print("Medians:")
print(rankings.median())


# rankings = topics.rankings_by_content_id_to_parent(holdout, f"title_{MODEL}")
# print(rankings.mean())
