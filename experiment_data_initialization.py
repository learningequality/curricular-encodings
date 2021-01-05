from dataloading import *
from embedding_utils import *
from classes import *

import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

plt.ion()

# MODEL = "use4"
# MODEL = "useml3"
# MODEL = "usel5"
MODEL = "usem3"
# MODEL = "l1"

CHUNK_SIZE = 1000 if "l" in MODEL else 10000

data = load_data()

load_embeddings()
calculate_embeddings(data, fields=["title", "description"], model=MODEL, chunksize=CHUNK_SIZE)
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
training = data.filter(lambda x: x.group == "")

dataframe = data.as_dataframe()
training_df = training.as_dataframe()
