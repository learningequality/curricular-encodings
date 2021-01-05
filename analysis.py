import time

dots = []
for node in data.filter(lambda x: x.children and x.children[0].kind != "topic"):
	titles = [n.title for n in node.children]
	if len(titles) < 2:
		continue
	em = embed(titles, model=MODEL)
	d = np.dot(em, em.T)
	dots += [max(d[np.triu_indices(d.shape[0], k=True)])]

# average_dot_with_sibling_mean
dots = []
for node in data.filter(lambda x: x.children and x.children[0].kind != "topic"):
	titles = [n.title for n in node.children]
	if len(titles) < 2:
		continue
	em = embed(titles, model=MODEL)
	emm = em.mean(axis=0)
	d = np.dot(em, emm)
	dots += list(d)


# average_dot_with_sibling_mean_excluding_self
dots = []
for node in data.filter(lambda x: x.children and x.children[0].kind != "topic"):
	titles = [n.title for n in node.children]
	if len(titles) < 2:
		continue
	for title in titles:
		em = embed([t for t in titles if t != title], model=MODEL)
		emm = em.mean(axis=0)
		emn = embed([title], model=MODEL)
		d = np.dot(emm, emn.T)[0]
		dots += [d]


dots = []
for node in data.filter(lambda x: x.children and x.children[0].kind != "topic"):
	titles = [n.title for n in node.children]
	if len(titles) < 2:
		continue
	em = embed(titles, model=MODEL)
	emp = embed([node.title], model=MODEL)
	d = np.dot(em, emp.T)[0]
	dots += [d]


dots = []
for node in data.filter(lambda x: x.children and x.children[0].kind != "topic"):
	titles = [n.title for n in node.children]
	if len(titles) < 2:
		continue
	em = embed(titles, model=MODEL).mean(axis=0)
	emp = embed([node.title], model=MODEL)
	d = np.dot(em, emp.T)[0]
	dots += [d]

# random_content_pairing_dot
dots = []
contentnodes = data.filter(lambda x: x.kind != "topic")
for i in range(50000):
	node1, node2 = choice(contentnodes, 2, replace=False)
	em = embed([node1.title, node2.title], model=MODEL)
	d = np.dot(em[0], em[1])
	dots += [d]


sns.histplot([float(d) for d in dots])


def ancestors(node):
	if node is None:
		return []
	else:
		return [node.index] + ancestors(node.parent)

def traversal_distance(node1, node2):
	a1 = ancestors(node1)
	a2 = ancestors(node2)
	if a1[-1] != a2[-1]:
		return np.Inf
	while a1 and a2 and a1[-1] == a2[-1]:
		a1.pop()
		a2.pop()
	return len(a1) + len(a2)

# random_content_pairing_dots_to_distances
dots = []
dists = []
contentnodes = data.filter(lambda x: x.kind != "topic")
channels = list(set(contentnodes.column("channel_id")))
filtered_by_channel = {c: contentnodes.filter(lambda x: x.channel_id==c) for c in channels}
times = [0] * 4
for i in range(100000):
	if i % 100 == 0:
		print("REACHED", i)
		print(times)
	t = time.time()
	channel = choice(channels, 1)[0]
	node1, node2 = choice(filtered_by_channel[channel], 2, replace=False)
	times[0] += time.time() - t; t = time.time()
	dist = traversal_distance(node1.parent, node2.parent)
	times[1] += time.time() - t; t = time.time()
	dists.append(dist)
	e1 = embeddings[f"title_{MODEL}"].loc[node1.index]
	e2 = embeddings[f"title_{MODEL}"].loc[node2.index]
	times[2] += time.time() - t; t = time.time()
	d = np.dot(e1, e2)
	dots.append(d)
	times[3] += time.time() - t; t = time.time()

results = pd.DataFrame(zip(dists, dots), columns=("distance", "similarity")) 

# similarity_by_distance
sns.boxplot(results.distance, results.similarity) 

# similarity_by_distance_even_only
# only show even distances
sns.boxplot(results.distance[results.distance % 2 == 0], results.similarity[results.distance % 2 == 0]) 

# similarity_by_distance_round_up_odd_distances
# cluster odd with the next highest
d2s = results.distance.copy()
d2s[d2s % 2 == 1] += 1
sns.boxplot(d2s, results.similarity) 

