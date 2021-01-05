from dataloading import *
from embedding_utils import *
from classes import *
from experiment_data_initialization import *


def measure_performance(topic_key, content_key=None, topics=topics, content=content, group="testing"):
    rankings = topics.rankings_by_content_id_to_parent(content, group, self_key=topic_key, other_key=content_key)
    return rankings.percentile.mean(), rankings.percentile.median()


def measure_performance_on_holdout_topics(topic_key, content_key=None, topics=topics, content=content, group="testing"):
    # measure performance, but only for predictions made from topics that were never used as inputs during training
    rankings = topics.rankings_by_content_id_to_parent(content, group, self_key=topic_key, other_key=content_key)
    holdout_indices = []
    for node in data.filter(lambda x: x.children and x.children[0].kind != "topic"):
        groups = set(node.children.column("group"))
        if "" not in groups:
            for child in node.children:
                if child.group == group:
                    holdout_indices.append(child.index)

    percentiles = rankings.loc[holdout_indices].percentile

    return percentiles.mean(), percentiles.median()


class Experiment(object):

    def run(self, df=training_df, indices=None):
        # should return a tuple with the embedding keys (topic and content)
        pass

    def name(self):
        return self.__class__.__name__

    def get_indices_for_group(self, df=dataframe, group="testing"):
        content_indices = df[(df.group == group) & (df.kind != "topic")].index
        parent_indices = df.loc[content_indices].parent_index
        return parent_indices, content_indices        

    def get_group_scores_with_parents(self, df=dataframe, group="testing", run=False):
        parent_indices, content_indices = self.get_indices_for_group(df=df, group=group)
        scores = self.get_sorted_pair_scores_for_group(df=df, group=group, run=run)
        return scores.loc[zip(parent_indices, content_indices)].sort_values(by="score")

    def get_scores_for_group(self, df=dataframe, group="testing", run=False):
        parent_indices, content_indices = self.get_indices_for_group(df=df, group=group)
        if run:
            self.run(df, indices=list(parent_indices) + list(content_indices))
        topic_key, content_key = self.get_output_keys()
        topic_embeddings = embeddings[topic_key].loc[parent_indices]
        content_embeddings = embeddings[content_key].loc[content_indices]
        return topic_embeddings.dot(content_embeddings.T)

    def get_sorted_pair_scores_for_group(self, df=dataframe, group="testing", run=False):
        scores = self.get_scores_for_group(df=df, group=group, run=run).stack().sort_values(ascending=False).to_frame("score")
        topic_indices, content_indices = zip(*scores.index)
        scores["topic"] = data.by_indices(topic_indices)
        scores["content"] = data.by_indices(content_indices)
        return scores

    def evaluate(self, df=dataframe, group="testing", run=False):
        if run:
            parent_indices, content_indices = self.get_indices_for_group(df=df, group=group)
            self.run(df, indices=list(parent_indices.unique()) + list(content_indices))
        topic_key, content_key = self.get_output_keys()
        return measure_performance(topic_key, content_key=content_key, group=group)

    def rankings(self, df=dataframe, group="testing", run=False):
        if run:
            self.run(df)
        topic_key, content_key = self.get_output_keys()
        return topics.rankings_by_content_id_to_parent(content, group, self_key=topic_key, other_key=content_key)

    def rankings_by_channel_on_holdouts(self, df=dataframe):
        rankings_t = self.rankings(df=df, group="testing")
        rankings_v = self.rankings(df=df, group="validation")
        rankings = rankings_t.append(rankings_v)
        groups = rankings.groupby("channel")
        by_channel = groups.mean().sort_values(by="channel").drop(["before", "after"], axis=1)
        by_channel = by_channel.rename({"percentile": "percentile_mean"}, axis=1)
        by_channel["percentile_median"] = groups.median().sort_values(by="channel").percentile
        by_channel["counts"] = groups.channel.count()
        by_channel["title"] = data.by_ids(by_channel.index).column("title")
        return by_channel

    def get_output_keys(self):
        # should return a tuple with the embedding keys (for topics and content)
        raise NotImplementedError()


class TrainableExperiment(Experiment):

    model = None

    def train(self, df=dataframe, epochs=1, fraction=1):
        
        if self.model is None:
            self.build_model()

        x_training, y_training = self.prepare_data(df, group="", fraction=fraction)
        x_validation, y_validation = self.prepare_data(df, group="validation")
        x_testing, y_testing = self.prepare_data(df, group="testing")

        self.model.fit(
            x_training,
            y_training,
            epochs=epochs,
            validation_data=(x_validation, y_validation),
            callbacks=[self.get_tensorboard_callback(), self.get_checkpoint_callback()],
        )

        return self.model.evaluate(x_testing, y_testing, verbose=2)

    def get_tensorboard_callback(self):
        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + self.name()
        return tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    def get_checkpoint_callback(self):
        return tf.keras.callbacks.ModelCheckpoint(f"saved_models/checkpoints/{self.name()}")

    def saved_filename(self):
        return f"saved_models/{self.name()}"

    def save(self):
        assert self.model
        self.model.save(self.saved_filename())

    def load(self):
        self.model = tf.keras.models.load_model(self.saved_filename())

    def run(self, df=dataframe, indices=None, chunk_size=10000):

        assert self.model

        if not indices:
            indices = df.index

        chunks = []

        for indices_chunk in tqdm(grouper(chunk_size, indices), total=len(indices) / chunk_size):
            inputs = self.prepare_inputs(df, indices_chunk)
            chunks.append(self.model(inputs).numpy())
        predictions = np.concatenate(chunks)
        
        output_key, _ = self.get_output_keys()
        
        embeddings[output_key] = pd.DataFrame(predictions, index=indices)



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



