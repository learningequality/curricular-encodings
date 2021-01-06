from experiment_utils import *
from pd_utils import *
from tqdm import tqdm
import tensorflow as tf
import time

class RawTitleComparison(Experiment):

    def run(self, df=training_df, indices=None):
        pass

    def get_output_keys(self):
        # should return a tuple with the embedding keys (for topics and content)
        return f"title_{MODEL}", f"title_{MODEL}"


class RawTitleDescriptionComparison(Experiment):

    def run(self, df=training_df, indices=None):
        pass

    def get_output_keys(self):
        # should return a tuple with the embedding keys (for topics and content)
        return f"title_description_{MODEL}", f"title_description_{MODEL}"


class BottomUpTopDownTitles(Experiment):

    def run(self, df=training_df, indices=None):
        # should return a tuple with the embedding keys (topic and content)
        source_key = f"title_{MODEL}"
        bu_key = f"bu_title_{MODEL}"
        td_bu_key = f"td_bu_title_{MODEL}"

        # do the bottom-up pass
        if bu_key not in embeddings:
            print("Cloning embedding for doing bottom-up pass...")
            embs = embeddings[bu_key] = embeddings[source_key].copy()
            print("Performing bottom-up pass... ", end="")
            for index in tqdm(df[df.kind=="topic"].index):
                embs.loc[index] += get_mean_descendant_embedding(df, index, source_key, content_only=True)
                embs.loc[index] /= 2
            print("Done!")

        # do the top-down pass
        if td_bu_key not in embeddings:
            print("Cloning embedding for doing top-down pass...")
            embs = embeddings[td_bu_key] = embeddings[bu_key].copy()
            print("Performing top-down pass... ", end="")
            for index in tqdm(df[df.kind=="topic"].index):
                embs.loc[index] += get_mean_ancestor_embedding(df, index, bu_key)
                embs.loc[index] /= 2
            print("Done!")

        return td_bu_key, source_key


class PredictContentTitlesFromParentTitles(TrainableExperiment):

    input_key = f"title_{MODEL}"
    target_key = f"parent_title_prediction_{MODEL}"

    def prepare_data(self, df, group, fraction=1):

        content_indices = df[(df.group == group) & (df.kind != "topic")].sample(frac=fraction).index
        parent_indices = df.loc[content_indices].parent_index

        inputs = self.prepare_inputs(df, parent_indices, target_indices=content_indices)
        outputs = self.prepare_outputs(df, content_indices)

        return inputs, outputs

    def prepare_inputs(self, df, indices, target_indices=None):
        return np.array(embeddings[self.input_key].loc[indices])

    def prepare_outputs(self, df, indices):
        return np.array(embeddings[self.input_key].loc[indices])

    def build_model(self):

        self.model = tf.keras.models.Sequential([
          tf.keras.layers.Dense(256, activation="relu"),
          tf.keras.layers.Dropout(0.2),
          tf.keras.layers.Dense(512)
        ])

        loss_fn = tf.keras.losses.CosineSimilarity()

        self.model.compile(
            optimizer="adam",
            loss=loss_fn,
            metrics=[tf.keras.metrics.CosineSimilarity()]
        )

    def get_output_keys(self):
        # should return a tuple with the embedding keys (for topics and content)
        return self.target_key, self.input_key


# class PredictContentTitlesFromParentAndSiblingTitles(PredictContentTitlesFromParentTitles):

#     input_key = f"title_{MODEL}"
#     target_key = f"parent_title_prediction_{MODEL}"

#     def prepare_inputs(self, df, indices, target_indices=None):
#         embedding = embeddings[self.input_key]
#         indices = list(indices)
#         if target_indices is None:
#             target_indices = [None] * len(indices)
#         parent_embeddings = embedding.loc[indices]
#         sibling_embeddings = pd.DataFrame(np.zeros(parent_embeddings.shape), dtype=np.float32, index=indices)
#         # print("Generating training data...")
#         # for parent_i, target in tqdm(enumerate(target_indices), total=len(indices)):
#         #     try:
#         #         sibling_indices = data.by_indices([indices[parent_i]])[0].children.filter(lambda x: x.index != target).indices()
#         #     except:
#         #         import IPython; IPython.embed()
#         #     sibling_embeddings.iloc[parent_i] = embedding.loc[sibling_indices].mean()

#         return {"parent_embedding": parent_embeddings}#, "sibling_embedding": sibling_embeddings}

#     def prepare_outputs(self, df, indices):
#         return np.array(embeddings[self.input_key].loc[indices])

#     def build_model(self):

#         parent_input = tf.keras.Input(shape=(512,), name="parent_embedding")
#         # sibling_input = tf.keras.Input(shape=(512,), name="sibling_embedding")

#         # x = tf.keras.layers.concatenate([parent_input, sibling_input])
#         x = tf.keras.layers.concatenate([parent_input, parent_input])
#         # x = parent_input

#         x = tf.keras.layers.Dense(256, activation="relu")(x)

#         x = tf.keras.layers.Dropout(0.2)(x)

#         output = tf.keras.layers.Dense(512)(x)

#         self.model = tf.keras.Model(
#             inputs=[parent_input],
#             # inputs=[parent_input, sibling_input],
#             outputs=[output],
#         )

#         loss_fn = tf.keras.losses.CosineSimilarity()

#         self.model.compile(
#             optimizer="adam",
#             loss=loss_fn,
#             metrics=[tf.keras.metrics.CosineSimilarity()]
#         )

#     def get_output_keys(self):
#         # should return a tuple with the embedding keys (for topics and content)
#         return self.target_key, self.input_key



class PredictContentTitlesFromParentTitleAndDescription(PredictContentTitlesFromParentTitles):

    title_key = f"title_{MODEL}"
    description_key = f"description_{MODEL}"
    target_key = f"parent_title_description_prediction_{MODEL}"

    def prepare_inputs(self, df, indices, target_indices=None):
        embedding = embeddings[self.input_key]
        parent_title = np.array(embeddings[self.title_key].loc[indices])
        parent_description = np.array(embeddings[self.description_key].loc[indices])
        return {"parent_title": parent_title, "parent_description": parent_description}

    def prepare_outputs(self, df, indices):
        return np.array(embeddings[self.input_key].loc[indices])

    def build_model(self):

        title_input = tf.keras.Input(shape=(512,), name="parent_title")
        description_input = tf.keras.Input(shape=(512,), name="parent_description")

        x = tf.keras.layers.concatenate([title_input, description_input])

        x = tf.keras.layers.Dense(256, activation="relu")(x)

        x = tf.keras.layers.Dropout(0.2)(x)

        output = tf.keras.layers.Dense(512)(x)

        self.model = tf.keras.Model(
            inputs=[title_input, description_input],
            outputs=[output],
        )

        loss_fn = tf.keras.losses.CosineSimilarity()

        self.model.compile(
            optimizer="adam",
            loss=loss_fn,
            metrics=[tf.keras.metrics.CosineSimilarity()]
        )

    def get_output_keys(self):
        # should return a tuple with the embedding keys (for topics and content)
        return self.target_key, self.input_key




class PredictContentTitlesFromAncestorTitles(PredictContentTitlesFromParentTitles):

    title_key = f"title_{MODEL}"
    description_key = f"description_{MODEL}"
    target_key = f"ancestor_titles_prediction_{MODEL}"

    def prepare_inputs(self, df, indices, target_indices=None):
        embedding = embeddings[self.input_key]
        inputs = []
        print("Extracting ancestor embeddings...")
        for index in tqdm(indices):
            ancestor_indices = data.by_index(index).get_ancestors(include_self=True).column("index")
            inputs.append(np.array(embedding.loc[ancestor_indices]))
        print("Done!")
        print("Converting to ragged tensor...")
        before = time.time()
        ragged = tf.ragged.constant(inputs)
        print("Done! Took {} seconds...".format(time.time() - before))
        return ragged

    def prepare_outputs(self, df, indices):
        return np.array(embeddings[self.input_key].loc[indices])

    def build_model(self):

        title_input = tf.keras.Input(shape=(None, 512), name="ancestor_titles", ragged=True)

        x = tf.keras.layers.LSTM(256, use_bias=False)(title_input)

        x = tf.keras.layers.Dense(256, activation="relu")(x)

        x = tf.keras.layers.Dropout(0.2)(x)

        output = tf.keras.layers.Dense(512)(x)

        self.model = tf.keras.Model(
            inputs=[title_input],
            outputs=[output],
        )

        loss_fn = tf.keras.losses.CosineSimilarity()

        self.model.compile(
            optimizer="adam",
            loss=loss_fn,
            metrics=[tf.keras.metrics.CosineSimilarity()]
        )

    def get_output_keys(self):
        # should return a tuple with the embedding keys (for topics and content)
        return self.target_key, self.input_key

    def run_for_example(self, titles, language=None):
        embs = embed(titles, model=MODEL)
        em = self.model(embs.reshape(1,embs.shape[0],embs.shape[1]))[0].numpy()
        return content.ranked_matches_for_embedding(em, self.get_output_keys()[1], language=language)


class PredictContentTitlesFromAncestorTitlesNotChannel(PredictContentTitlesFromAncestorTitles):

    target_key = f"ancestor_titles_not_channel_prediction_{MODEL}"

    def prepare_inputs(self, df, indices, target_indices=None):
        embedding = embeddings[self.input_key]
        inputs = []
        print("Extracting ancestor embeddings...")
        for index in tqdm(indices):
            ancestor_indices = data.by_index(index).get_ancestors(include_self=True).column("index")[1:]
            inputs.append(np.array(embedding.loc[ancestor_indices]))
        print("Done!")
        print("Converting to ragged tensor...")
        before = time.time()
        ragged = tf.ragged.constant(inputs)
        print("Done! Took {} seconds...".format(time.time() - before))
        return ragged



class PredictContentTitlesAndDescriptionsFromAncestorTitlesAndDescriptionsNoChannel(PredictContentTitlesFromParentTitles):

    title_key = f"title_{MODEL}"
    description_key = f"description_{MODEL}"
    target_key = f"ancestor_titles_and_description_prediction_not_channels_{MODEL}"

    def prepare_inputs(self, df, indices, target_indices=None):
        title_embedding = embeddings[self.title_key]
        description_embedding = embeddings[self.description_key]
        inputs = []
        print("Extracting ancestor embeddings...")
        for index in tqdm(indices):
            ancestor_indices = data.by_index(index).get_ancestors(include_self=True).column("index")
            arrays = np.concatenate(
                (
                    np.array(title_embedding.loc[ancestor_indices]),
                    np.array(description_embedding.loc[ancestor_indices])
                ),
                axis=1
            )
            inputs.append(arrays)
        print("Done!")
        print("Converting to ragged tensor...")
        before = time.time()
        ragged = tf.ragged.constant(inputs)
        print("Done! Took {} seconds...".format(time.time() - before))
        return ragged

    def prepare_outputs(self, df, indices):
        return (
            np.array(embeddings[self.title_key].loc[indices]),
            np.array(embeddings[self.description_key].loc[indices])
        )

    def build_model(self):

        inputs = tf.keras.Input(shape=(None, 1024), name="ancestor_text", ragged=True)

        x = tf.keras.layers.LSTM(256, use_bias=False)(inputs)

        x = tf.keras.layers.Dense(256, activation="relu")(x)

        x = tf.keras.layers.Dropout(0.2)(x)

        output_title = tf.keras.layers.Dense(512, name="output_title")(x)
        output_description = tf.keras.layers.Dense(512, name="output_description")(x)

        self.model = tf.keras.Model(
            inputs=[inputs],
            outputs=[output_title, output_description],
        )

        loss_fn_title = tf.keras.losses.CosineSimilarity()
        loss_fn_description = tf.keras.losses.CosineSimilarity()

        self.model.compile(
            optimizer="adam",
            loss={"output_title": loss_fn_title, "output_description": loss_fn_description},
            loss_weights={"output_title": 0.666666666, "output_description": 0.33333333},
            metrics={"output_title": tf.keras.metrics.CosineSimilarity(), "output_description": tf.keras.metrics.CosineSimilarity()},
        )

    def get_output_keys(self):
        # should return a tuple with the embedding keys (for topics and content)
        concat_comparison_rep = f"title_description_{MODEL}"
        if concat_comparison_rep not in embeddings:
            embedding = embeddings[concat_comparison_rep] = pd.concat((embeddings[f"title_{MODEL}"], embeddings[f"description_{MODEL}"]), axis=1)
            embedding.columns = range(1024)
            normalize_embedding(concat_comparison_rep)
        return self.target_key, concat_comparison_rep

    def run(self, df=dataframe, indices=None, chunk_size=10000):

        assert self.model

        if not indices:
            indices = df.index

        chunks = []

        for indices_chunk in tqdm(grouper(chunk_size, indices), total=len(indices) / chunk_size):
            inputs = self.prepare_inputs(df, indices_chunk)
            outputs_titles, outputs_descriptions = self.model(inputs)
            chunks.append(np.concatenate((outputs_titles.numpy(), outputs_descriptions.numpy()), axis=1))

        predictions = np.concatenate(chunks)

        output_key, _ = self.get_output_keys()

        embeddings[output_key] = pd.DataFrame(predictions, index=indices)



class ParentLearnedEncoding(PredictContentTitlesFromParentTitles):

    title_key = f"title_{MODEL}"
    description_key = f"description_{MODEL}"
    target_key = f"parent_learned_encoding_{MODEL}"
    shared_encoder_layers = None

    def build_shared_encoder_layers(self):
        if self.shared_encoder_layers is None:
            self.shared_encoder_layers = {
                "language_embedding": tf.keras.layers.Embedding(len(languages), 16, name="language_embedding"),
                "kind_embedding": tf.keras.layers.Embedding(len(content_kinds), 8, name="kind_embedding"),
                "hidden_layer": tf.keras.layers.Dense(256, activation="relu", name="hidden_layer"),
                "prenorm_output": tf.keras.layers.Dense(512, name="prenorm_output"),
            }

    def get_encoder_network(self, prefix=""):

        self.build_shared_encoder_layers()

        title_input = tf.keras.layers.Input(512, name=f"{prefix}title_input")
        description_input = tf.keras.layers.Input(512, name=f"{prefix}description_input")
        language_input = tf.keras.layers.Input(1, name=f"{prefix}language_input")
        kind_input = tf.keras.layers.Input(1, name=f"{prefix}kind_input")

        language_embedding = self.shared_encoder_layers["language_embedding"](language_input)
        kind_embedding = self.shared_encoder_layers["kind_embedding"](kind_input)

        flat_language_embedding = tf.keras.layers.Flatten(name=f"{prefix}language_flattener")(language_embedding)
        flat_kind_embedding = tf.keras.layers.Flatten(name=f"{prefix}kind_flattener")(kind_embedding)

        x = tf.keras.layers.Concatenate(axis=1, name=f"{prefix}input_concatenation")([title_input, description_input, flat_language_embedding, flat_kind_embedding])
        x = self.shared_encoder_layers["hidden_layer"](x)
        x = tf.keras.layers.Dropout(0.2, name=f"{prefix}dropout")(x)
        x = self.shared_encoder_layers["prenorm_output"](x)

        output = tf.keras.layers.Lambda(lambda val: tf.math.l2_normalize(val, axis=1), name=f"{prefix}encoding")(x)

        inputs = {
            f"{prefix}title_input": title_input,
            f"{prefix}description_input": description_input,
            f"{prefix}language_input": language_input,
            f"{prefix}kind_input": kind_input,
        }

        return inputs, output

    def get_comparer_model(self):

        inputs_parent, output_parent = self.get_encoder_network("parent_")
        inputs_child, output_child = self.get_encoder_network("child_")

        compare = tf.keras.layers.Dot(axes=1)([output_parent, output_child])

        all_inputs = {}
        all_inputs.update(inputs_parent)
        all_inputs.update(inputs_child)

        return tf.keras.Model(
            inputs=all_inputs,
            outputs=[compare],
        )

    def get_encoder_model(self):

        inputs, output = self.get_encoder_network()

        return tf.keras.Model(
            inputs=inputs,
            outputs=[output],
        )

    def create_input_dict(self, indices, df=dataframe, prefix=""):
        if isinstance(indices, tuple):
            indices = list(indices)
        return {
            f"{prefix}title_input": embeddings[self.title_key].loc[indices],
            f"{prefix}description_input": embeddings[self.description_key].loc[indices],
            f"{prefix}language_input": np.array(dataframe.language_int[indices]).reshape(len(indices), 1),
            f"{prefix}kind_input": np.array(dataframe.kind_int[indices]).reshape(len(indices), 1),
        }

    def prepare_data(self, df, group, fraction=1):

        content_indices = df[(df.group == group) & (df.kind != "topic")].sample(frac=fraction).index
        parent_indices = df.loc[content_indices].parent_index

        inputs = self.prepare_training_inputs(df, parent_indices, target_indices=content_indices)
        outputs = self.prepare_outputs(df, content_indices)

        return inputs, outputs

    def prepare_training_inputs(self, df, indices, target_indices=None):
        all_inputs = self.create_input_dict(indices, df=df, prefix="parent_")
        all_inputs.update(self.create_input_dict(target_indices, df=df, prefix="child_"))
        return all_inputs

    def prepare_inputs(self, df, indices, target_indices=None):
        return self.create_input_dict(indices, df=df)

    def prepare_outputs(self, df, indices):
        return np.ones((len(indices), 1))

    def build_model(self):
        self.encoder = self.get_encoder_model()
        self.model = self.get_comparer_model()

        self.model.compile(
            optimizer="adam",
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[lambda y_true, y_pred: tf.reduce_mean(y_pred, axis=-1)],
        )

    def get_output_keys(self):
        # should return a tuple with the embedding keys (for topics and content)
        return self.target_key, self.target_key

    def run(self, df=dataframe, indices=None, chunk_size=10000):

        assert self.encoder

        if not indices:
            indices = df.index

        chunks = []

        for indices_chunk in tqdm(grouper(chunk_size, indices), total=len(indices) / chunk_size):
            inputs = self.prepare_inputs(df, indices_chunk)
            output_chunk = self.encoder(inputs).numpy()
            chunks.append(output_chunk)

        predictions = np.concatenate(chunks)

        output_key, _ = self.get_output_keys()

        embeddings[output_key] = pd.DataFrame(predictions, index=indices)


#e = PredictContentTitlesFromParentTitles()
e = ParentLearnedEncoding()
# e.build_model()
#e.train()
