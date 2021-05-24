from experiment_utils import *
from pd_utils import *
from tqdm import tqdm
import tensorflow as tf
import time


class PracticalUpperLimitTopicEmbedding(Experiment):

    """Calculate the topic embedding as the average of its child embeddings, without excluding test/validation sets."""

    source_key = f"title_description_{MODEL}"
    dest_key = "upper_limit_topic_embedding"

    def get_output_keys(self):
        # should return a tuple with the embedding keys (for topics and content)
        return self.dest_key, self.source_key

    def run(self, df=training_df, indices=None):
        leaf_topics = data.filter(lambda x: x.kind == "topic" and x.children and x.children.filter(lambda y: y.kind != "topic"))
        source_embedding = embeddings[self.source_key]
        leaf_topics.initialize_empty_embedding(self.dest_key, length=source_embedding.shape[1])
        dest_embedding = embeddings[self.dest_key]
        print(source_embedding.shape, dest_embedding.shape)
        for topic in leaf_topics:
            dest_embedding.loc[topic.index] = source_embedding.loc[topic.children.indices()].mean(axis=0)


class PracticalUpperLimitContentEmbedding(Experiment):

    """Calculate the topic embedding as the average of its child embeddings, without excluding test/validation sets."""

    source_key = f"title_description_{MODEL}"
    dest_key = "upper_limit_content_embedding"

    def get_output_keys(self):
        # should return a tuple with the embedding keys (for topics and content)
        return self.source_key, self.dest_key

    def run(self, df=training_df, indices=None):
        leaf_topics = data.filter(lambda x: x.kind == "topic" and x.children and x.children.filter(lambda y: y.kind != "topic"))
        source_embedding = embeddings[self.source_key]
        data.filter(lambda x: x.kind != "topic").initialize_empty_embedding(self.dest_key, length=source_embedding.shape[1])
        dest_embedding = embeddings[self.dest_key]
        for topic in leaf_topics:
            kids = topic.children.filter(lambda x: x.kind != "topic").indices()
            dest_embedding.loc[kids] = np.repeat(np.reshape(np.array(source_embedding.loc[topic.index]), (1, source_embedding.shape[1])), len(kids), axis=0)


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

        inputs = self.prepare_training_inputs(df, parent_indices, target_indices=content_indices)
        outputs = self.prepare_training_outputs(df, content_indices)

        return inputs, outputs

    def prepare_inputs(self, df, indices, target_indices=None):
        return np.array(embeddings[self.input_key].loc[indices])

    def prepare_training_outputs(self, df, indices):
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


class PredictContentTitlesFromParentTitleAndDescription(PredictContentTitlesFromParentTitles):

    title_key = f"title_{MODEL}"
    description_key = f"description_{MODEL}"
    target_key = f"parent_title_description_prediction_{MODEL}"

    def prepare_inputs(self, df, indices, target_indices=None):
        embedding = embeddings[self.input_key]
        parent_title = np.array(embeddings[self.title_key].loc[indices])
        parent_description = np.array(embeddings[self.description_key].loc[indices])
        return {"parent_title": parent_title, "parent_description": parent_description}

    def prepare_training_outputs(self, df, indices):
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

    def prepare_training_outputs(self, df, indices):
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

    def prepare_training_outputs(self, df, indices):
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

    encoded_key = f"parent_learned_encoding_raw_{MODEL}"
    down_predicted_key = f"parent_learned_encoding_down_predicted_{MODEL}"
    up_predicted_key = f"parent_learned_encoding_up_predicted_{MODEL}"

    shared_layers = None

    encoded_size = 512

    def build_shared_layers(self):
        if self.shared_layers is None:
            self.shared_layers = {
                "language_embedding": tf.keras.layers.Embedding(len(languages), 16, name="language_embedding"),
                "kind_embedding": tf.keras.layers.Embedding(len(content_kinds), 8, name="kind_embedding"),
                "hidden": tf.keras.layers.Dense(256, activation="relu", name="hidden"),
                "prenorm_output": tf.keras.layers.Dense(self.encoded_size, name="prenorm_output"),
                "parent_predictor_hidden": tf.keras.layers.Dense(256, activation="relu", name="parent_predictor_hidden"),
                "child_predictor_hidden": tf.keras.layers.Dense(256, activation="relu", name="child_predictor_hidden"),
                "encoded_parent_prenorm": tf.keras.layers.Dense(self.encoded_size, name="encoded_parent_prenorm"),
                "encoded_child_prenorm": tf.keras.layers.Dense(self.encoded_size, name="encoded_child_prenorm"),
                "decoder_hidden": tf.keras.layers.Dense(256, activation="relu", name="decoder_hidden"),
                "title_output_prenorm": tf.keras.layers.Dense(512, name="title_output_prenorm"),
                "description_output_prenorm": tf.keras.layers.Dense(512, name="description_output_prenorm"),
                "kind_output": tf.keras.layers.Dense(len(content_kinds), name="kind_output"),
                "language_output": tf.keras.layers.Dense(len(languages), name="language_output"),
            }

    def get_encoder_network(self, prefix=""):

        self.build_shared_layers()

        title_input = tf.keras.layers.Input(512, name=f"{prefix}title_input")
        description_input = tf.keras.layers.Input(512, name=f"{prefix}description_input")
        language_input = tf.keras.layers.Input(1, name=f"{prefix}language_input")
        kind_input = tf.keras.layers.Input(1, name=f"{prefix}kind_input")

        language_embedding = self.shared_layers["language_embedding"](language_input)
        kind_embedding = self.shared_layers["kind_embedding"](kind_input)

        flat_language_embedding = tf.keras.layers.Flatten(name=f"{prefix}language_flattener")(language_embedding)
        flat_kind_embedding = tf.keras.layers.Flatten(name=f"{prefix}kind_flattener")(kind_embedding)

        x = tf.keras.layers.Concatenate(axis=1, name=f"{prefix}input_concatenation")([title_input, description_input, flat_language_embedding, flat_kind_embedding])
        x = self.shared_layers["hidden"](x)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = self.shared_layers["prenorm_output"](x)

        output = tf.keras.layers.Lambda(lambda val: tf.math.l2_normalize(val, axis=1), name=f"{prefix}encoding")(x)

        inputs = {
            f"{prefix}title_input": title_input,
            f"{prefix}description_input": description_input,
            f"{prefix}language_input": language_input,
            f"{prefix}kind_input": kind_input,
        }

        return inputs, output

    def get_predictor_network(self, downwards=True, input_tensor=None):

        self.build_shared_layers()

        if downwards:
            source_name, target_name = "parent", "child"
        else:
            source_name, target_name = "child", "parent"

        encoded_input = tf.keras.layers.Input(self.encoded_size, name=f"encoded_{source_name}_input") if input_tensor is None else input_tensor

        x = self.shared_layers[f"{target_name}_predictor_hidden"](encoded_input)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = self.shared_layers[f"encoded_{target_name}_prenorm"](x)

        encoded_output = tf.keras.layers.Lambda(lambda val: tf.math.l2_normalize(val, axis=1), name=f"{target_name}predicted_encoding")(x)

        return encoded_input, encoded_output

    def get_decoder_network(self, prefix="", input_tensor=None):

        self.build_shared_layers()

        encoded_input = tf.keras.layers.Input(self.encoded_size, name=f"encoded_{prefix}input") if input_tensor is None else input_tensor

        x = self.shared_layers["decoder_hidden"](encoded_input)
        x = tf.keras.layers.Dropout(0.2)(x)

        title_output_prenorm = self.shared_layers["title_output_prenorm"](x)
        title_output = tf.keras.layers.Lambda(lambda val: tf.math.l2_normalize(val, axis=1), name=f"{prefix}title_output")(title_output_prenorm)

        description_output_prenorm = self.shared_layers["description_output_prenorm"](x)
        description_output = tf.keras.layers.Lambda(lambda val: tf.math.l2_normalize(val, axis=1), name=f"{prefix}description_output")(description_output_prenorm)

        kind_output = self.shared_layers["kind_output"](x)
        language_output = self.shared_layers["language_output"](x)

        outputs = {
            f"{prefix}title_output": title_output,
            f"{prefix}description_output": description_output,
            f"{prefix}language_output": language_output,
            f"{prefix}kind_output": kind_output,
        }

        return encoded_input, outputs

    def get_encoder_model(self, prefix=""):

        inputs, output = self.get_encoder_network(prefix=prefix)

        return tf.keras.Model(
            inputs=inputs,
            outputs=[output],
        )

    def get_autoencoder_model(self, prefix=""):
        encoder = self.get_encoder_model(prefix=prefix)
        decoder = self.get_decoder_model(prefix=prefix)
        inputs = encoder.input
        outputs = decoder(encoder(encoder.input))
        return tf.keras.Model(
            inputs=inputs,
            outputs=outputs,
        )

    def get_decoder_model(self, prefix=""):

        inputs, outputs = self.get_decoder_network(prefix=prefix)

        return tf.keras.Model(
            inputs=[inputs],
            outputs=outputs,
        )

    def get_predictor_model(self, downwards=True):

        inputs, output = self.get_predictor_network(downwards=downwards)

        return tf.keras.Model(
            inputs=inputs,
            outputs=[output],
        )

    def get_comparer_model(self):

        inputs_parent, encoded_parent = self.get_encoder_network("parent_")
        inputs_child, encoded_child = self.get_encoder_network("child_")

        _, predicted_child_encoding = self.get_predictor_network(downwards=True, input_tensor=encoded_parent)
        _, predicted_parent_encoding = self.get_predictor_network(downwards=False, input_tensor=encoded_child)

        _, decoded_parent = self.get_decoder_network("parent_", input_tensor=encoded_parent)
        _, decoded_child = self.get_decoder_network("child_", input_tensor=encoded_child)

        down_compare = tf.keras.layers.Dot(axes=1, name="down_compare")([encoded_child, predicted_child_encoding])
        up_compare = tf.keras.layers.Dot(axes=1, name="up_compare")([encoded_parent, predicted_parent_encoding])
        raw_compare = tf.keras.layers.Dot(axes=1, name="raw_compare")([encoded_child, encoded_parent])
        predicted_compare = tf.keras.layers.Dot(axes=1, name="predicted_compare")([predicted_child_encoding, predicted_parent_encoding])

        all_inputs = {}
        all_inputs.update(inputs_parent)
        all_inputs.update(inputs_child)

        all_outputs = {
            "down_compare": down_compare,
            "up_compare": up_compare,
            "raw_compare": raw_compare,
            "predicted_compare": predicted_compare,
        }
        all_outputs.update(decoded_parent)
        all_outputs.update(decoded_child)

        # import IPython; IPython.embed()

        all_outputs["encoded_parent"] = encoded_parent
        all_outputs["encoded_child"] = encoded_child

        return tf.keras.Model(
            inputs=all_inputs,
            outputs=all_outputs,
        )

    def get_comparer_model_old(self):

        inputs_parent, encoded_parent = self.get_encoder_network("parent_")
        inputs_child, encoded_child = self.get_encoder_network("child_")

        down_predictor = self.get_predictor_model(downwards=True)
        up_predictor = self.get_predictor_model(downwards=False)

        predicted_child_encoding = down_predictor(encoded_parent)
        predicted_parent_encoding = up_predictor(encoded_child)

        parent_decoder = self.get_decoder_model("parent_")
        child_decoder = self.get_decoder_model("child_")

        decoded_parent = parent_decoder(encoded_parent)
        decoded_child = child_decoder(encoded_child)

        down_compare = tf.keras.layers.Dot(axes=1, name="down_compare")([encoded_child, predicted_child_encoding])
        up_compare = tf.keras.layers.Dot(axes=1, name="up_compare")([encoded_parent, predicted_parent_encoding])
        raw_compare = tf.keras.layers.Dot(axes=1, name="raw_compare")([encoded_child, encoded_parent])
        predicted_compare = tf.keras.layers.Dot(axes=1, name="predicted_compare")([predicted_child_encoding, predicted_parent_encoding])

        all_inputs = {}
        all_inputs.update(inputs_parent)
        all_inputs.update(inputs_child)

        all_outputs = {
            "down_compare": down_compare,
            "up_compare": up_compare,
            "raw_compare": raw_compare,
            "predicted_compare": predicted_compare,
        }
        all_outputs.update(decoded_parent)
        all_outputs.update(decoded_child)

        # import IPython; IPython.embed()

        return tf.keras.Model(
            inputs=all_inputs,
            outputs=all_outputs,
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

    def create_output_dict(self, indices, df=dataframe, prefix=""):
        if isinstance(indices, tuple):
            indices = list(indices)
        return {
            f"{prefix}title_output": embeddings[self.title_key].loc[indices],
            f"{prefix}description_output": embeddings[self.description_key].loc[indices],
            f"{prefix}language_output": tf.keras.utils.to_categorical(dataframe.language_int[indices], num_classes=len(languages)),
            f"{prefix}kind_output": tf.keras.utils.to_categorical(dataframe.kind_int[indices], num_classes=len(content_kinds)),
        }

    def prepare_data(self, df, group, fraction=1):

        content_indices = df[(df.group == group) & (df.kind != "topic")].sample(frac=fraction).index
        parent_indices = df.loc[content_indices].parent_index

        inputs = self.prepare_training_inputs(df, parent_indices, target_indices=content_indices)
        outputs = self.prepare_training_outputs(df, content_indices, target_indices=content_indices)

        return inputs, outputs

    def prepare_training_inputs(self, df, indices, target_indices=None):
        all_inputs = self.create_input_dict(indices, df=df, prefix="parent_")
        all_inputs.update(self.create_input_dict(target_indices, df=df, prefix="child_"))
        return all_inputs

    def prepare_inputs(self, df, indices, target_indices=None):
        return self.create_input_dict(indices, df=df)

    def prepare_training_outputs(self, df, indices, target_indices):

        targets = {
            "down_compare": 1,
            "up_compare": 1,
            "raw_compare": 0,
            "predicted_compare": 0,
        }

        outputs = {key: np.ones((len(indices), 1)) * val for key, val in targets.items()}

        outputs["encoded_parent"] = np.zeros((len(indices), self.encoded_size))
        outputs["encoded_child"] = np.zeros((len(indices), self.encoded_size))

        outputs.update(self.create_output_dict(indices, df=df, prefix="parent_"))
        outputs.update(self.create_output_dict(target_indices, df=df, prefix="child_"))

        return outputs

    def build_model(self):
        self.encoder = self.get_encoder_model()
        self.decoder = self.get_decoder_model()
        self.down_predictor = self.get_predictor_model(downwards=True)
        self.up_predictor = self.get_predictor_model(downwards=False)

        # the actual trainable model
        self.model = self.get_comparer_model()

        self.model.compile(
            optimizer="adam",
            loss={
                "down_compare": tf.keras.losses.MeanSquaredError(name="down_compare_loss"),
                "up_compare": tf.keras.losses.MeanSquaredError(name="up_compare_loss"),
                "raw_compare": tf.keras.losses.MeanSquaredError(name="raw_compare_loss"),
                "predicted_compare": tf.keras.losses.MeanSquaredError(name="predicted_compare_loss"),
                "parent_language_output": tf.keras.losses.CategoricalCrossentropy(name="parent_language_output_loss"),
                "parent_kind_output": tf.keras.losses.CategoricalCrossentropy(name="parent_kind_output_loss"),
                "child_language_output": tf.keras.losses.CategoricalCrossentropy(name="child_language_output_loss"),
                "child_kind_output": tf.keras.losses.CategoricalCrossentropy(name="child_kind_output_loss"),
                "parent_title_output": tf.keras.losses.CosineSimilarity(name="parent_title_output_loss"),
                "parent_description_output": tf.keras.losses.CosineSimilarity(name="parent_description_output_loss"),
                "child_title_output": tf.keras.losses.CosineSimilarity(name="child_title_output_loss"),
                "child_description_output": tf.keras.losses.CosineSimilarity(name="child_description_output_loss"),
                "encoded_parent": std_loss,
                "encoded_child": std_loss,
            },
            loss_weights={
                "down_compare": 1,
                "up_compare": 0.1,
                "raw_compare": 0,
                "predicted_compare": 0,
                "parent_language_output": 1,
                "parent_kind_output": 0.5,
                "child_language_output": 1,
                "child_kind_output": 0.5,
                "parent_title_output": 1,
                "parent_description_output": 0.8,
                "child_title_output": 1,
                "child_description_output": 0.8,
                "encoded_parent": 1,
                "encoded_child": 1,
            },
            metrics={
                "down_compare": mean_metric("mean"),
                "up_compare": mean_metric("mean"),
                # "raw_compare": mean_metric("mean"),
                # "predicted_compare": mean_metric("mean"),
                # "parent_language_output": 0.2,
                # "parent_kind_output": 0.2,
                # "child_language_output": 0.2,
                # "child_kind_output": 0.2,
                # "parent_title_output": 0.2,
                # "parent_description_output": 0.2,
                # "child_title_output": 0.2,
                # "child_description_output": 0.2,
            },
        )

    def get_output_keys(self):
        # should return a tuple with the embedding keys (for topics and content)
        return self.down_predicted_key, self.encoded_key

    def run(self, df=dataframe, indices=None, chunk_size=10000):

        assert self.encoder
        assert self.decoder
        assert self.up_predictor
        assert self.down_predictor

        if not indices:
            indices = df.index

        encoded_chunks = []
        down_predicted_chunks = []
        up_predicted_chunks = []

        for indices_chunk in tqdm(grouper(chunk_size, indices), total=len(indices) / chunk_size):
            inputs = self.prepare_inputs(df, indices_chunk)

            encoded = self.encoder(inputs)
            encoded_chunks.append(encoded.numpy())

            down_predicted_chunks.append(self.down_predictor(encoded).numpy())
            up_predicted_chunks.append(self.up_predictor(encoded).numpy())

        embeddings[self.encoded_key] = pd.DataFrame(np.concatenate(encoded_chunks), index=indices)
        embeddings[self.down_predicted_key] = pd.DataFrame(np.concatenate(down_predicted_chunks), index=indices)
        embeddings[self.up_predicted_key] = pd.DataFrame(np.concatenate(up_predicted_chunks), index=indices)

    def save(self):
        assert self.model
        assert self.encoder
        assert self.decoder
        assert self.up_predictor
        assert self.down_predictor

        self.model.save(self.saved_filename() + "/model")
        self.encoder.save(self.saved_filename() + "/encoder")
        self.decoder.save(self.saved_filename() + "/decoder")
        self.up_predictor.save(self.saved_filename() + "/up_predictor")
        self.down_predictor.save(self.saved_filename() + "/down_predictor")

    def load(self):
        self.model = tf.keras.models.load_model(self.saved_filename() + "/model")
        self.encoder = tf.keras.models.load_model(self.saved_filename() + "/encoder")
        self.decoder = tf.keras.models.load_model(self.saved_filename() + "/decoder")
        self.up_predictor = tf.keras.models.load_model(self.saved_filename() + "/up_predictor")
        self.down_predictor = tf.keras.models.load_model(self.saved_filename() + "/down_predictor")


def std_loss(y_true, y_pred):

    stds = tf.math.reduce_std(y_pred, axis=0)
    loss = tf.reduce_mean(stds)

    return -loss


def mean_metric(name):

    def metric_callback(y_true, y_pred):
        return tf.reduce_mean(y_pred, axis=-1)

    metric_callback.__name__ = name

    return metric_callback


e = PredictContentTitlesAndDescriptionsFromAncestorTitlesAndDescriptionsNoChannel()
# e = ParentLearnedEncoding()
# e.build_model()
#e.train()
