import tensorflow as tf
import numpy as np

content.assign_groups({"test": 0.01})

y_train_nodes = content.filter(lambda x: x.group == "")
y_test_nodes = content.filter(lambda x: x.group == "test")

x_train_indices = [n.parent.index for n in y_train_nodes]
x_test_indices = [n.parent.index for n in y_train_nodes]

y_train_indices = [n.index for n in y_train_nodes]
y_test_indices = [n.index for n in y_train_nodes]

x_train = embeddings[f"title_{MODEL}"].loc[x_train_indices]
x_test = embeddings[f"title_{MODEL}"].loc[x_test_indices]
y_train = embeddings[f"title_{MODEL}"].loc[y_train_indices]
y_test = embeddings[f"title_{MODEL}"].loc[y_test_indices]

model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(512)
])

predictions = model(x_train[:1]).numpy()


loss_fn = tf.keras.losses.CosineSimilarity()

model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy', tf.keras.metrics.CosineSimilarity()])

model.fit(x_train, y_train, epochs=5)

model.evaluate(x_test, y_test, verbose=2)