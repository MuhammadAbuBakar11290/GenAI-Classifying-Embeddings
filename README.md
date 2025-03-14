# Day 2 - Classifying Embeddings with Keras and the Gemini API

This notebook is part of the Kaggle 5-day Generative AI course, designed to help you learn how to use embeddings produced by the Gemini API to train a model that can classify newsgroup posts into categories based on their content. This technique leverages the Gemini API's embeddings as input, allowing the model to perform well with relatively few examples compared to training a text model from scratch.

## Table of Contents
1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Preprocessing](#preprocessing)
4. [Creating Embeddings](#creating-embeddings)
5. [Building a Classification Model](#building-a-classification-model)
6. [Training the Model](#training-the-model)
7. [Evaluating Model Performance](#evaluating-model-performance)
8. [Custom Prediction](#custom-prediction)
9. [Conclusion](#conclusion)

## Introduction
In this notebook, you will learn how to use the Gemini API to generate embeddings for text data and then use these embeddings to train a classification model with Keras. The model will classify newsgroup posts into specific categories based on their content. This approach avoids the need to train on raw text input directly, making it efficient and effective with fewer examples.

## Dataset
The dataset used in this tutorial is the [20 Newsgroups Text Dataset](https://scikit-learn.org/0.19/datasets/twenty_newsgroups.html), which contains 18,000 newsgroup posts on 20 topics. The dataset is divided into training and test sets based on messages posted before and after a specific date. For this tutorial, we will use sampled subsets of the training and test sets, focusing on science-related categories.

### Dataset Preprocessing
The dataset is preprocessed to remove sensitive information like names and email addresses. Only the subject and body of each message are retained, and the text is truncated to 5,000 characters. The data is then transformed into a Pandas dataframe for easier manipulation.

```python
import email
import re
import pandas as pd

def preprocess_newsgroup_row(data):
    msg = email.message_from_string(data)
    text = f"{msg['Subject']}\n\n{msg.get_payload()}"
    text = re.sub(r"[\w\.-]+@[\w\.-]+", "", text)
    text = text[:5000]
    return text

def preprocess_newsgroup_data(newsgroup_dataset):
    df = pd.DataFrame({"Text": newsgroup_dataset.data, "Label": newsgroup_dataset.target})
    df["Text"] = df["Text"].apply(preprocess_newsgroup_row)
    df["Class Name"] = df["Label"].map(lambda l: newsgroup_dataset.target_names[l])
    return df
```

### Sampling the Data
To make the tutorial more manageable, we sample 100 data points from the training dataset and 25 from the test dataset, focusing on science-related categories.

```python
def sample_data(df, num_samples, classes_to_keep):
    df = df.groupby("Label")[df.columns].apply(lambda x: x.sample(num_samples)).reset_index(drop=True)
    df = df[df["Class Name"].str.contains(classes_to_keep)]
    df["Class Name"] = df["Class Name"].astype("category")
    df["Encoded Label"] = df["Class Name"].cat.codes
    return df

TRAIN_NUM_SAMPLES = 100
TEST_NUM_SAMPLES = 25
CLASSES_TO_KEEP = "sci"

df_train = sample_data(df_train, TRAIN_NUM_SAMPLES, CLASSES_TO_KEEP)
df_test = sample_data(df_test, TEST_NUM_SAMPLES, CLASSES_TO_KEEP)
```

## Creating Embeddings
Embeddings are generated for each piece of text using the Gemini API's `text-embedding-004` model. The embeddings are tailored for classification tasks by setting the `task_type` parameter to `"classification"`.

```python
from google.api_core import retry
from tqdm.rich import tqdm

tqdm.pandas()

@retry.Retry(timeout=300.0)
def embed_fn(text: str) -> list[float]:
    response = genai.embed_content(
        model="models/text-embedding-004", content=text, task_type="classification"
    )
    return response["embedding"]

def create_embeddings(df):
    df["Embeddings"] = df["Text"].progress_apply(embed_fn)
    return df

df_train = create_embeddings(df_train)
df_test = create_embeddings(df_test)
```

## Building a Classification Model
A simple classification model is built using Keras. The model accepts the raw embedding data as input, has one hidden layer, and an output layer specifying the class probabilities.

```python
import keras
from keras import layers

def build_classification_model(input_size: int, num_classes: int) -> keras.Model:
    return keras.Sequential(
        [
            layers.Input([input_size], name="embedding_inputs"),
            layers.Dense(input_size, activation="relu", name="hidden"),
            layers.Dense(num_classes, activation="softmax", name="output_probs"),
        ]
    )

embedding_size = len(df_train["Embeddings"].iloc[0])
classifier = build_classification_model(embedding_size, len(df_train["Class Name"].unique()))
classifier.summary()

classifier.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(),
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    metrics=["accuracy"],
)
```

## Training the Model
The model is trained using the training dataset. Early stopping is used to exit the training loop once the loss value stabilizes.

```python
import numpy as np

NUM_EPOCHS = 20
BATCH_SIZE = 32

y_train = df_train["Encoded Label"]
x_train = np.stack(df_train["Embeddings"])
y_val = df_test["Encoded Label"]
x_val = np.stack(df_test["Embeddings"])

early_stop = keras.callbacks.EarlyStopping(monitor="accuracy", patience=3)

history = classifier.fit(
    x=x_train,
    y=y_train,
    validation_data=(x_val, y_val),
    callbacks=[early_stop],
    batch_size=BATCH_SIZE,
    epochs=NUM_EPOCHS,
)
```

## Evaluating Model Performance
The model's performance is evaluated on the test dataset using Keras's `Model.evaluate` method.

```python
classifier.evaluate(x=x_val, y=y_val, return_dict=True)
```

## Custom Prediction
You can also make predictions with new, hand-written data to see how the model performs.

```python
new_text = """
First-timer looking to get out of here.

Hi, I'm writing about my interest in travelling to the outer limits!

What kind of craft can I buy? What is easiest to access from this 3rd rock?

Let me know how to do that please.
"""
embedded = embed_fn(new_text)

inp = np.array([embedded])
[result] = classifier.predict(inp)

for idx, category in enumerate(df_test["Class Name"].cat.categories):
    print(f"{category}: {result[idx] * 100:0.2f}%")
```

## Conclusion
This notebook demonstrates how to use the Gemini API to generate embeddings and train a classification model with Keras. By leveraging embeddings, the model can classify text data efficiently with relatively few examples. This approach can be applied to various text classification tasks, making it a powerful tool for natural language processing.

For more information on training models with Keras, including how to visualize the model training metrics, read [Training & evaluation with built-in methods](https://www.tensorflow.org/guide/keras/training_with_built_in_methods).
