from tensorflow.keras.optimizers import Adam
import tensorflow.keras as keras
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer
import pyarrow as pa
from datasets import Dataset, DatasetDict
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class Model:
    def __init__(self, model_path, useGPU=False) -> None:
        if useGPU:
            # TODO: create training with CUDA
            self.model = TFAutoModelForSequenceClassification.from_pretrained(model_path).to("cuda")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path).to("cuda")
        else:
            self.model = TFAutoModelForSequenceClassification.from_pretrained(model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    @staticmethod
    def create_dataset(data):
        schema = pa.schema([
            ("text", pa.string()),
            ("label", pa.int64()),
            ("hashtags", pa.list_(pa.string())),
            ("emojis", pa.list_(pa.string())),
            ("polarity", pa.float64()),
            ("subjectivity", pa.float64()),
            ("sentiment", pa.string()),
        ])

        arrow_table = pa.Table.from_batches([], schema=schema)
        dataset = Dataset(arrow_table=arrow_table)

        return dataset.from_pandas(data)

    def tokenize_dataset(self, dataset):
        return self.tokenizer(dataset["text"])

    def prepare_dataset(self, dataset):
        dataset = dataset.map(self.tokenize_dataset)
        tf_dataset = self.model.prepare_tf_dataset(
            dataset, batch_size=16, shuffle=True, tokenizer=self.tokenizer
        )

        return tf_dataset

    def prepare_train_test_data(self, dataset):
        train_data, test_data = train_test_split(dataset, train_size=0.8)
        train_dataset = self.create_dataset(train_data)
        test_dataset = self.create_dataset(test_data)

        tf_train = self.prepare_dataset(train_dataset)
        tf_test = self.prepare_dataset(test_dataset)

        return tf_train, tf_test

    def compile(self):
        loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metrics = keras.metrics.SparseCategoricalAccuracy('accuracy')
        self.model.compile(optimizer=Adam(3e-5), loss=loss, metrics=[metrics])

    def fit(self, train_data, validation_data, epochs=3):
        return self.model.fit(train_data, epochs=epochs, validation_data=validation_data)

    def evaluate(self, dataset):
        return self.model.evaluate(dataset)

    @staticmethod
    def load_saved_model(model_path, useGPU=False):
        return Model(model_path=model_path, useGPU=useGPU)

    def classify_text(self, text):
        encoded_text = self.tokenizer.encode(text, truncation=True, padding=True, return_tensors='tf')
        prediction = self.model(encoded_text).logits.numpy()[0]
        probs = np.exp(prediction) / np.sum(np.exp(prediction))
        predicted_class = np.argmax(probs)

        print("Predicted class:", predicted_class)
        print("Probability distribution:", probs)

        return [predicted_class, probs]

    def save_model(self, save_path):
        self.tokenizer.save_pretrained(save_path)
        self.model.save_pretrained(save_path)

