import pandas as pd
import numpy as np
import nltk
import gensim.downloader as api
import torch
import torch.nn as nn
from models import NeuralNetwork,RNNModel, BiLSTMModel
from torch.utils.data import DataLoader, Dataset, random_split
import csv
from transformers import BertTokenizer
from transformers import BertForSequenceClassification
from transformers import AdamW, get_linear_schedule_with_warmup
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer




def GlovePreprocess(file):
    df = pd.read_csv(file)
    df = df.dropna()
    df = df[df["label"].apply(lambda x: str(x).isdigit())]
    df["label"] = df["label"].astype(int)
    df.reset_index(drop=True, inplace=True)
    glove_embs = api.load("glove-wiki-gigaword-50")
    embedding_layer = nn.Embedding(len(glove_embs), 50)
    pretrained_weights = glove_embs.vectors
    embedding_layer.from_pretrained(torch.FloatTensor(pretrained_weights))

    maxlen = max(len(text) for text in df["text"])
    df["text"] = df["text"].apply(lambda x: [glove_embs.key_to_index.get(word, 2) for word in x] + [2] * (maxlen - len(x)))

    df["text"] = df["text"].apply(lambda x: embedding_layer(torch.LongTensor(x)).detach().numpy())
    return df, maxlen


def BOWPreprocess(file):
    df = pd.read_csv("output.csv")
    df = df.dropna()
    df = df[df["label"].apply(lambda x: str(x).isdigit())]
    df["label"] = df["label"].astype(int)
    df.reset_index(drop=True, inplace=True)

    df['text'] = df.apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
    sentences = df['text'].values

    vectorizer = CountVectorizer(max_features=1000)

    vectorizer.fit(sentences)

    vocab = vectorizer.vocabulary_
    vocab_dict = {word: idx+1 for word, idx in vocab.items()}

    embeddings = []
    for sentence in sentences:
        embedding = vectorizer.transform([sentence]).toarray().flatten()
        embeddings.append(embedding)

    df_emb = pd.DataFrame({'text': embeddings, 'label': df['label']})
    return df_emb, len(embeddings[0])

def BertProcess(file):
    df = pd.read_csv(file)
    df = df.dropna()
    df = df[df["label"].apply(lambda x: str(x).isdigit())]
    df["label"] = df["label"].astype(int)
    df.reset_index(drop=True, inplace=True)


    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    df["text"] = df["text"].apply(lambda x: tokenizer.encode(x, add_special_tokens=True))
    maxlen = max(len(text) for text in df["text"])
    df["text"] = df["text"].apply(lambda x: x + [0] * (maxlen - len(x)))
    return df, maxlen


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = torch.tensor(self.data.loc[idx, "text"], dtype=torch.float32)
        label = torch.tensor(self.data.loc[idx, "label"], dtype=torch.long)
        return text, label

class BertDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.loc[idx]
        tokenized_text = row['text']
        label = row['label']

        input_ids = tokenized_text[:self.max_length] + [0] * (self.max_length - len(tokenized_text))
        attention_mask = [1 if token_id != 0 else 0 for token_id in input_ids]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.long)
        }
    
def NNmain(method):
    if method == "bow":
        df, maxlen = BOWPreprocess("output.csv")
        dataset = MyDataset(df)

        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        # hidden = [128, 256, 512, 1024]
        # hidden2 = [64, 128, 256, 512]
        # resdic =  {}
        # for j in hidden:
        #     for k in hidden2:
        hidden1 = 128
        # hidden2 = 64
        input_size = maxlen
        output_size = len(df["label"].unique())
        model = NeuralNetwork(input_size, hidden1, output_size)
        epochs = 10
        learning_rate = 0.001
        model.train(train_loader, epochs, learning_rate)
        # with open('results.csv', 'w', newline='') as csvfile:
        #     csv_writer = csv.writer(csvfile)
        #     csv_writer.writerow(['Input', 'Predicted Label', 'Actual Label'])

        #     for i, (input_data, label) in enumerate(test_dataset):
        #         prediction = model.predict(input_data.unsqueeze(0))
        #         csv_writer.writerow([i + 1, prediction, label.item()])

        accuracy = model.accuracy(test_loader)
        # resdic[(j,k)] = accuracy
        # print("hidden size: ", j,k)
        print(f"Model accuracy: {accuracy:.2f}%")
    elif method == "glove":
        df, maxlen = GlovePreprocess("output.csv")
        dataset = MyDataset(df)

        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        # hidden = [128, 256, 512, 1024]
        # hidden2 = [64, 128, 256, 512]
        # resdic =  {}
        # for j in hidden:
        #     for k in hidden2:
        hidden1 = 128
        input_size = 50
        output_size = len(df["label"].unique())
        model = NeuralNetwork(input_size, hidden1, output_size)
        epochs = 10
        learning_rate = 0.001
        model.train(train_loader, epochs, learning_rate)
        # with open('results.csv', 'w', newline='') as csvfile:
        #     csv_writer = csv.writer(csvfile)
        #     csv_writer.writerow(['Input', 'Predicted Label', 'Actual Label'])

        #     for i, (input_data, label) in enumerate(test_dataset):
        #         prediction = model.predict(input_data.unsqueeze(0))
        #         csv_writer.writerow([i + 1, prediction, label.item()])

        accuracy = model.accuracy(test_loader)
        # resdic[(j,k)] = accuracy
        # print("hidden size: ", j,k)
        print(f"Model accuracy: {accuracy:.2f}%")
    elif method == "bert":
        df, maxlen = BertProcess("output.csv")
        dataset = MyDataset(df)

        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        # hidden = [128, 256, 512, 1024]
        # hidden2 = [64, 128, 256, 512]
        # resdic =  {}
        # for j in hidden:
        #     for k in hidden2:
        hidden1 = 128
        # hidden2 = 64
        input_size = maxlen
        output_size = len(df["label"].unique())
        model = NeuralNetwork(input_size, hidden1, output_size)
        epochs = 10
        learning_rate = 0.001
        model.train(train_loader, epochs, learning_rate)
        # with open('results.csv', 'w', newline='') as csvfile:
        #     csv_writer = csv.writer(csvfile)
        #     csv_writer.writerow(['Input', 'Predicted Label', 'Actual Label'])

        #     for i, (input_data, label) in enumerate(test_dataset):
        #         prediction = model.predict(input_data.unsqueeze(0))
        #         csv_writer.writerow([i + 1, prediction, label.item()])

        accuracy = model.accuracy(test_loader)
        # resdic[(j,k)] = accuracy
        # print("hidden size: ", j,k)
        print(f"Model accuracy: {accuracy:.2f}%")

def RNNmain(method):
    if method == "bow":
        df, maxlen = BOWPreprocess("output.csv")
        dataset = MyDataset(df)

        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        hidden = 512
        input_size = maxlen
        output_size = len(df["label"].unique())
        model = RNNModel(input_size, hidden, output_size)

        epochs = 10
        learning_rate = 0.001
        model.train(train_loader, epochs, learning_rate)
        # with open('testres.csv', 'w', newline='') as csvfile:
        #     csv_writer = csv.writer(csvfile)
        #     csv_writer.writerow(['Input', 'Predicted Label', 'Actual Label'])

        #     for i, (input_data, label) in enumerate(test_dataset):
        #         prediction = model.predict(input_data.unsqueeze(0))
        #         csv_writer.writerow([i + 1, prediction, label.item()])

        accuracy = model.accuracy(test_loader)
        print(f"Model accuracy: {accuracy:.2f}%")
    elif method == "glove":
        df, maxlen = GlovePreprocess("output.csv")
        dataset = MyDataset(df)

        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        hidden = 512
        input_size = 50
        output_size = len(df["label"].unique())
        model = RNNModel(input_size, hidden, output_size)

        epochs = 10
        learning_rate = 0.001
        model.train(train_loader, epochs, learning_rate)
        # with open('testres.csv', 'w', newline='') as csvfile:
        #     csv_writer = csv.writer(csvfile)
        #     csv_writer.writerow(['Input', 'Predicted Label', 'Actual Label'])

        #     for i, (input_data, label) in enumerate(test_dataset):
        #         prediction = model.predict(input_data.unsqueeze(0))
        #         csv_writer.writerow([i + 1, prediction, label.item()])

        accuracy = model.accuracy(test_loader)
        print(f"Model accuracy: {accuracy:.2f}%")
    elif method == "bert":
        df, maxlen = BertProcess("output.csv")
        dataset = MyDataset(df)

        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        hidden = 512
        input_size = maxlen
        output_size = len(df["label"].unique())
        model = RNNModel(input_size, hidden, output_size)

        epochs = 10
        learning_rate = 0.001
        model.train(train_loader, epochs, learning_rate)
        # with open('testres.csv', 'w', newline='') as csvfile:
        #     csv_writer = csv.writer(csvfile)
        #     csv_writer.writerow(['Input', 'Predicted Label', 'Actual Label'])

        #     for i, (input_data, label) in enumerate(test_dataset):
        #         prediction = model.predict(input_data.unsqueeze(0))
        #         csv_writer.writerow([i + 1, prediction, label.item()])

        accuracy = model.accuracy(test_loader)
        print(f"Model accuracy: {accuracy:.2f}%")


def LSTMmain(method):
    if method == "bow":
        df, maxlen = BOWPreprocess("output.csv")
        dataset = MyDataset(df)

        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        hidden = 128
        input_size = maxlen
        output_size = len(df["label"].unique())
        model = BiLSTMModel(input_size, hidden, output_size)

        epochs = 10
        learning_rate = 0.001
        model.train(train_loader, epochs, learning_rate)
        # with open('testres.csv', 'w', newline='') as csvfile:
        #     csv_writer = csv.writer(csvfile)
        #     csv_writer.writerow(['Input', 'Predicted Label', 'Actual Label'])

        #     for i, (input_data, label) in enumerate(test_dataset):
        #         prediction = model.predict(input_data.unsqueeze(0))
        #         csv_writer.writerow([i + 1, prediction, label.item()])

        accuracy = model.accuracy(test_loader)
        print(f"Model accuracy: {accuracy:.2f}%")
    elif method == "glove":
        df, maxlen = GlovePreprocess("output.csv")
        dataset = MyDataset(df)

        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        hidden = 128
        input_size = 50
        output_size = len(df["label"].unique())
        model = BiLSTMModel(input_size, hidden, output_size)

        epochs = 10
        learning_rate = 0.001
        model.train(train_loader, epochs, learning_rate)
        # with open('testres.csv', 'w', newline='') as csvfile:
        #     csv_writer = csv.writer(csvfile)
        #     csv_writer.writerow(['Input', 'Predicted Label', 'Actual Label'])

        #     for i, (input_data, label) in enumerate(test_dataset):
        #         prediction = model.predict(input_data.unsqueeze(0))
        #         csv_writer.writerow([i + 1, prediction, label.item()])

        accuracy = model.accuracy(test_loader)
        print(f"Model accuracy: {accuracy:.2f}%")
    elif method == "bert":
        df, maxlen = BertProcess("output.csv")
        dataset = MyDataset(df)

        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        hidden = 128
        input_size = maxlen
        output_size = len(df["label"].unique())
        model = BiLSTMModel(input_size, hidden, output_size)

        epochs = 10
        learning_rate = 0.001
        model.train(train_loader, epochs, learning_rate)
        # with open('testres.csv', 'w', newline='') as csvfile:
        #     csv_writer = csv.writer(csvfile)
        #     csv_writer.writerow(['Input', 'Predicted Label', 'Actual Label'])

        #     for i, (input_data, label) in enumerate(test_dataset):
        #         prediction = model.predict(input_data.unsqueeze(0))
        #         csv_writer.writerow([i + 1, prediction, label.item()])

        accuracy = model.accuracy(test_loader)
        print(f"Model accuracy: {accuracy:.2f}%")

def SVMmain():
    df, maxlen = BOWPreprocess("output.csv")
    X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)
    model = SVC(kernel='linear')
    model.fit(X_train.tolist(), y_train.tolist())
    y_pred = model.predict(X_test.tolist())
    correct_count = 0
    total_count = len(y_test)
    for i in range(total_count):
        if y_test.iloc[i] == y_pred[i] or y_test.iloc[i] == y_pred[i]+1 or y_test.iloc[i] == y_pred[i]-1:
            correct_count += 1
    accuracy = (correct_count) / total_count
    print("Accuracy:", accuracy)


def BerdModel():
    df, maxlen = BertProcess("output.csv")
    df['label'] = df['label'] - 1
    train_df = df.sample(frac=0.8, random_state=200).reset_index(drop=True)
    test_df = df.drop(train_df.index).reset_index(drop=True)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    train_dataset = BertDataset(train_df, tokenizer, maxlen)
    test_dataset = BertDataset(test_df, tokenizer, maxlen)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    num_labels = 11

    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)

    epochs = 10
    learning_rate = 2e-5
    warmup_steps = 0
    total_steps = len(train_loader) * epochs

    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    criterion = torch.nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.4f}")

        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)

                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                logits = outputs.logits
                test_loss += loss.item()

                predictions = torch.argmax(logits, dim=1)
                correct_mask = (predictions == labels) | (predictions == labels - 1) | (predictions == labels + 1)
                correct += correct_mask.sum().item()

        avg_test_loss = test_loss / len(test_loader)
        test_accuracy = 100 * correct / len(test_dataset)
        print(f"Epoch {epoch + 1}/{epochs}, Test Loss: {avg_test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

if __name__ == "__main__":
    # NNmain("bow")
    # NNmain("bert")
    # NNmain("glove")
    # RNNmain("bow")
    # RNNmain("glove")
    # RNNmain("bert")
    # LSTMmain("bow")
    # LSTMmain("glove")
    # LSTMmain("bert")
    # BerdModel()
    SVMmain()
    
