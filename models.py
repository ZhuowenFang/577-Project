import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size1, output_size):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size1)
        self.layer2 = nn.ReLU()
        # self.layer3 = nn.Linear(hidden_size1, hidden_size2)
        # self.layer4 = nn.ReLU()
        self.layer5 = nn.Linear(hidden_size1, output_size)



    def flatten(self, x):
        return x.view(x.size(0), -1)

    def forward(self, x):
        x = self.flatten(x)
        x = self.layer1(x)
        x = self.layer2(x)
        # x = self.layer3(x)
        # x = self.layer4(x)
        x = self.layer5(x)
        x = torch.softmax(x, dim=1)
        return x


    def train(self, train_loader, epochs, learning_rate):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        for epoch in range(epochs):
            running_loss = 0.0
            for i, (inputs, labels) in enumerate(train_loader, 0):
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            print(f"Epoch {epoch+1}, Loss: {running_loss / (i+1)}")

    def predict(self, x):
        with torch.no_grad():
            output = self(x)
            _, predicted = torch.max(output, 1)
            return predicted.item()

    def accuracy(self, test_loader):
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = self(inputs)
                _, predicted = torch.max(outputs, 1)
                # print(predicted)
                total += labels.size(0)
                correct += torch.sum(torch.abs(predicted - labels) <= 1).item()
        return 100 * correct / total
    

class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size1, output_size):
        super(RNNModel, self).__init__()
        self.input_size = input_size
        self.hidden_size1 = hidden_size1
        self.output_size = output_size
        self.rnn = nn.RNN(input_size, hidden_size1, batch_first=True)
        self.fc1 = nn.Linear(hidden_size1, output_size)

        
    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(1, batch_size, self.hidden_size1).to(x.device)
        x = x.view(batch_size, -1, self.input_size)
        out, _ = self.rnn(x, h0)
        out = self.fc1(out[:, -1, :])
        # out = F.relu(out)
        out = torch.softmax(out, dim=1)
        return out
    
    
    def train(self, train_loader, n_epochs, learning_rate):
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        for epoch in range(n_epochs):
            running_loss = 0.0
            for i, data in enumerate(train_loader, 0):
                inputs, labels = data
                inputs = inputs.unsqueeze(2)
                optimizer.zero_grad()
                outputs = self(inputs)
                labels_onehot = F.one_hot(labels, num_classes=self.output_size).float()
                loss = criterion(outputs, labels_onehot)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print('[Epoch %d] loss: %.3f' % (epoch + 1, running_loss / len(train_loader)))


    def accuracy(self, test_loader):
        correct = 0
        total = 0
        with torch.no_grad():
            for data in test_loader:
                inputs, labels = data
                inputs = inputs.unsqueeze(2)
                outputs = self(inputs)
                _, predicted = torch.max(outputs, 1)
                predicted = predicted.cpu().numpy()
                labels = labels.cpu().numpy()
                correct += (abs(predicted - labels) <= 1).sum().item()
                total += labels.shape[0]
        return correct / total

   

class BiLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size1, output_size):
        super(BiLSTMModel, self).__init__()
        self.input_size = input_size
        self.hidden_size1 = hidden_size1
        # self.hidden_size2 = hidden_size2
        self.output_size = output_size
        self.lstm = nn.LSTM(input_size, hidden_size1, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(hidden_size1*2, output_size)
        # self.fc2 = nn.Linear(hidden_size2, output_size)
        
    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(2, batch_size, self.hidden_size1).to(x.device)
        c0 = torch.zeros(2, batch_size, self.hidden_size1).to(x.device)
        x = x.view(batch_size, -1, self.input_size)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc1(out[:, -1, :])
        # out = F.relu(out)
        # out = self.fc2(out)
        out = torch.softmax(out, dim=1)
        return out

    
    def train(self, train_loader, n_epochs, learning_rate):
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        for epoch in range(n_epochs):
            running_loss = 0.0
            for i, data in enumerate(train_loader, 0):
                inputs, labels = data
                inputs = inputs.unsqueeze(2)
                optimizer.zero_grad()
                outputs = self(inputs)
                labels_onehot = F.one_hot(labels, num_classes=self.output_size).float()
                loss = criterion(outputs, labels_onehot)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print('[Epoch %d] loss: %.3f' % (epoch + 1, running_loss / len(train_loader)))


    def accuracy(self, test_loader):
        correct = 0
        total = 0
        with torch.no_grad():
            for data in test_loader:
                inputs, labels = data
                inputs = inputs.unsqueeze(2)
                outputs = self(inputs)
                _, predicted = torch.max(outputs, 1)
                predicted = predicted.cpu().numpy()
                labels = labels.cpu().numpy()
                correct += (abs(predicted - labels) <= 1).sum().item()
                total += labels.shape[0]
        return correct / total
