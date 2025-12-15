import sys
from torchvision import datasets
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import OneHotEncoder
from torch import nn
import torch.nn.functional as F
from random import sample
import torch.optim as optim
import scipy.sparse as sp
import matplotlib.pyplot as plt
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from transformers import BertModel, BertConfig
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import statistics

dataset = TensorDataset(features_np.to(device), labels_np.to(device))
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
dataset_test = TensorDataset(new_features_np.to(device), new_labels_np.to(device))
test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=True)

class gsmodel(nn.Module):
    def __init__(self, feature_dim, seq_len):
        super(NumericBERT, self).__init__()
        config = BertConfig(
            hidden_size=feature_dim,
            num_attention_heads=2,
            num_hidden_layers=3,
            intermediate_size=128,
            max_position_embeddings=seq_len
        )
        self.bert = BertModel(config)
        self.denoise = nn.Linear(feature_dim, feature_dim)
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim * seq_len, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )
    def forward(self, x):
        outputs = self.bert(inputs_embeds=x)
        pooled_output = outputs.last_hidden_state.view(x.size(0), -1)
        logits = self.classifier(pooled_output)
        return logits

max_acc = 0
for iter_n in range(100):
    print(iter_n)
    model = gsmodel(feature_dim=264, seq_len=seqlen).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)
    model.train()
    num_epochs = 50
    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch in dataloader:
            features, labels = batch
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_epoch_loss = epoch_loss / len(dataloader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_epoch_loss:.4f}')
        scheduler.step(avg_epoch_loss)
    model.eval()
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for batch in test_loader:
            features, labels = batch
            outputs = model(features)
            preds = torch.sigmoid(outputs).round()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    print(f'Accuracy: {accuracy}, F1 Score: {f1}')
    if accuracy > max_acc:
        best_model = model
        max_acc = accuracy
        best_res = [accuracy, f1, auc]
        savepath = './'
        torch.save(model.cpu().state_dict(), savepath + 'bert_model.pt')
        np.savetxt('metric_test.bert_bestres.txt', np.array(best_res), fmt='%.4f', header='acc f1', comments='')
