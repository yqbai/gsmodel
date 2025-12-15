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
from torch.optim.lr_scheduler import ReduceLROnPlateau

model.eval()
with torch.no_grad():
    x = features_np
    outputs = model.bert(inputs_embeds=x)
    pooled_output = outputs.last_hidden_state
    embeddings = pooled_output

mat_link = pd.read_csv('streamline_info.csv', header=0)
mat_link['tag'] = mat_link.iloc[:, 0].astype(str) + '_' + mat_link.iloc[:, 1].astype(str)
mat_link_region = pd.read_csv('D99_labeltable_r.txt', header=None, sep=' ')
mat_link_region.drop(columns=[0], inplace=True)
t = pd.merge(mat_link, mat_link_region, left_on='streamline_end_region', right_on=1)

contingency_table = pd.crosstab(t['tag'], t.iloc[:,8])
common_indices = contingency_table.index.intersection(labels_samplename_ori)
contingency_table = contingency_table.reindex(common_indices, axis=0)

embeddings2 = embeddings[[labels_samplename_ori.index(label) for label in common_indices]]
a = [labels_samplename_ori[labels_samplename_ori.index(label)] for label in common_indices]

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
contingency_table2 = torch.tensor(contingency_table.values, dtype=torch.float32)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.optim.lr_scheduler import StepLR
from sklearn.decomposition import PCA
import random

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

features = torch.tensor(embeddings2.permute(0,2,1), dtype=torch.float32)
labels = torch.tensor(contingency_table2, dtype=torch.float32)

pca = PCA(n_components=1)
features_np = features.numpy().reshape(features.shape[0], -1)  # Flatten features for PCA
pc1_values = pca.fit_transform(features_np).flatten()
pc1_min = np.percentile(pc1_values, 10)
pc1_max = np.percentile(pc1_values, 90) 

valid_indices = np.where((pc1_values >= pc1_min) & (pc1_values <= pc1_max))[0]
filtered_features = features[valid_indices]
filtered_labels = labels[valid_indices]
dataset = TensorDataset(filtered_features, filtered_labels)

class CNNModel(nn.Module):
    def __init__(self, input_channels, seq_len, output_dim):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * (seq_len // 4), 256)
        self.fc2 = nn.Linear(256, output_dim)
        self.dropout = nn.Dropout(0.2)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

input_channels = features.shape[1]  # 264
seq_len = features.shape[2]         # 20
output_dim = labels.shape[1]        # 150

import torch.nn.functional as F

def cosine_similarity_loss(output, target):
    return 1 - F.cosine_similarity(output, target, dim=1).mean()

model = CNNModel(input_channels=input_channels, seq_len=seq_len, output_dim=output_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        #loss = criterion(outputs, targets)
        loss = cosine_similarity_loss(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    avg_train_loss = running_loss / len(train_loader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_train_loss:.4e}')
    scheduler.step(avg_train_loss)
