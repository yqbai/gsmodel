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
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
import statistics

def get_label_onehot(labels_gs, encoder = None):
    labels_gs2 = []
    for label in labels_gs:
        if label in test_gs_list:
            labels_gs2.append(label)
        else:
            labels_gs2.append('else')
    # One-hot encode the labels
    labels_gs2 = np.array(labels_gs2).reshape(-1, 1)
    if encoder is None:
        encoder = OneHotEncoder(sparse=False)
        encoder.fit(labels_gs2)
        labels_gs2 = encoder.transform(labels_gs2)
        return torch.FloatTensor(labels_gs2), encoder
    else:
        labels_gs2 = encoder.transform(labels_gs2)
        return torch.FloatTensor(labels_gs2)

class sulcimodel(nn.Module):
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
            nn.Linear(32, 7) #labels_gs2.shape[1]
        )
    def forward(self, x):
        outputs = self.bert(inputs_embeds=x)
        pooled_output = outputs.last_hidden_state.view(x.size(0), -1)
        logits = self.classifier(pooled_output)
        return logits

labels_gs_r = []
for label in labels_gs:
    if label in test_gs_list:
        labels_gs_r.append(label)
    else:
        labels_gs_r.append('else')

new_labels_gs_r = []
for label in new_labels_gs:
    if label in test_gs_list:
        new_labels_gs_r.append(label)
    else:
        new_labels_gs_r.append('else')

filtered_indices = [i for i, label in enumerate(labels_gs) if label in test_gs_list]
new_filtered_indices = [i for i, label in enumerate(new_labels_gs) if label in test_gs_list]

max_f1 = 0
for iter_n in range(50):
    encoder = OneHotEncoder(sparse=False)
    labels_gs2 = torch.FloatTensor(encoder.fit_transform(np.array([labels_gs_r[i] for i in filtered_indices]).reshape(-1, 1)))
    new_labels_gs2 = torch.FloatTensor(encoder.transform(np.array([new_labels_gs_r[i] for i in new_filtered_indices]).reshape(-1, 1)))
    class_counts = np.bincount(np.argmax(labels_gs2, axis=1))
    class_weights = 1.0 / class_counts
    class_weights = torch.FloatTensor(class_weights).to(device)
    dataset = TensorDataset(features_np[filtered_indices].to(device), labels_gs2.to(device))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    dataset_test = TensorDataset(new_features_np[new_filtered_indices].to(device), new_labels_gs2.to(device))
    test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=True)
    print(iter_n)
    model = NumericBERT(feature_dim=264, seq_len=seqlen).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)
    model.train()
    num_epochs = 50
    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch in dataloader:
            features, labels = batch
            labels = torch.argmax(labels, dim=1)
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
            labels = torch.argmax(labels, dim=1)
            outputs = model(features)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    print(f'Accuracy: {accuracy}, F1 Score: {f1}')
    f1_per_class = f1_score(all_labels, all_preds, average=None)
    class_report = classification_report(all_labels, all_preds, target_names=encoder.categories_[0])
    print("F1 Score for each class:", f1_per_class)
    print("\nClassification Report:\n", class_report)
    if min(f1_per_class) > max_f1:
        best_model = model
        max_f1 = min(f1_per_class)
        best_res = [[accuracy, f1]]
        torch.save(model.cpu().state_dict(), 'bert_model.' + str(len(test_gs_list)) + '.pt')
        joblib.dump(encoder, 'encoder.' + str(len(test_gs_list)) + '.pkl')
        np.savetxt('metric_test.bert_bestres.' + str(len(test_gs_list)) + '.txt', np.array(best_res), fmt='%.4f', header='acc\tf1', comments='', delimiter = '\t')
        report_file = f'metric_test.bert_bestres_classreport.{len(test_gs_list)}.txt'
        with open(report_file, 'w') as f:
            f.write(class_report)
