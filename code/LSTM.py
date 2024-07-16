# ORIGINAL LSTM + ATTENTION

import os
import random
import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchmetrics
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torchmetrics.classification import MulticlassF1Score, MulticlassAccuracy

# Configuration
class Config:
    SR = 32000
    N_MFCC = 13
    ROOT_FOLDER = './'
    N_CLASSES = 2
    BATCH_SIZE = 72
    N_EPOCHS = 20  # Increase epochs
    LR = 3e-4
    SEED = 42
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    output_dir = './output'
    file_name = 'best_model'
    lr_reduce_factor = 0.5
    lr_patience = 5
    train_json_path = './train.json'
    test_json_path = './test.json'
    pin_memory = True
    n_workers = 7
    classifier_output = 2
    feature_size = 13
    hidden_size = 256  # Increase hidden size
    num_layers = 3  # Increase number of layers
    dropout = 0.3
    bidirectional = True  # Use bidirectional LSTM
    epochs = 20  # Increase epochs
    batch_size = 72

CONFIG = Config()

# Set random seed for reproducibility
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(CONFIG.SEED)

# Load Data
df = pd.read_csv('../../open/train.csv')
train_df, val_df = train_test_split(df, test_size=0.2, random_state=CONFIG.SEED)

# Data Preprocessing: MFCC
def get_mfcc_feature(df, train_mode=True):
    features = []
    labels = []
    for _, row in tqdm(df.iterrows()):
        y, sr = librosa.load(row['path'], sr=CONFIG.SR)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=CONFIG.N_MFCC)
        mfcc = np.mean(mfcc.T, axis=0)
        features.append(mfcc)
        if train_mode:
            label = row['label']
            label_vector = np.zeros(CONFIG.N_CLASSES, dtype=float)
            label_vector[0 if label == 'fake' else 1] = 1
            labels.append(label_vector)
    if train_mode:
        return features, labels
    return features

train_mfcc, train_labels = get_mfcc_feature(train_df, True)
val_mfcc, val_labels = get_mfcc_feature(val_df, True)

# Dataset
class CustomDataset(Dataset):
    def __init__(self, mfcc, label):
        self.mfcc = mfcc
        self.label = label

    def __len__(self):
        return len(self.mfcc)

    def __getitem__(self, index):
        mfcc = self.mfcc[index]
        if self.label is not None:
            label = self.label[index]
            return mfcc, label
        return mfcc

train_dataset = CustomDataset(train_mfcc, train_labels)
val_dataset = CustomDataset(val_mfcc, val_labels)

train_loader = DataLoader(train_dataset, batch_size=CONFIG.BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=CONFIG.BATCH_SIZE, shuffle=False)

# Model
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, lstm_output):
        attention_weights = self.attention(lstm_output)
        attention_weights = torch.softmax(attention_weights, dim=1)
        context_vector = torch.sum(attention_weights * lstm_output, dim=1)
        return context_vector, attention_weights

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, dropout=0.1, bidirectional=False, device='cpu'):
        super(LSTMModel, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.directions = 2 if bidirectional else 1
        self.device = device
        self.layer_norm = nn.LayerNorm(input_dim)
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim,
                            num_layers=num_layers, dropout=dropout,
                            bidirectional=bidirectional, batch_first=True)
        self.attention = Attention(hidden_dim * self.directions)
        self.classifier = nn.Linear(hidden_dim * self.directions, output_dim)

    def _init_hidden(self, batch_size):
        n, d, hs = self.num_layers, self.directions, self.hidden_dim
        return (torch.zeros(n * d, batch_size, hs).to(self.device),
                torch.zeros(n * d, batch_size, hs).to(self.device))

    def forward(self, x):
        x = self.layer_norm(x)
        x = x.unsqueeze(1)  # Add sequence dimension
        hidden = self._init_hidden(x.size(0))
        lstm_out, (hn, cn) = self.lstm(x, hidden)
        context_vector, attention_weights = self.attention(lstm_out)
        out = self.classifier(context_vector)
        return out

# Pytorch Lightning Module
class LitModel(LightningModule):
    def __init__(self):
        super(LitModel, self).__init__()
        self.model = LSTMModel(input_dim=CONFIG.N_MFCC, hidden_dim=CONFIG.hidden_size,
                               output_dim=CONFIG.classifier_output, num_layers=CONFIG.num_layers,
                               dropout=CONFIG.dropout, bidirectional=CONFIG.bidirectional,
                               device=CONFIG.device)
        self.criterion = nn.BCEWithLogitsLoss()
        self.f1_score = MulticlassF1Score(num_classes=CONFIG.N_CLASSES, average="weighted").to(CONFIG.device)
        self.accuracy = MulticlassAccuracy(num_classes=CONFIG.N_CLASSES).to(CONFIG.device)

    def forward(self, x):
        if isinstance(x, tuple):
            mfcc, _ = x
        else:
            mfcc = x
        logit = self.model(mfcc)
        pred = torch.sigmoid(logit)
        return pred

    def training_step(self, batch, batch_idx):
        mfcc, labels = batch
        logits = self.model(mfcc)
        loss = self.criterion(logits, labels)
        preds = torch.sigmoid(logits)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        mfcc, labels = batch
        logits = self.model(mfcc)
        loss = self.criterion(logits, labels)
        preds = torch.sigmoid(logits)
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        mfcc, labels = batch
        logits = self.model(mfcc)
        loss = self.criterion(logits, labels)
        preds = torch.sigmoid(logits)
        self.log('test_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=CONFIG.LR)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=CONFIG.lr_reduce_factor,
                                                         patience=CONFIG.lr_patience, verbose=True)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

# Function to load model with partial state_dict
def load_checkpoint_partial(model, checkpoint_path, ignore_layers=[]):
    checkpoint = torch.load(checkpoint_path)
    state_dict = checkpoint['state_dict']

    new_state_dict = model.state_dict().copy()
    for name, param in state_dict.items():
        if any(ignore_layer in name for ignore_layer in ignore_layers):
            continue
        if name in new_state_dict and new_state_dict[name].shape == param.shape:
            new_state_dict[name] = param
    model.load_state_dict(new_state_dict)
    return model

# Train and Validate the Model
def main():
    output_dir = os.path.join(CONFIG.output_dir, 'checkpoints')
    os.makedirs(output_dir, exist_ok=True)
    model_checkpoint = ModelCheckpoint(dirpath=output_dir,
                                       filename=CONFIG.file_name,
                                       monitor="val_loss",
                                       save_top_k=1,
                                       mode="min")
    learning_rate_monitor = LearningRateMonitor(logging_interval="epoch")
    early_stopping = EarlyStopping(monitor="val_loss", patience=10, verbose=True, mode="min")
    logger = TensorBoardLogger("tb_logs", name="audio_classification")

    trainer = Trainer(accelerator='gpu' if CONFIG.device == "cuda" else 'cpu',
                      devices=1,
                      max_epochs=CONFIG.N_EPOCHS,
                      callbacks=[model_checkpoint, learning_rate_monitor, early_stopping],
                      logger=logger)
    lit_model = LitModel()
    trainer.fit(model=lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    best_model_path = model_checkpoint.best_model_path
    trainer.test(lit_model, ckpt_path=best_model_path, dataloaders=val_loader)

if __name__ == '__main__':
    main()

# Inference
def inference(model, test_loader, device):
    model.to(device)
    model.eval()
    predictions = []
    with torch.no_grad():
        for features in tqdm(test_loader):
            features = features.float().to(device)
            probs = model(features)
            probs = probs.cpu().detach().numpy()
            predictions += probs.tolist()
    return predictions

# Run Inference
test_df = pd.read_csv('../../open/test.csv')
test_mfcc = get_mfcc_feature(test_df, False)
test_dataset = CustomDataset(test_mfcc, None)
test_loader = DataLoader(test_dataset, batch_size=CONFIG.BATCH_SIZE, shuffle=False)

# Load model with partial loading to ignore unmatched keys
best_model = LitModel()
best_model = load_checkpoint_partial(best_model, './output/checkpoints/best_model.ckpt',
                                     ignore_layers=['model.lstm.weight_ih_l', 'model.lstm.weight_hh_l',
                                                    'model.lstm.bias_ih_l', 'model.lstm.bias_hh_l',
                                                    'model.classifier.weight', 'model.classifier.bias'])

preds = inference(best_model, test_loader, CONFIG.device)

# Submission
submit = pd.read_csv('../../open/sample_submission.csv')
submit.iloc[:, 1:] = preds
submit.to_csv('../../open/lstm.csv', index=False)