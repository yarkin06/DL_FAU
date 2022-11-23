import torch as t
from data import ChallengeDataset
from trainer import Trainer
from matplotlib import pyplot as plt
import numpy as np
import model
import pandas as pd
from sklearn.model_selection import train_test_split

csv_path = 'data.csv'
tab = pd.read_csv(csv_path, sep=';')
rand = np.random.randint(1, 1000)
train_tab, test_tab = train_test_split(tab, test_size=0.99, random_state=rand)

train_dl = t.utils.data.DataLoader(ChallengeDataset(train_tab, 'train'), batch_size=10)
val_dl = t.utils.data.DataLoader(ChallengeDataset(test_tab, 'val'), batch_size=10)
model = model.ResNet()
crit = t.nn.BCELoss()
optimizer = t.optim.Adam(model.parameters(), lr=0.001)
trainer = Trainer(model, crit, optimizer, train_dl, val_dl, cuda=True)
trainer.restore_checkpoint(0, 'checkpoints/')
trainer.save_onnx('last_model.onnx')