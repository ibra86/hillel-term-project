import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

import random
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset
import os
import time
from utils import ZNormalizer
from constants import DATA_DIR, DATA_PREDICT_FILE_NAME, LOSS_FILE_NAME

start_time = time.time()
from data_preprocessing import df

# observed only
df_obs = df[df.var_obs == 1].reset_index(drop=True)
df_obs0 = df[df.var_obs == 0].reset_index(drop=True)

x_train, x_test, y_train, y_test = train_test_split(df_obs.var_features, df_obs.var_target, random_state=0)

x_train, x_test = np.array(x_train.values.tolist()), np.array(x_test.values.tolist())
y_train, y_test = np.array(y_train.values.tolist()), np.array(y_test.values.tolist())
print('Training...')

zn = ZNormalizer()
zn.fit(x_train)

x_train, x_test = zn.transform(x_train), zn.transform(x_test)
x_train, x_test = torch.tensor(x_train).float(), torch.tensor(x_test).float()
y_train, y_test = torch.tensor(y_train).float(), torch.tensor(y_test).float()

N_input_dim = x_train.shape[1]


class MixupAug(Dataset):

    def __init__(self, x_data, y_data, alpha):
        self.x_data = x_data
        self.y_data = y_data
        self.alpha = alpha

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, i):
        x, y = self.x_data[i], self.y_data[i]
        if self.alpha != 0:
            x2 = self.x_data[random.randint(0, len(self.x_data) - 1)]
            lam = np.random.beta(self.alpha, self.alpha)
            x = lam * x + (1. - lam) * x2
        return x, y


train_ds = MixupAug(x_train, y_train, 0.4)


class AAreg(nn.Module):
    def __init__(self, N_input_dim):
        super().__init__()

        size = N_input_dim
        depth = 1
        size1 = size // 2
        size2 = size1 // 2
        self.layer1 = nn.Linear(N_input_dim, size1)
        self.layer2 = nn.ModuleList([nn.Linear(size1, size2) for _ in range(depth)])
        self.layer3 = nn.Linear(size2, 1)
        self.dropout = nn.Dropout2d(p=0.5)

    def forward(self, x):
        x = self.dropout(torch.relu(self.layer1(x)))
        for l in self.layer2:
            x = self.dropout(torch.relu(l(x)))
        x = self.layer3(x)
        x = x.squeeze(1)
        return x


model = AAreg(N_input_dim)
model = model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
loss = nn.MSELoss()

train_dl = torch.utils.data.DataLoader(train_ds, batch_size=512, shuffle=True)

loss_train_list, loss_test_list = [], []

epochs = 500
k = 50

for epoch in range(epochs + 1):
    model.train()

    for x_batch_train, y_batch_train in train_dl:
        x_batch_train = x_batch_train.cuda()
        y_batch_train = y_batch_train.cuda()

        y_hat_batch_train = model(x_batch_train)
        loss_train = loss(y_batch_train, y_hat_batch_train)

        break

    if epoch % k == 0:
        with torch.no_grad():
            model.eval()

            x_test = x_test.cuda()
            y_test = y_test.cuda()
            y_hat_test = model(x_test)
            loss_test = loss(y_test, y_hat_test)

            delta_space_print = len(f'[{epochs}/{epochs} ({epochs / epochs * 100:.0f}%)]') - \
                                len(f'[{epoch}/{epochs} ({epoch / epochs * 100:.0f}%)]')
            print(f'Train Epoch: [{epoch}/{epochs} ({epoch / epochs * 100:.0f}%)]' +
                  ' ' * (4 + delta_space_print) +
                  f' Loss: {loss_test:.2f}')

        loss_train_list.append(loss_train.data.cpu().numpy())
        loss_test_list.append(loss_test.data.cpu().numpy())

    optimizer.zero_grad()
    loss_train.backward()
    optimizer.step()

plt.figure(figsize=(15, 5))
plt.plot(loss_train_list, label='factual')
plt.plot(loss_test_list, label='predicted')
plt.legend()
plt.title('Loss')
plt.ylim((0, 2000))
plt.grid(True)
# plt.show()
plt.savefig(os.path.join(DATA_DIR, LOSS_FILE_NAME))
print('loss plot saved to file %s' % os.path.join(DATA_DIR, LOSS_FILE_NAME))

x_predict = np.array(df_obs0.var_features.values.tolist())
x_predict = zn.transform(x_predict)
x_predict = torch.tensor(x_predict).float()

y_predict = model(x_predict.cuda())

df_obs0 = df_obs0.drop('var_features', axis=1)
df_obs0['var_target_hat'] = y_predict.data.cpu().numpy()
df_obs0['var_target_hat'] = df_obs0['var_target_hat'].map(lambda x: int(x))

# saving prediction result
df.to_csv(os.path.join(DATA_DIR, DATA_PREDICT_FILE_NAME), index=False)
print('prediction saved to file %s' % os.path.join(DATA_DIR, DATA_PREDICT_FILE_NAME))

print('Final score:')
print('MSE(train/test): %s/%s' % (int(loss_train_list[-1]), int(loss_test_list[-1])), '--',
      'RMSE(train/test): %s/%s' % (int(np.sqrt(loss_train_list[-1])), int(np.sqrt(loss_test_list[-1]))))

print('--Total time of execution: %0.2fs' % (time.time() - start_time))
