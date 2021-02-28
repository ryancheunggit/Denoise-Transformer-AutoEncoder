import torch
import numpy as np
from datetime import datetime
from util import AverageMeter
from model import SwapNoiseMasker, TransformerAutoEncoder
from data import get_data, SingleDataset
from torch.utils.data import DataLoader
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error


# Hyper-params
model_params = dict(
    hidden_size=1024,
    num_subspaces=8,
    embed_dim=128,
    num_heads=8,
    dropout=0,
    feedforward_dim=512,
    emphasis=.75,
    mask_loss_weight=2
)
batch_size = 384
init_lr = 3e-4
lr_decay = .998
max_epochs = 2001

repeats = [  2,  2,  2,  4,  4,  4,  8,  8,  7, 15,  14]
probas =  [.95, .4, .7, .9, .9, .9, .9, .9, .9, .9, .25]
swap_probas = sum([[p] * r for p, r in zip(probas, repeats)], [])

#  get data
X, Y, n_cats, n_nums = get_data()

train_dl = DataLoader(
    dataset=SingleDataset(X),
    batch_size=batch_size,
    shuffle=True,
    pin_memory=True,
    drop_last=True
)

# setup model
model = TransformerAutoEncoder(
    num_inputs=X.shape[1],
    n_cats=n_cats,
    n_nums=n_nums,
    **model_params
).cuda()
model_checkpoint = 'model_checkpoint.pth'

print(model)

noise_maker = SwapNoiseMasker(swap_probas)
optimizer = torch.optim.Adam(model.parameters(), lr=init_lr)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay)

# train model
for epoch in range(max_epochs):
    t0 = datetime.now()
    model.train()
    meter = AverageMeter()
    for i, x in enumerate(train_dl):
        x = x.cuda()
        x_corrputed, mask = noise_maker.apply(x)
        optimizer.zero_grad()
        loss = model.loss(x_corrputed, x, mask)
        loss.backward()
        optimizer.step()

        meter.update(loss.detach().cpu().numpy())

    delta = (datetime.now() - t0).seconds
    scheduler.step()
    print('\r epoch {:5d} - loss {:.6f} - {:4.6f} sec per epoch'.format(epoch, meter.avg, delta), end='')

torch.save({
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "model": model.state_dict()
    }, model_checkpoint
)
model_state = torch.load(model_checkpoint)
model.load_state_dict(model_state['model'])

# extract features
dl = DataLoader(dataset=SingleDataset(X), batch_size=1024, shuffle=False, pin_memory=True, drop_last=False)
features = []
model.eval()
with torch.no_grad():
    for x in dl:
        features.append(model.feature(x.cuda()).detach().cpu().numpy())
features = np.vstack(features)

# downstream supervised regressor
alpha = 1250 # 1000
X = features[:300_000, :]
scores = []
for train_idx, valid_idx in KFold().split(X, Y):
    scores.append(mean_squared_error(Y[valid_idx], Ridge(alpha=1250).fit(X[train_idx], Y[train_idx]).predict(X[valid_idx]), squared=False))
print(np.mean(scores))

np.save('dae_features.npy', features)

