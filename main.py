import os
import torch
import numpy as np
from scipy.io import loadmat
import matplotlib as mpl
import matplotlib.pyplot as plt
from BiNMF import BiNMF
from spectral.graphics.graphics import get_rgb_meta
import warnings
warnings.filterwarnings("ignore")


def HSI2RGB(X, w, h, b):
    cube = X.T.view(w, h, b).cpu().detach().numpy()

    rgb, meta = get_rgb_meta(cube, bands=(29, 19, 9))

    return rgb

## GPU or CPU
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    print(torch.cuda.device_count(), "x", torch.cuda.get_device_name())
    dtype = torch.cuda.FloatTensor
else:
    dtype = torch.FloatTensor
    print("CPU")

dataset = 'moffett'
data = loadmat(f'{dataset}.mat')
X = torch.tensor(data['X']).type(dtype)
M = data['M']
M = (M - M.min(axis=0)) / (M.max(axis=0) - M.min(axis=0))

w = data['w'][0][0]
h = data['h'][0][0]
b = X.shape[0]
r = M.shape[1]
##################

W, H, Z, Omega, rel_err = BiNMF(X, r=r, iters=50, lam=1.2, gamma=0.9)

##################

mats =[H[0,:], H[1,:], H[2,:], Z[0,:], Z[1,:], Z[2,:]]
vmin = min(torch.min(M) for M in mats)
vmax = max(torch.max(M) for M in mats)


fig = plt.figure(figsize=(15, 9))

plt.subplot(3, 3, 1)
plt.imshow(HSI2RGB(X, w, h, b))
plt.title('HSI')
plt.axis("off")

plt.subplot(3, 3, 2)
plt.plot(M)
plt.title('Ground-truth')

Ws = W[:-1, :].cpu().numpy()
Ws = (Ws - Ws.min(axis=0)) / (Ws.max(axis=0) - Ws.min(axis=0))

plt.subplot(3, 3, 3)
plt.plot(Ws)
plt.title('BiNMF')

plt.subplot(3, 3, 4)
plt.imshow(H[0,:].view(w, h).cpu().numpy(), cmap='gray', vmin=vmin, vmax=vmax)
plt.title('H_1 (Soil)')
plt.axis("off")

plt.subplot(3, 3, 5)
plt.imshow(H[1,:].view(w, h).cpu().numpy(), cmap='gray', vmin=vmin, vmax=vmax)
plt.title('H_2 (Vegetation)')
plt.axis("off")

plt.subplot(3, 3, 6)
plt.imshow(H[2,:].view(w, h).cpu().numpy(), cmap='gray', vmin=vmin, vmax=vmax)
plt.title('H_3 (Water)')
plt.axis("off")

plt.subplot(3, 3, 7)
plt.imshow(Z[0,:].view(w, h).cpu().numpy(), cmap='gray', vmin=vmin, vmax=vmax)
plt.title('Z_1 (So + Ve)')
plt.axis("off")

plt.subplot(3, 3, 8)
plt.imshow(Z[1,:].view(w, h).cpu().numpy(), cmap='gray', vmin=vmin, vmax=vmax)
plt.title('Z_2 (So + Wa)')
plt.axis("off")

plt.subplot(3, 3, 9)
plt.imshow(Z[2,:].view(w, h).cpu().numpy(), cmap='gray', vmin=vmin, vmax=vmax)
plt.title('Z_3 (Ve + Wa)')
plt.axis("off")


plt.show()
