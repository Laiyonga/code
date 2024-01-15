import os
import torch
import torch.nn as nn
#from model import SLKNet
import torch.optim as optim
import numpy as np
import random
from PIL import Image
from tqdm import tqdm
from dataset import MyData
from torch.utils.data import DataLoader
#from model import UNet
from segnet import SegNet
from Transform import Net
from DownTransformer import DownNet
from focal_frequency_loss import FocalFrequencyLoss as FFL
from loss import CustomLoss
from CDDFuseNet import CDDFuseNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#model = Net(upscale_factor=1).to(device)
model = DownNet(upscale_factor=1).to(device)
# model.load_state_dict(torch.load(r'E:\code\SkL-pytorch\last.pt'))

optimizer = optim.Adam(model.parameters(), lr=0.0001)

#criterion = CustomLoss()
mse = nn.MSELoss()

train_dataset = MyData(r"C:\Users\Administrator\Desktop\data\train")
print(len(train_dataset))
train_loader = DataLoader(
    train_dataset,
    batch_size=1,
    shuffle=True
)

best = 100
for epoch in range(50):
    pbar = tqdm(train_loader)
    for idx_iter, (batch_data, batch_labels) in enumerate(train_loader):
        # 前向计算
        batch_data = batch_data.to(device).float()
        batch_labels = batch_labels.to(device).float()
        outputs = model(batch_data)
        #loss = criterion(outputs, batch_labels) + ffl(outputs, batch_labels)
        #loss1 = criterion(outputs, batch_labels)
        # loss2 = ffl(outputs, batch_labels)
        #mae_loss, ssim_loss, gradient_loss = criterion(outputs, batch_labels)
        mse_loss = mse(outputs, batch_labels)
        loss = mse_loss
        # 反向传播和优化
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        #pbar.set_description(f'Loss: {loss.item():.6f}, loss1:{loss1.item():.6f},loss2:{loss2.item():.6f}')
        pbar.set_description(f'epoch:{epoch+1},loss:{loss.item():.6f}')
        pbar.update(1)
        if idx_iter % 100 == 0:
            #print(idx_iter, float(np.array(loss_epoch).mean()), float(np.array(optimizer.param_groups[0]['lr']).mean()))
            idx = random.randint(0, 0)
            d = batch_data[idx, :, :, :].clone().detach().requires_grad_(False)
            d = torch.transpose(d, 0, 1)
            d = torch.transpose(d, 1, 2).cpu().numpy() * 255

            rf = outputs[idx, :, :, :].clone().detach().requires_grad_(False)
            rf = torch.transpose(rf, 0, 1)
            rf = torch.transpose(rf, 1, 2).cpu().numpy() * 255
            r = batch_labels[idx, :, :, :].clone().detach().requires_grad_(False)
            r = torch.transpose(r, 0, 1)
            r = torch.transpose(r, 1, 2).cpu().numpy() * 255


            # d = np.concatenate([d, mid_left,rf, r, d1, mid_right,rf1, r1], axis=1)
            d = np.concatenate([d, rf, r], axis=0)
            d = np.squeeze(d)
            image_name = 'sample' + "/out" + str(epoch) + '_' + str(idx_iter) + ".png"
            # Image.fromarray(np.uint8(d)).convert('RGB').save(image_name)
            Image.fromarray(np.uint8(d)).convert('L').save(image_name)
    if loss.item() < best:
        torch.save(model.state_dict(), 'best.pt')
        best = loss.item()
    torch.save(model.state_dict(), 'last.pt')

