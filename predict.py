import cv2
import torch
import os
import re
import torchvision.transforms
from matplotlib import pyplot as plt
import scipy.io
import numpy as np
from BpFilter import BpFilter
from utils import *
from matplotlib.colors import ListedColormap
from model import UNet
from segnet import SegNet
from Transform import Net
from DownTransformer import DownNet
from CDDFuseNet import CDDFuseNet
from FkFilter import FkFilter
from LocalFkFilter import LocalFkFilter
from FilterMask import PieShapeFkFilterMask
from PIL import Image
import time
def sortKey(s):
    re_digits = re.compile(r'(\d+)')
    pieces = re_digits.split(s)
    pieces[1::2] = map(int, pieces[1::2])
    return pieces
start_time = time.time()
device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
model = DownNet(upscale_factor=1).to(device)
model.eval()
model.load_state_dict(torch.load(r'E:\code\SkL-pytorch\best.pt'), strict=True)

# 测试数据
#savePath = r'E:\code\SkL-pytorch\result33'
filelist = os.listdir(r'C:\Users\Administrator\Desktop\333')
filelist1 = os.listdir((r'C:\Users\Administrator\Desktop\data\test\processed'))
filelist.sort(key=sortKey)
filelist1.sort(key=sortKey)
for index,file in enumerate(filelist):
    #[startTime, endTime, xStart, xEnd, nt, nx, dt, dx, data] = ReadBinary1(r'E:\yongjinTraffic\data_stack\\'+file)
    data = Image.open(r'C:\Users\Administrator\Desktop\333\\'+file).convert('L')
    data2 = Image.open(r'C:\Users\Administrator\Desktop\data\test\processed\\' + filelist1[index]).convert('L')
    data1 = torchvision.transforms.ToTensor()(data)
    data1 = torchvision.transforms.Resize((224, 224))(data1)
    data1 = data1.unsqueeze(0)
    # data1 = torch.from_numpy(data).reshape(1, 1, data.shape[0], data.shape[1]).float()
    data1.to(device)
    # 使用模型进行预测
    with torch.no_grad():
        result = model(data1)
    predicted = result.numpy()
    predicted = predicted.squeeze()
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(3, 1, 1)
    plt.imshow(data, cmap='gray', aspect='auto')
    # #plt.axis('off')
    # plt.title('stack')

    ax = fig.add_subplot(3, 1, 2)
    #cv2.imwrite(savePath + "/" + file[:-4] + "_" + ".png", predicted * 255)
    # predicted[predicted>0.5*np.max(np.max(abs(predicted)))]= 0.5*np.max(np.max(abs(predicted)))
    # predicted[predicted<-0.5*np.max(np.max(abs(predicted)))]= -0.5*np.max(np.max(abs(predicted)))
    # print(np.max(np.max(predicted)))
    # print(np.min(np.min(predicted)))
    plt.imshow(predicted, cmap='gray',aspect='auto')
    #plt.show()
    #plt.axis('off')
    #plt.savefig(savePath + "/" + file[:-4] + ".png", bbox_inches='tight', pad_inches=-0.1)
    # stackStep = 10
    # stackDt = stackStep * 0.002
    # ntData, nxData = predicted.shape
    # #dataXEnd = xStart + nxData
    # dx = 2
    # fid = open(savePath + "/" + file[:-4] + ".bin", "wb")
    # #fid.write(pack('2d', *[startTime, startTime + ntData * stackDt]))
    # #fid.write(pack('2d', *[xStart * dx, dataXEnd * dx]))
    # fid.write(pack('2i', *[ntData, nxData]))
    # fid.write(pack('2d', *[stackDt, dx]))
    # fid.write(pack(str(ntData * nxData) + 'd', *(predicted.astype(float).flat)))
    # fid.close()

    ax = fig.add_subplot(3, 1, 3)
    #[startTime, endTime, xStart, xEnd, nt, nx, dt, dx, data2] = ReadBinary1(r"E:\yongjinTraffic\test\processed\2023-03-09_09_48_18_3_processed.bin")
    #[startTime, endTime, xStart, xEnd, nt, nx, dt, dx, data2] = ReadBinary1(r'E:\yongjinTraffic\data_processed_img\\'+filelist1[index])

    plt.imshow(data2, cmap='gray', aspect='auto')
    # # # plt.axis('off')
    # plt.title('v15')
    plt.show()
    # # plt.savefig(r'./test2.png', bbox_inches='tight',pad_inches =-0.1)
    # plt.savefig(savePath + "/" + file[:-4] + ".png",bbox_inches='tight', pad_inches=-0.1)
    # plt.savefig(savePath + "/" + file[:-4] + ".png",bbox_inches='tight')
    # plt.close()