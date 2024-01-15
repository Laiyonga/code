import glob
import os
from torch.utils.data import Dataset
from utils import *
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
class MyData(Dataset):
    def __init__(self, data_path):
        super().__init__()
        self.data_path = data_path
        self.noise_path = os.path.join(data_path, 'stack')
        self.clean_path1 = os.path.join(data_path, 'processed')
        # self.clean_path2 = os.path.join(data_path, '')
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224))
        ])
        self.x, self.y = self.load_data()[0], self.load_data()[1]
        print('load dataset over !!!')

        assert len(self.x) == len(self.y), f'真实图片和标签数据长度不一致'

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def load_data(self):
        # w1 = 64
        # w2 = 64
        # z1 = 32
        # z2 = 32

        x = []
        y = []
        for i in tqdm(glob.glob(self.noise_path + '\\*.png')):
            # file_name = os.path.join(self.clean_path, 'clean_' + i.split('\\')[-1].split('.bin')[0][6:] + '.bin')
            #file_name = os.path.join(self.clean_path, i.split('\\')[-1].split('.bin')[0] + '_processed' + '.bin')
            file_name = os.path.join(self.clean_path1, i.split('\\')[-1])
            print(i)
            print(file_name)
            # file_name1 = os.path.join(self.clean_path2, i.split(('\\')[-1].split('.png')[0]+''+'.png'))
            # x_data = yc_patch(ReadBinary(i), w1, w2, z1, z2)
            # y_data = yc_patch(ReadBinary(file_name), w1, w2, z1, z2)
            # print(i)
            # print(file_name)
            # x_data = ReadBinary(i)
            # y_data = ReadBinary(file_name)
            x_data = Image.open(i).convert('L')
            y_data = Image.open(file_name).convert('L')
            x.append(self.transform(x_data))
            y.append(self.transform(y_data))

        # X = np.array(x).astype(np.float32)
        # Y = np.array(y).astype(np.float32)
        #
        # X = np.reshape(X, (X.shape[0]*X.shape[1], w1, w2, 1))
        # Y = np.reshape(Y, (Y.shape[0]*Y.shape[1], w1, w2, 1))
        # x = torch.from_numpy(X)
        # y = torch.from_numpy(Y)
        # return x.permute(0, 3, 1, 2), y.permute(0, 3, 1, 2)
        return x, y


if __name__ == '__main__':
    path = r'E:\data'
    Dataset = MyData(path)
    print(Dataset)
    # x, y = Dataset.load_data()
    X, Y = Dataset[0]
    print(X.shape, Y.shape)

