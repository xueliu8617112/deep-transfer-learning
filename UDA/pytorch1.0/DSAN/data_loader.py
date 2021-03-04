from torchvision import datasets, transforms
import torch
import torch.utils.data
from torch.utils.data import DataLoader, Dataset, TensorDataset
import numpy as np
from scipy.io import loadmat


def load_training(root_path, dir, batch_size, kwargs):
    transform = transforms.Compose(
        [transforms.Resize([256, 256]),
         transforms.RandomCrop(224),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor()])
    data = datasets.ImageFolder(root=root_path + dir, transform=transform)
    train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
    return train_loader

def load_testing(root_path, dir, batch_size, kwargs):
    transform = transforms.Compose(
        [transforms.Resize([224, 224]),
         transforms.ToTensor()])
    data = datasets.ImageFolder(root=root_path + dir, transform=transform)
    test_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, **kwargs)
    return test_loader

class DealDataset1(Dataset):
    """
        下载数据、初始化数据，都可以在这里完成
    """

    def __init__(self):
        path = r'E:\code-Xue\deep-transfer-learning\UDA\pytorch1.0\DSAN\PUFFTmixed02.mat'
        f = loadmat(path)
        xy = f['data_s_train']
        lab = f['label_s_train']
        lab = np.reshape(lab, (-1,)).tolist()
        xy = np.reshape(xy, (-1, 512, 1))
        self.x_data = np.transpose(xy, (0, 2, 1)).astype(np.float32)
        self.y_data = lab
        self.len = xy.shape[0]

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


class DealDataset(Dataset):
    """
        下载数据、初始化数据，都可以在这里完成
    """

    def __init__(self):
        path = r'E:\code-Xue\deep-transfer-learning\UDA\pytorch1.0\DSAN\PUFFTmixed02.mat'
        f = loadmat(path)
        xy = f['data_t_train']
        lab = f['label_t_train']
        lab = np.reshape(lab, (-1,)).tolist()
        xy = np.reshape(xy, (-1, 512, 1))
        self.x_data = np.transpose(xy, (0, 2, 1)).astype(np.float32)
        self.y_data = lab
        self.len = xy.shape[0]

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

class DealDataset2(Dataset):
    """
        下载数据、初始化数据，都可以在这里完成
    """

    def __init__(self):
        path = r'E:\code-Xue\deep-transfer-learning\UDA\pytorch1.0\DSAN\PUFFTmixed02.mat'
        f = loadmat(path)
        xy = f['data_s_test']
        lab = f['label_s_test']
        lab = np.reshape(lab, (-1,)).tolist()
        xy = np.reshape(xy, (-1, 512, 1))
        self.x_data = np.transpose(xy, (0, 2, 1)).astype(np.float32)
        self.y_data = lab
        self.len = xy.shape[0]

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

class DealDataset3(Dataset):
    """
        下载数据、初始化数据，都可以在这里完成
    """

    def __init__(self):
        path = r'E:\code-Xue\deep-transfer-learning\UDA\pytorch1.0\DSAN\PUFFTmixed02.mat'
        f = loadmat(path)
        xy = f['data_t_test']
        lab = f['label_t_test']
        lab = np.reshape(lab, (-1,)).tolist()
        xy = np.reshape(xy, (-1, 512, 1))
        self.x_data = np.transpose(xy, (0, 2, 1)).astype(np.float32)
        self.y_data = lab
        self.len = xy.shape[0]

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

class DealDataset4(Dataset):
    """
        下载数据、初始化数据，都可以在这里完成
    """

    def __init__(self):
        path = r'E:\code-Xue\deep-transfer-learning\UDA\pytorch1.0\DSAN\PUFFTnormal2add.mat'
        f = loadmat(path)
        xy = f['data']
        lab = f['label']
        lab = np.reshape(lab, (-1,)).tolist()
        xy = np.reshape(xy, (-1, 512, 1))
        self.x_data = np.transpose(xy, (0, 2, 1)).astype(np.float32)
        self.y_data = lab
        self.len = xy.shape[0]

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len
