import torch, urllib
import torchvision
import os, math
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from data_handler.base_handler import BaseHandler
from torchvision import datasets, transforms

class Handler(BaseHandler):
    def __init__(self, cfg:dict):
        self.cfg = cfg

    # 加载MNIST训练数据
    def load_data(self):
        if not os.path.exists(self.cfg["base_path"] + "/data/cifa-100"):
            os.makedirs(self.cfg["base_path"] + "/data/cifa-100")
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        data_train = datasets.CIFAR100(root = self.cfg["base_path"] + "/data/cifa-100",
                                    # transform=transform,
                                    train = True, transform=transform,
                                    download = True)
        data_loader = torch.utils.data.DataLoader(data_train, batch_size=len(data_train.data), shuffle = False)
        
        
        for data in data_loader:
            data_train.data, data_train.targets = data
        # for data in test_loader:
        #     data_test.data, data_test.targets = data        

        images, labels = [], []

        images.extend(data_train.data.cpu().detach().numpy())
        labels.extend(data_train.targets.cpu().detach().numpy())

        images = np.array(images)
        labels = np.array(labels)

        return images, labels

    # 保存分配好的数据到文件夹
    def save_data_to_folders(self, data_splits, num_clients):
        save_dir = self.cfg["base_path"] + '/data/cifa-100/client_data/{}'.format(str(num_clients))

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for i, (images, labels) in enumerate(data_splits):
            client_dir = os.path.join(save_dir, f'client_{i}')
            if not os.path.exists(client_dir):
                os.makedirs(client_dir)

            images_file = os.path.join(client_dir, 'images.npy')
            labels_file = os.path.join(client_dir, 'labels.npy')
            np.save(images_file, images)
            np.save(labels_file, labels)
    
    # 从文件夹加载数据进行训练
    def load_data_from_folders(self, client_id, num_clients):
        data_dir = os.path.join(self.cfg["base_path"] + '/data/cifa-100/client_data/{}/client_{}'.format(str(num_clients), str(client_id)))
        images_file = os.path.join(data_dir, 'images.npy')
        labels_file = os.path.join(data_dir, 'labels.npy')
        images = np.load(images_file)
        labels = np.load(labels_file)
        # images = self.preprocess_data(images)

        return images, labels


class DataLoader:
    def __init__(self, cfg, dev_id, batch_size = None):
        self.cfg = cfg
        self.data_dir = os.path.join(cfg["base_path"]+r"/data/cifa-100/client_data/{}".format(str(cfg["num_clients"])),"client_{}".format(str(dev_id)))
        self.batch_size = batch_size
        self.images, self.labels = self.load_data()
        # print(self.labels.shape)
    
    def get_class_num(self):
        return 100

    def __len__(self):
        return self.labels.shape[0]

    def load_data(self):
        images_file = os.path.join(self.data_dir, 'images.npy')
        labels_file = os.path.join(self.data_dir, 'labels.npy')

        images = np.load(images_file)
        labels = np.load(labels_file)

        return images, labels

    def preprocess_data(self, images):
        pass
    
    def load_data_manually(self, images:list, labels:list):
        self.images = np.array(images)
        self.labels = np.array(labels)
        self.images = np.append(self.images, images)
        self.labels = np.append(self.labels, labels)



    def get_batches(self, process = True, bz = 0):
        if bz == 0:
            bz = self.batch_size
        num_samples = len(self.labels)
        num_batches = math.ceil(num_samples / bz)
        shuffle_idx = list(range(len(self.images)))
        np.random.shuffle(shuffle_idx)
        self.images = self.images[shuffle_idx]
        self.labels = self.labels[shuffle_idx]
        for i in range(num_batches):
            start = i * bz
            end = min(start + bz, len(self.images))
            batch_images = self.images[start:end]
            # if process:
            #     batch_images = self.preprocess_data(batch_images)
            batch_images = torch.from_numpy(batch_images)
            batch_labels = self.labels[start:end]
            batch_labels = torch.nn.functional.one_hot(torch.from_numpy(batch_labels).to(torch.int64),num_classes=self.get_class_num()).to(dtype=torch.float32)
            # print("{}, {}".format(batch_images.shape,batch_labels.shape))
            yield batch_images, batch_labels

class TestLoader:
    def __init__(self, cfg, batch_size):
        self.cfg = cfg
        self.batch_size = batch_size
        self.images, self.labels = self.load_data()

    def get_class_num(self):
        return 100
    
    def load_data(self):
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        data_test = datasets.CIFAR100(root = self.cfg["base_path"] + "/data/cifa-100",
                                transform = transform,
                                train = False, download = True)
        test_loader = torch.utils.data.DataLoader(data_test, batch_size=len(data_test.data), shuffle = False)


        for data in test_loader:
            data_test.data, data_test.targets = data     

        images, labels = [], []

        images.extend(data_test.data.cpu().detach().numpy())
        labels.extend(data_test.targets.cpu().detach().numpy())

        images = np.array(images)
        labels = np.array(labels)

        return images, labels

    def preprocess_data(self, images):
        pass

    def get_batches(self):
        num_samples = len(self.labels)
        num_batches = num_samples // self.batch_size

        for i in range(num_batches):
            start = i * self.batch_size
            end = start + self.batch_size
            batch_images = self.images[start:end]
            batch_labels = self.labels[start:end]
            # batch_images = self.preprocess_data(batch_images)
            batch_images = torch.from_numpy(batch_images)
            batch_labels = torch.from_numpy(batch_labels)
            
            yield batch_images, batch_labels


if __name__ == "__main__":
    pass
