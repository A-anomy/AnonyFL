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
        
    def download(self):
        data_train = datasets.MNIST(root = self.cfg["base_path"] + "/data",
                                    # transform=transform,
                                    train = True,
                                    download = True)

        data_test = datasets.MNIST(root = self.cfg["base_path"] + "/data",
                                #    transform = transform,
                                train = False)


    # 加载MNIST训练数据
    def load_data(self, load_test = False):
        self.download()
        images_file = self.cfg["base_path"] + '/data/MNIST/raw/train-images-idx3-ubyte'
        labels_file = self.cfg["base_path"] + '/data/MNIST/raw/train-labels-idx1-ubyte'

        images = np.fromfile(images_file, dtype=np.uint8)[16:].reshape((-1, 28, 28))
        labels = np.fromfile(labels_file, dtype=np.uint8)[8:]

        if load_test == True:
            images_file = self.cfg["base_path"] + '/data/MNIST/raw/t10k-images-idx3-ubyte'
            labels_file = self.cfg["base_path"] + '/data/MNIST/raw/t10k-labels-idx1-ubyte'
            images2 = np.fromfile(images_file, dtype=np.uint8)[16:].reshape((-1, 28, 28))
            labels2 = np.fromfile(labels_file, dtype=np.uint8)[8:]
            images = np.append(images, images2, axis=0)
            labels = np.append(labels, labels2, axis=0)

        return images, labels


    # 保存分配好的数据到文件夹
    def save_data_to_folders(self, data_splits, num_clients):
        save_dir = self.cfg["base_path"] + '/data/MNIST/client_data/{}'.format(str(num_clients))

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
        data_dir = os.path.join(self.cfg["base_path"] + '/data/MNIST/client_data/{}/client_{}'.format(str(num_clients), str(client_id)))
        images_file = os.path.join(data_dir, 'images.npy')
        labels_file = os.path.join(data_dir, 'labels.npy')
        images = np.load(images_file)
        labels = np.load(labels_file)
        # images = self.preprocess_data(images)

        return images, labels


class DataLoader():
    def __init__(self, cfg, dev_id, batch_size = None):
        self.cfg = cfg
        self.data_dir = os.path.join(cfg["base_path"]+r"/data/MNIST/client_data/{}".format(str(cfg["num_clients"])),"client_{}".format(str(dev_id)))
        self.batch_size = batch_size
        self.images, self.labels = self.load_data()
        # print(self.labels.shape)
    
    def get_class_num(self):
        return 10

    def __len__(self):
        return self.labels.shape[0]

    def load_data(self):
        images_file = os.path.join(self.data_dir, 'images.npy')
        labels_file = os.path.join(self.data_dir, 'labels.npy')
        images = np.load(images_file)
        labels = np.load(labels_file)
        if self.cfg["test_method"] == "distribution":
            images = images[:int(images.__len__()*0.75)]
            labels = labels[:int(labels.__len__()*0.75)]
        return images, labels

    def preprocess_data(self, images):
        images = images.astype(np.float32)
        images /= 255.0
        mean = np.mean(images)
        std = np.std(images)
        images = (images - mean) / std

        return images
    
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
            if process:
                batch_images = self.preprocess_data(batch_images)
            batch_images = torch.from_numpy(batch_images)
            batch_labels = self.labels[start:end]
            batch_labels = torch.nn.functional.one_hot(torch.from_numpy(batch_labels).to(torch.int64),num_classes=10).to(dtype=torch.float32)
            # print("{}, {}".format(batch_images.shape,batch_labels.shape))
            yield batch_images, batch_labels

class TestLoader:
    def __init__(self, cfg, batch_size, distribute = False, client_id = 0):
        self.cfg = cfg
        self.data_dir = cfg["base_path"] + r"/data/MNIST"
        self.batch_size = batch_size
        self.images, self.labels = self.load_data(distribute=distribute, client_id=client_id)

    def get_class_num(self):
        return 10
    
    def load_data(self, distribute = False, client_id = 0):
        if distribute:
            data_dir = os.path.join(self.cfg["base_path"]+r"/data/MNIST/client_data/{}".format(str(self.cfg["num_clients"])),"client_{}".format(str(client_id)))
            images_file = os.path.join(data_dir, 'images.npy')
            labels_file = os.path.join(data_dir, 'labels.npy')
            images = np.load(images_file)
            labels = np.load(labels_file)
            images = images[int(images.__len__()*0.75+1):]
            labels = labels[int(labels.__len__()*0.75+1):]
        else:
            images_file = os.path.join(self.data_dir, 'raw/t10k-images-idx3-ubyte')
            labels_file = os.path.join(self.data_dir, 'raw/t10k-labels-idx1-ubyte')
            images = np.fromfile(images_file, dtype=np.uint8)[16:].reshape((-1, 28, 28))
            labels = np.fromfile(labels_file, dtype=np.uint8)[8:]
            images = self.preprocess_data(images)
        return images, labels

    def preprocess_data(self, images):
        images = images.astype(np.float32)
        images /= 255.0
        mean = np.mean(images)
        std = np.std(images)
        images = (images - mean) / std

        return images

    def get_batches(self):
        num_samples = len(self.labels)
        num_batches = int(math.ceil(num_samples / self.batch_size))

        for i in range(num_batches):
            start = i * self.batch_size
            end = min(start + self.batch_size, self.labels.__len__())
            batch_images = self.images[start:end]
            batch_labels = self.labels[start:end]
            batch_images = self.preprocess_data(batch_images)
            batch_images = torch.from_numpy(batch_images)
            batch_labels = torch.from_numpy(batch_labels)
            
            yield batch_images, batch_labels


if __name__ == "__main__":
    project_path = r"/data/wangweicheng/czj/Vateer-FL"
    data_train = torchvision.datasets.MNIST(root = os.path.join(project_path,"data"),
                            # transform=transform,
                            train = True,
                            download = False)

    data_test = torchvision.datasets.MNIST(root = os.path.join(project_path,"data"),
                        #    transform = transform,
                           train = False, download=False)
    for i,j in data_train:
        print(j)
