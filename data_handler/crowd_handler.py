import torch, urllib
import torchvision
import os, math, random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from data_handler.base_handler import BaseHandler
from torchvision import datasets, transforms

class Handler(BaseHandler):
    def __init__(self, cfg:dict):
        self.cfg = cfg

    def preprocess_data(self, images):
        images = images.astype(np.float32)
        images /= 255.0
        mean = np.mean(images)
        std = np.std(images)
        images = (images - mean) / std

        return images
    
    # 加载训练数据
    def load_data(self, load_test = False):
        images, labels = [], []
        file_labels = []
        file_labels1 = []
        # m1 = Image.open("./data/crowd/UCSDped1/Train/Train001/001.tif")
        # m2 = Image.open("./data/crowd/UCSDped2/Train/Train001/001.tif")
        with open("data/crowd/label/UCSDped1_train.txt","r") as f:
            file_labels1 = f.readlines()
        file_labels += file_labels1
        with open("data/crowd/label/UCSDped1_test.txt","r") as f:
            file_labels1 = f.readlines()
        file_labels += file_labels1
        with open("data/crowd/label/UCSDped2_train.txt","r") as f:
            file_labels1 = f.readlines()
        file_labels += file_labels1
        with open("data/crowd/label/UCSDped2_test.txt","r") as f:
            file_labels1 = f.readlines()
        file_labels += file_labels1
        cnt = 0
        for item in file_labels:
            try:
                path = item.split(" ")[0]
                label = int(item.split(" ")[1][0])
                image = Image.open(path)
                image = image.resize((238,158)) # 360*240
                # image = image.resize((100,70))
                images.append(image)
                labels.append(label)
            except:
                cnt += 1
                print("cannot load file {}".format(item))

        images = np.array(images)
        labels = np.array(labels)
        images = self.preprocess_data(images)
        return images, labels

    # 保存分配好的数据到文件夹
    def save_data_to_folders(self, data_splits, num_clients):
        save_dir = self.cfg["base_path"] + '/data/crowd/client_data/{}'.format(str(num_clients))

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
        data_dir = os.path.join(self.cfg["base_path"] + '/data/crowd/client_data/{}/client_{}'.format(str(num_clients), str(client_id)))
        images_file = os.path.join(data_dir, 'images.npy')
        labels_file = os.path.join(data_dir, 'labels.npy')
        images = np.load(images_file)
        labels = np.load(labels_file)
        # images = self.preprocess_data(images)

        return images, labels

    def create_splits(self, images: np.array, labels:np.array):
        num_samples = labels.shape[0]
        samples_per_client = num_samples // self.cfg["num_clients"]

        idx1 = list(range(6800+7200-1))
        idx2 = list(range(6800+7200, num_samples))
        random.shuffle(idx1)
        random.shuffle(idx2)
        idx = idx1 + idx2
        indices = np.array(idx)
        # 使用打乱后的索引重新排序images和labels
        images = images[indices]
        labels = labels[indices]

        # 分配数据给每个客户端
        data_splits = []
        start = 0
        for i in range(self.cfg["num_clients"]):
            end = int(start + samples_per_client)
            client_images = images[start:end]
            client_labels = labels[start:end]
            data_splits.append([client_images, client_labels])
            start = end

        return data_splits


class DataLoader:
    def __init__(self, cfg, dev_id, batch_size = None):
        self.cfg = cfg
        self.data_dir = os.path.join(cfg["base_path"]+r"/data/crowd/client_data/{}".format(str(cfg["num_clients"])),"client_{}".format(str(dev_id)))
        self.batch_size = batch_size
        self.images, self.labels = self.load_data()
        # print(self.labels.shape)
    
    def get_class_num(self):
        return 2

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
    def __init__(self, cfg, batch_size, distribute = False, client_id = 0):
        self.cfg = cfg
        self.batch_size = batch_size
        self.images, self.labels = self.load_data(distribute=distribute, client_id=client_id)

    def get_class_num(self):
        return 2
    
    def load_data(self, distribute = False, client_id = 0):
        if distribute:
            data_dir = os.path.join(self.cfg["base_path"]+r"/data/crowd/client_data/{}".format(str(self.cfg["num_clients"])),"client_{}".format(str(client_id)))
            images_file = os.path.join(data_dir, 'images.npy')
            labels_file = os.path.join(data_dir, 'labels.npy')
            images = np.load(images_file)
            labels = np.load(labels_file)
            images = images[int(images.__len__()*0.75+1):]
            labels = labels[int(labels.__len__()*0.75+1):]
        else:
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
            data_test = datasets.CIFAR10(root = self.cfg["base_path"] + "/data/crowd",
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
        num_batches = int(math.ceil(num_samples / self.batch_size))

        for i in range(num_batches):
            start = i * self.batch_size
            end = min(start + self.batch_size, self.labels.__len__())
            batch_images = self.images[start:end]
            batch_labels = self.labels[start:end]
            # batch_images = self.preprocess_data(batch_images)
            batch_images = torch.from_numpy(batch_images)
            batch_labels = torch.from_numpy(batch_labels)
            
            yield batch_images, batch_labels


if __name__ == "__main__":
    pass
