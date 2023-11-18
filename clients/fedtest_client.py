from torchvision import datasets
import torch, os, random
import numpy as np
from data_handler import *
from clients.base_client import BaseClient
from models import *
import math

class Client(BaseClient):
    def __init__(self, dev, dev_id, batch_size, cfg:dict):
        net = None
        net_name = cfg["dataset"]
        net_name = net_name[0].upper() + net_name[1:]
        net_name=cfg["net"]+"."+net_name+"_"+cfg["net"]
        net = eval(net_name)()
        self.cfg = cfg
        super().__init__(dev_id = dev_id, model_para = None, model_net = net, batch_size = batch_size, dev = dev)
        self.loader = None
        self.dev = cfg["gpu"]
        self.net = self.net.to(self.dev)
        self.epoch = cfg["epoch"]
        self.loader = eval(cfg["dataset"] + "_handler").DataLoader(cfg, dev_id ,batch_size)
        if cfg["loss_func"] == "cross_entropy":
            self.loss_fun = torch.nn.functional.cross_entropy
    
    def report_distribute(self, server):
        self.local_distribute = [0]*self.loader.get_class_num()
        images, labels = [], []
        for image, label in self.loader.get_batches(bz = 1, process=False):
            label = torch.argmax(label, dim=1)
            self.local_distribute[label] += 1
            images.append(image), labels.append(label[0])
        sorted_indices = np.argsort(labels)
        images = np.concatenate(images, axis=0)[sorted_indices]
        labels = np.array(labels)[sorted_indices]
        idx = 0
        upload_images, upload_label = [], []
        while idx + self.cfg["mean_batch"] < len(images):
            if labels[idx] == labels[idx + self.cfg["mean_batch"]]:
                upload_images.append(np.mean(images[idx: idx + self.cfg["mean_batch"]], axis=0))
                upload_label.append(labels[idx])
            idx += self.cfg["mean_batch"]
        server.receive_distribute(self.local_distribute, upload_images, upload_label)
        
    def balance_distribute(self, server):
        self.download_images, self.download_labels, self.global_distribute = server.balance_distribute(self.local_distribute)
        self.download_images = self.download_images.astype(np.float32)        
        self.idxs = [[]] * self.loader.get_class_num()
        self.add_class_num = 0
        for i, idx in enumerate(self.idxs):
            idx = np.where(self.download_labels == i)[0]
            if idx.__len__() > 0:
                self.add_class_num += 1
        if self.cfg["naive"] == 1:
            self.loader.add_data(self.download_images, self.download_labels)


    def train(self, now_epoch = 0):
        self.net.load_state_dict(self.model_para, strict=True)
        self.net.train()
        avg_loss = 0.0
        self.opt = torch.optim.SGD(self.net.parameters(), lr=self.cfg["lr"] * math.pow(self.cfg["lr_decay_accumulated"], now_epoch))
        if self.cfg["naive"] == 1: 
            for epoch in range(self.epoch):
                total_loss = 0.0
                num_batches = 0 
                for i, (data, label) in enumerate(self.loader.get_batches()):
                    data, label = data.to(self.dev), label.to(self.dev)
                    preds = self.net(data)
                    loss = self.loss_fun(preds, label)
                    self.opt.zero_grad()
                    loss.backward()
                    self.opt.step()
                    total_loss += loss.item()
                    num_batches += 1
            avg_loss = total_loss / num_batches
            print('Client {}, Average Loss: {:.8f}'.format(self.dev_id, avg_loss))
        elif self.cfg["naive"] == 2: 
            lamb = self.cfg["lamb"]
            for epoch in range(self.epoch):
                total_loss = 0.0
                num_batches = 0 
                for i, (data, label) in enumerate(self.loader.get_batches()):
                    idx = np.random.choice(np.array(range(len(self.download_images))), size = label.shape[0], replace = False)
                    xg, yg = self.download_images[idx], self.download_labels[idx]
                    xg, yg = torch.tensor(xg), torch.tensor(yg)
                    data = (1 - lamb)*data + lamb * xg
                    yg = yg.to(self.dev)
                    data, label = data.to(self.dev), label.to(self.dev)
                    preds = self.net(data)
                    loss1 = (1-lamb)*self.loss_fun(preds, label)
                    loss2 = lamb*self.loss_fun(preds, yg)
                    loss = loss1 + loss2
                    self.opt.zero_grad()
                    loss.backward()
                    self.opt.step()
                    total_loss += loss.item()
                    num_batches += 1
            avg_loss = total_loss / num_batches
            print('Client {}, Average Loss: {:.8f}'.format(self.dev_id, avg_loss))   
        elif self.cfg["naive"] == 3:
            idx = np.array(range(len(self.download_images)))
            np.random.shuffle(idx)
            self.download_images = self.download_images[idx]
            self.download_labels = self.download_labels[idx]
            lamb = self.cfg["lamb"]
            a2x = max(1, self.download_labels.__len__() // self.loader.__len__())
            a2x *= self.cfg["batch_size"]
            flag = 1
            for epoch in range(self.epoch):
                total_loss = 0.0
                num_batches = 0.0
                
                for i, (data, label) in enumerate(self.loader.get_batches()):
                    
                    data, label = data.to(self.dev), label.to(self.dev)
                    preds = self.net(data)
                    loss = self.loss_fun(preds, label)
                    self.opt.zero_grad()
                    loss.backward()
                    self.opt.step()
                    total_loss += loss.item()
                    num_batches += 1
                    
                    delta = 0.0
                    if now_epoch <= self.cfg["dead_lamb2"]:
                        if now_epoch % 30 == 0:
                            self.cfg["lamb2"] = self.cfg["lamb2"] * self.cfg["decay_lamb2"]
                            flag = 0
                        idx = random.sample(range(len(self.download_images)), a2x)
                        
                        data, label = self.download_images[idx], self.download_labels[idx]
                        label = torch.nn.functional.one_hot(torch.tensor(label).to(torch.int64), num_classes=self.loader.get_class_num()).to(dtype=torch.float32)
                        data = torch.from_numpy(data)
                        data = data.to(self.dev)
                        label = label.to(self.dev)
                        preds = self.net(data)
                        loss = self.cfg["lamb2"] * self.loss_fun(preds, label)
                        self.opt.zero_grad()
                        loss.backward()
                        self.opt.step()
                        total_loss += loss.item()
                        delta = self.cfg["lamb2"]

                    num_batches += + delta
            avg_loss = total_loss / num_batches
            print('Client {}, Average Loss: {:.8f}'.format(self.dev_id, avg_loss))
        elif self.cfg["naive"] == 4: 
            idx = np.array(range(len(self.download_images)))
            np.random.shuffle(idx)
            self.download_images = self.download_images[idx]
            self.download_labels = self.download_labels[idx]
            lamb = self.cfg["lamb"]
            a2x = max(1, self.download_labels.__len__() // self.loader.__len__())
            a2x *= self.cfg["batch_size"]
            lamb2 = self.cal_lamb2(now_epoch)
            flag = 0
            for epoch in range(self.epoch):
                total_loss = 0.0
                num_batches = 0.0              
                for i, (data, label) in enumerate(self.loader.get_batches()):
                    data, label = data.to(self.dev), label.to(self.dev)
                    preds = self.net(data)
                    loss = self.loss_fun(preds, label)
                    self.opt.zero_grad()
                    loss.backward()
                    self.opt.step()
                    total_loss += loss.item()
                    num_batches += 1
                    delta = 0.0
                    idx = random.sample(range(len(self.download_images)), a2x)
                    
                    data, label = self.download_images[idx], self.download_labels[idx]
                    label = torch.nn.functional.one_hot(torch.tensor(label).to(torch.int64), num_classes=self.loader.get_class_num()).to(dtype=torch.float32)
                    data = torch.from_numpy(data)
                    data = data.to(self.dev)
                    label = label.to(self.dev)
                    preds = self.net(data)
                    loss = lamb2 * self.loss_fun(preds, label)
                    self.opt.zero_grad()
                    loss.backward()
                    self.opt.step()
                    total_loss += loss.item()

                    num_batches += lamb2
                    flag = 1
            avg_loss = total_loss / num_batches
            print('Client {}, Average Loss: {:.8f}'.format(self.dev_id, avg_loss))
            if flag == 1:
                print("lamb2 "+str(lamb2))
        elif self.cfg["naive"] == 5:
            idx = np.array(range(len(self.download_images)))
            np.random.shuffle(idx)
            self.download_images = self.download_images[idx]
            self.download_labels = self.download_labels[idx]
            lamb = self.cfg["lamb"]
            a2x = self.cfg["batch_size"]
            lamb2 = self.cal_lamb2(now_epoch)
            flag = 0
            for epoch in range(self.epoch):
                total_loss = 0.0
                num_batches = 0.0              
                for i, (data, label) in enumerate(self.loader.get_batches()):
                    data, label = data.to(self.dev), label.to(self.dev)
                    preds = self.net(data)
                    loss = self.loss_fun(preds, label)
                    self.opt.zero_grad()
                    loss.backward()
                    self.opt.step()
                    total_loss += loss.item()
                    num_batches += 1
                    delta = 0.0
                    if now_epoch <= self.cfg["dead_lamb2"]:
                        idx = random.sample(range(len(self.download_images)), a2x)
                        
                        data, label = self.download_images[idx], self.download_labels[idx]
                        label = torch.nn.functional.one_hot(torch.tensor(label).to(torch.int64), num_classes=self.loader.get_class_num()).to(dtype=torch.float32)
                        data = torch.from_numpy(data)
                        data = data.to(self.dev)
                        label = label.to(self.dev)
                        preds = self.net(data)
                        loss = lamb2 * self.loss_fun(preds, label)
                        self.opt.zero_grad()
                        loss.backward()
                        self.opt.step()
                        total_loss += loss.item()
                        flag = 1 

            avg_loss = total_loss / num_batches
            print('Client {}, Average Loss: {:.8f}'.format(self.dev_id, avg_loss))
            if flag == 1:
                print("lamb2 "+str(lamb2))
        return avg_loss
    def cal_lamb2(self, now_epoch):
        lamb2 = self.cfg["lamb2"]
        if self.cfg["decay_lamb2"] <= 1.0:
            lamb2_delta = now_epoch // self.cfg["interval"]
        elif self.cfg["decay_lamb2"] > 1:
            if now_epoch > self.cfg["decay_lamb2"]:
                lamb2_delta = (now_epoch - self.cfg["decay_lamb2"]) // self.cfg["interval"]
            else:
                lamb2_delta = 0
        while(lamb2_delta > 0):
            lamb2_delta -= 1
            lamb2 *= self.cfg["decay_lamb2"]
        return lamb2
