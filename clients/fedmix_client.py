from torchvision import datasets
import torch, os
import numpy as np
from data_handler import *
from clients.base_client import BaseClient
from models import *
import wandb,math

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
        if cfg["opti"] == "sgd":
            self.opt = torch.optim.SGD(self.net.parameters(), lr=self.cfg["lr"])
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.opt, 
            gamma=self.cfg["lr_decay_accumulated"]
        )

    def train(self, lamb, now_epoch):
        self.net.load_state_dict(self.model_para, strict=True)
        self.opt = torch.optim.SGD(self.net.parameters(), lr=self.cfg["lr"] * math.pow(self.cfg["lr_decay_accumulated"], now_epoch))
        for epoch in range(self.epoch):
            total_loss = 0.0
            num_batches = 0 
            for i, (data, label) in enumerate(self.loader.get_batches()):
                make_data = (1-lamb) * data
                make_data.requires_grad_()
                idg = torch.randint(len(self.get_train_mean), (1, ))
                xg = self.get_train_mean[idg:idg+1]
                yg = self.get_label_mean[idg:idg+1]

                data, label = data.to(self.dev), label.to(self.dev)
                make_data=make_data.to(self.dev)
                xg = xg.to(self.dev)
                yg = yg.to(self.dev)

                self.opt.zero_grad()
                
                preds = self.net(make_data)
                loss1 = (1 - lamb) * self.loss_fun(preds, label)
                loss2 = lamb * self.loss_fun(preds, yg.expand_as(preds))

                gradients = torch.autograd.grad(outputs=loss1, inputs=make_data,
                                          create_graph=True, retain_graph=True)[0]
                
                loss3 = lamb * torch.inner(gradients.flatten(start_dim=1), xg.flatten(start_dim=1))
                loss3 = torch.mean(loss3)
                loss = loss1 + loss2 + loss3
                loss.backward()
                self.opt.step()
                total_loss += loss.item()
                num_batches += 1
            
        avg_loss = total_loss / num_batches
        print('Client {}, Average Loss: {:.8f}, loss1:{:.8f}, loss2:{:.8f}, loss3:{:.8f}'.format(self.dev_id, avg_loss, loss1, loss2, loss3))
        if self.cfg["lr_decay_accumulated"] > 0:
            self.learning_rate_scheduler.step()
        return avg_loss
                
    def train_naive(self, lamb):
        self.net.load_state_dict(self.model_para, strict=True)
        self.opt = torch.optim.SGD(self.net.parameters(), lr=self.cfg["lr"])
        for epoch in range(self.epoch):
            total_loss = 0.0
            num_batches = 0 
            for i, (data, label) in enumerate(self.loader.get_batches()):
                idg = torch.randint(len(self.get_train_mean), (1, ))
                xg = self.get_train_mean[idg:idg+1]
                yg = self.get_label_mean[idg:idg+1]
                data = (1 - lamb)*data + lamb * xg
                data, label = data.to(self.dev), label.to(self.dev)
                yg = yg.to(self.dev)
                yg = yg.expand_as(label)
                preds = self.net(data)
                loss1 = (1-lamb)*self.loss_fun(preds, label)
                loss2 = lamb * self.loss_fun(preds, yg)
                loss = loss1 + loss2
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        print('Client {}, Average Loss: {:.8f}, loss1:{:.8f}, loss2:{:.8f}'.format(self.dev_id, avg_loss, loss1, loss2))
        return avg_loss
                
    def calculate_mean_data(self, mean_batch: int):
        data, label = [], []
        for x, y in self.loader.get_batches(bz = 1, process=True):
            data.append(x[0])
            label.append(y[0])

        data = torch.stack(data, dim=0)
        label = torch.stack(label, dim=0)

        random_ids = torch.randperm(len(data))
        data, label = data[random_ids], label[random_ids]
        data = torch.split(data, mean_batch)
        label = torch.split(label, mean_batch)

        self.Xmean, self.ymean = [], []
        for d, l in zip(data, label):
            self.Xmean.append(torch.mean(d, dim=0))
            self.ymean.append(torch.mean(l.to(dtype=torch.float32), dim=0))
        self.Xmean = torch.stack(self.Xmean, dim=0)
        self.ymean = torch.stack(self.ymean, dim=0)
        return self.Xmean, self.ymean

    def get_mean_data(self, Xg, Yg):
        self.get_train_mean = Xg
        self.get_label_mean = Yg
