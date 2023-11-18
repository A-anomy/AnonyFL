from torchvision import datasets
import torch, os
import numpy as np
from data_handler import *
from clients.base_client import BaseClient
from models import *
import wandb, math
from common import initialize_dp, get_dp_params, change2torchloader

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
        if self.cfg["privacy"] == True:
            self.loader = change2torchloader(self.loader, 9)

    def train(self, now_epoch):
        self.net.load_state_dict(self.model_para, strict=True)
        self.net.train()
        self.opt = torch.optim.SGD(self.net.parameters(), lr=self.cfg["lr"] * math.pow(self.cfg["lr_decay_accumulated"], now_epoch))
        if self.cfg["privacy"] == True:
            self.net, self.opt, self.loader, privacy_engine = \
                initialize_dp(self.net, self.opt, self.loader, self.cfg["dp_sigma"])
        for epoch in range(self.epoch):
            total_loss = 0.0
            num_batches = 0 
            for i, (data, label) in enumerate(self.loader.get_batches()):
                # print(data.shape)
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
        if self.cfg["privacy"]:
            eps, DELTA = get_dp_params(privacy_engine)
            print(f"Client {self.dev_id}", f"epsilon = {eps:.2f}, sigma = {DELTA}")
        return avg_loss