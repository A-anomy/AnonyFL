from torchvision import datasets
import torch, os
import numpy as np
from data_handler import *
from clients.base_client import BaseClient
from models import *
import wandb,subprocess,math
from fisco_py.client.datatype_parser import DatatypeParser

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
    def train(self):
        self.net.load_state_dict(self.model_para, strict=True)
        self.net.train()
        self.opt = torch.optim.SGD(self.net.parameters(), lr=self.cfg["lr"] * math.pow(self.cfg["lr_decay_accumulated"], now_epoch))
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
        if self.cfg["lr_decay_accumulated"] > 0:
            self.learning_rate_scheduler.step()
        return avg_loss
    def sent_parameter(self, bcos_client, ipfs, epoch):
        path = self.cfg["base_path"]+"cache/{}_upload.pkl".format(str(self.dev_id))
        torch.save(self.net.state_dict(), path)

        console_path = self.cfg["base_path"]+"/fisco_py/console2.py"
        account_path = self.cfg["base_path"]+"/fisco_py/bin/accounts/client{}_{}.keystore".format(str(self.dev_id), str(epoch))

        if not os.path.exists(account_path):
            command=f"python {console_path} " + "newaccount client{}_{}".format(str(self.dev_id), str(epoch)) + " 123456"

            p = subprocess.Popen(command,shell=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,encoding='utf-8')

            _ = p.communicate()[0]
        
        name = "client{}_{}".format(str(self.dev_id), str(epoch))
        res = ipfs.push_local_file(path)

        abi_file = "./contracts/Aggregrate.abi"
        data_parser = DatatypeParser()
        data_parser.load_abi_file(abi_file)
        contract_abi = data_parser.contract_abi
        command = 'python ./vateer_handler.py --address {} --func send_gradient --account_name {} --args \\[{},{},{}\\]'.format("a"+self.cfg["address"], name, res , self.loader.__len__(), 
        "a0x123")
        p =  subprocess.Popen(command,shell=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,encoding='utf-8')
