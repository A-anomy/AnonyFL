from engine.Base import Base
import os
import urllib.request
import numpy as np
from models.CNN import *
import torch, torchvision
from tqdm import tqdm
from clients.fedavg_client import Client
from servers.fedavg_server import Server
from common import *
from data_handler import *
import random, copy, wandb

class FedAvg(Base):
    def run(cfg:dict):
        data_handler = eval(cfg['dataset']+"_handler."+'Handler')(cfg)
        # data_handler.download()
        images, labels = data_handler.load_data()
        if "customer" in cfg and cfg["customer"] == 1:
            data_splits = data_handler.create_splits(images, labels)
        else:
            if cfg["iid"] == False: 
                data_splits = create_non_iid_data_splits(images, labels, cfg["num_clients"], batch_size=cfg["batch_size"], alpha = cfg["dataset_alpha"], distribute = cfg["dataset_distribute"], class_per_client=cfg["dataset_class_per_client"], balance=cfg["dataset_balance"])
            else:
                data_splits = create_iid_data_splits(images, labels, cfg["num_clients"])
        data_handler.save_data_to_folders(data_splits, cfg["num_clients"])
        clients = []
        server = Server(cfg)
        for idx in range(cfg['num_clients']):
            clients.append(Client(cfg["gpu"], idx, cfg["batch_size"], cfg))
            if "memory_save" in cfg and cfg["memory_save"] == True:
                clients[-1].set_net(server.get_model())
        accs = []
        for r in range(int(cfg["num_comm"])):
            print("\ncommunicate round {}".format(r+1))
            select_len = int(cfg["num_clients"]*cfg["frac"])
            selected = np.random.choice(np.array(range(cfg["num_clients"])), size=select_len, replace=False)
            server.send_model(clients, selected)
            avg_loss = 0.0
            for client in [clients[idx] for idx in selected]:
                avg_loss += client.train(r + 1)
                client.sent_parameter(server)
            avg_loss /= len(selected)
            print("Total Avg Loss: {}".format(str(avg_loss)))
            server.aggregate()
            if "test_method" in cfg and cfg["test_method"] == "distribution":
                acc = 0.0
                lss = selected
                if "temp" in cfg and cfg["temp"] == True:
                    lss = range(clients.__len__())
                for idx in lss:
                    acc += clients[idx].evaluate()
                accs.append(acc/lss.__len__()) 
            else:
                accs.append(server.evaluate())
            print("average acc : {}".format(accs[-1]))
            if cfg["wandb"] == 1:
                wandb.log({"loss":avg_loss,"acc":accs[-1]})
        for idx in range(len(accs)):
            print("round{}: {}".format(idx, accs[idx]))
