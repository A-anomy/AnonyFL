from engine.Base import Base
import os, sys
import urllib.request
import numpy as np
from models.CNN import *
import torch, torchvision
from tqdm import tqdm
from common import *
from data_handler import *
import random, copy, wandb
from clients.fedmix_client import Client
from servers.fedmix_server import Server


class FedMix(Base):
    def run(cfg:dict):
        data_handler = eval(cfg['dataset']+"_handler."+'Handler')(cfg)
        images, labels = data_handler.load_data()
        if cfg["iid"] == False: 
            data_splits = create_non_iid_data_splits(images, labels, cfg["num_clients"], batch_size=cfg["batch_size"], alpha = cfg["dataset_alpha"], distribute = cfg["dataset_distribute"], class_per_client=cfg["dataset_class_per_client"], balance=cfg["dataset_balance"])
        else:
            data_splits = create_iid_data_splits(images, labels, cfg["num_clients"])
        data_handler.save_data_to_folders(data_splits, cfg["num_clients"])
        server = Server(cfg)
        clients = []
        for idx in range(cfg['num_clients']):
            clients.append(Client(cfg["gpu"], idx, cfg["batch_size"], cfg))
            x_mean, y_mean = clients[-1].calculate_mean_data(mean_batch=cfg["mean_batch"])
            server.get_mean_data(x_mean, y_mean)
        server.generate_mean_data()
        for client in clients:
            server.send_mean_data(client)   
        server.bandwith += (sys.getsizeof(server.X_g.tolist()) + sys.getsizeof(server.Y_g.tolist())) * len(clients)
        print("All bandwith cost {} Bytes".format(server.get_bandwith()))
        accs = []    
        for r in range(int(cfg["num_comm"])):
            print("\ncommunicate round {}".format(r+1))
            select_len = int(cfg["num_clients"]*cfg["frac"])
            selected = np.random.choice(np.array(range(cfg["num_clients"])), size=select_len, replace=False)
            server.send_model(clients, selected)
            avg_loss = 0.0
            for client in [clients[idx] for idx in selected]:
                avg_loss += client.train(cfg["lamb"],r + 1)
                client.sent_parameter(server)
            avg_loss /= len(selected)
            print("Total Avg Loss: {}".format(str(avg_loss)))             
            server.aggregate()
            accs.append(server.evaluate())
            if cfg["wandb"] == 1:
                wandb.log({"loss":avg_loss,"acc":accs[-1]})   
        for idx in range(len(accs)):
            print("round{}: {}".format(idx, accs[idx]))

            


        

        

