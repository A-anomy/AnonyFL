from engine.Base import Base
import os
import urllib.request
import numpy as np
from models.CNN import *
import torch, torchvision
from tqdm import tqdm
from clients.bc_fedavg_client import Client
from servers.bc_fedavg_server import Server
from fisco_py.client.datatype_parser import DatatypeParser
from common import *
from data_handler import *
import random, copy, wandb, subprocess
from blockchain.IPFS import IPFS


class BCFedAvg(Base):
    def run(cfg:dict):
        data_handler = eval(cfg['dataset']+"_handler."+'Handler')(cfg)
        images, labels = data_handler.load_data()
        if cfg["iid"] == False: 
            data_splits = create_non_iid_data_splits(images, labels, cfg["num_clients"], batch_size=cfg["batch_size"], alpha = cfg["dataset_alpha"], distribute = cfg["dataset_distribute"], class_per_client=cfg["dataset_class_per_client"], balance=cfg["dataset_balance"])
        else:
            data_splits = create_iid_data_splits(images, labels, cfg["num_clients"])
        data_handler.save_data_to_folders(data_splits, cfg["num_clients"])
        clients = []
        for idx in range(cfg['num_clients']):
            clients.append(Client(cfg["gpu"], idx, cfg["batch_size"], cfg))
        server = Server(cfg)
        accs = []
        os.chdir('./fisco_py')
        from fisco_py.client.bcosclient import BcosClient
        bcos_client = BcosClient()
        print(bcos_client.getBlockNumber())

        abi_file = "./contracts/Aggregrate.abi"
        data_parser = DatatypeParser()
        data_parser.load_abi_file(abi_file)
        contract_abi = data_parser.contract_abi
        command = "python ./vateer_handler.py --address  {} --func set_threshold --args [{}]".format("a"+cfg["address"], int(cfg["num_clients"]*cfg["frac"]))
        p = subprocess.Popen(command,shell=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,encoding='utf-8')
        print(p.communicate()[0])

        ipfs = IPFS()
        for r in range(int(cfg["num_comm"])):
            print("\ncommunicate round {}".format(r+1))
            select_len = int(cfg["num_clients"]*cfg["frac"])
            selected = np.random.choice(np.array(range(cfg["num_clients"])), size=select_len, replace=False)
            server.send_model(clients, selected)
            avg_loss = 0.0
            for client in [clients[idx] for idx in selected]:
                avg_loss += client.train(r + 1)
                client.sent_parameter(bcos_client, ipfs, r)
            avg_loss /= len(selected)
            print("Total Avg Loss: {}".format(str(avg_loss)))
            server.aggregate(bcos_client, ipfs)
            accs.append(server.evaluate())
            if cfg["wandb"] == 1:
                wandb.log({"loss":avg_loss,"acc":accs[-1]})
        for idx in range(len(accs)):
            print("round{}: {}".format(idx, accs[idx]))
