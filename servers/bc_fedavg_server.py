from servers.base_server import BaseServer
from models import *
from data_handler import *
from fisco_py.client.datatype_parser import DatatypeParser
import torch, wandb, subprocess, time
class Server(BaseServer):
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        net_name = cfg["dataset"]
        net_name = net_name[0].upper() + net_name[1:]
        net_name=cfg["net"]+"."+net_name+"_"+cfg["net"]
        self.net = eval(net_name)()
        self.loader = eval(cfg["dataset"] + "_handler").TestLoader(cfg, 100)
        self.dev = cfg["gpu"]
        self.net = self.net.to(self.dev)
        self.parameter={}
        for key, var in self.net.state_dict().items():
            self.parameter[key] = var.clone()
        self.size_cnt = 0
        self.rec_list = []
    def send_model(self, clients:list, client_id:list):
        for idx in client_id:
            clients[idx].set_parameter(self.parameter)
    def receive(self, para, size):
        self.size_cnt += size
        self.rec_list.append((para, size))
    
    def receive_from_blockchain(self, bcos_client, ipfs):
        res = 'nul'
        while res == 'nul':
            command = "python ./vateer_handler.py --address {} --func get_gradient".format("a"+self.cfg["address"])
            p = subprocess.Popen(command,shell=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,encoding='utf-8')
            res = p.communicate()[0][:-2]
            if res == 'nul':
                print("cannot pull from blockchain, wait 1s ...")
                time.sleep(1)
        hex_list = res.split("|")

        idx = 0
        while idx < hex_list.__len__():
            hex = hex_list[idx]
            path = "../cache/down_model{}.pkl".format(str(idx))
            ipfs.download_loacl_file(hex, path)
            net = torch.load(path)
            idx += 1
            self.receive(net, int(hex_list[idx]))
            idx += 2

    def aggregate(self, bcos_client, ipfs):
        self.receive_from_blockchain(bcos_client, ipfs)
        self.parameter=None
        for para, size in self.rec_list:
            if self.parameter == None:
                self.parameter = {}
                for key, var in para.items():
                    self.parameter[key] = size/self.size_cnt * var.clone()
            else:
                for key, var in para.items():
                    self.parameter[key] += size/self.size_cnt * var.clone()
        self.size_cnt = 0
        self.rec_list = []
        self.net.load_state_dict(self.parameter, strict=True)

    def evaluate(self):
        self.net.eval()
        sum_accu = 0
        num = 0
        for data, label in self.loader.get_batches():
            data, label = data.to(self.dev), label.to(self.dev)
            preds = self.net(data)
            preds = torch.argmax(preds, dim=1)
            sum_accu += (preds == label).float().mean()
            num += 1      
        print('accuracy: {}'.format(sum_accu / num))
        return sum_accu / num
        

        
    
    