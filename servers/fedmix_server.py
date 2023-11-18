from servers.base_server import BaseServer
from models import *
from data_handler import *
import torch
import sys

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
        self.X_g, self.Y_g = [], []
        self.bandwith = 0.0

    def aggregate(self):
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
    
    def get_mean_data(self, x_mean, y_mean):
        self.X_g.append(x_mean)
        self.Y_g.append(y_mean)
        self.bandwith += sys.getsizeof(x_mean.tolist()) + sys.getsizeof(y_mean.tolist())

    def generate_mean_data(self):
        self.X_g = torch.cat(self.X_g, dim=0)
        self.Y_g = torch.cat(self.Y_g, dim=0)
    
    def send_mean_data(self, client):
        client.get_mean_data(self.X_g, self.Y_g)

    def get_bandwith(self):
        return self.bandwith
        
    
    