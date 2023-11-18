from servers.base_server import BaseServer
from models import *
from data_handler import *
import torch, random, sys
import numpy as np
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
        self.distribute_list = [0]*self.loader.get_class_num()
        self.rec_images = [[]]*self.loader.get_class_num()
        self.bandwith = 0.0
    def send_model(self, clients:list, client_id:list):
        for idx in client_id:
            clients[idx].set_parameter(self.parameter)
    def receive(self, para, size):
        self.size_cnt += size
        self.rec_list.append((para, size))
    
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

    def receive_distribute(self, distribute:list, images:list, labels:list):
        for idx, val in enumerate(distribute):
            self.distribute_list[idx] += val
        for idx in range(len(images)):
            self.rec_images[labels[idx]].append(images[idx])
        self.bandwith += sys.getsizeof(distribute) + sys.getsizeof(images) + sys.getsizeof(labels)

    
    def balance_distribute(self, distribute:list):
        temp_ls = [d2/d1 for d1, d2 in zip(self.distribute_list, distribute)]
        mx_val = max(temp_ls)
        mx_idx = temp_ls.index(mx_val)
        ret_images, ret_labels = [], []
        for idx, rec_image in enumerate(self.rec_images):
            delta = int(self.distribute_list[idx] * mx_val - distribute[idx])
            if delta > 0:
                ret_images = ret_images + random.sample(rec_image, delta)
                ret_labels = ret_labels + [idx] * delta
        self.bandwith += sys.getsizeof(ret_images) + sys.getsizeof(ret_labels) + sys.getsizeof(distribute) + sys.getsizeof(self.distribute_list)
        return np.array(ret_images), np.array(ret_labels), self.distribute_list
    
    def get_bandwith(self):
        return self.bandwith
    
        

        
        

        
    
    