import copy, torch
from data_handler import *
class BaseClient(object):
    def __init__(self, dev_id = None, cfg=None, model_para = None, model_net = None, batch_size = None, dev = None) -> None:
        if cfg:
            self.cfg = cfg
        if model_para:
            self.model_para = model_para
        else:
            self.model_para = {}
        self.net = model_net
        self.dev_id = dev_id
        self.batch_size = batch_size
        self.dev = dev
        if "test_method" in self.cfg:
            if self.cfg["test_method"] == "distribution":
                self.test_loader = eval(self.cfg["dataset"] + "_handler").TestLoader(self.cfg, 100, distribute = True, client_id = dev_id)

    def get_train_num(self):
        return self.loader.__len__()

    def sent_parameter(self, server):
        server.receive(self.net.state_dict(), self.loader.__len__())
        
    def get_parameter(self):
        return self.net.state_dict()
    
    def set_parameter(self, model_para):
        for key, val in model_para.items():
            self.model_para[key] = val.clone()

    def evaluate(self):
        
        self.net.eval()
        sum_accu = 0
        num = 0
        for data, label in self.test_loader.get_batches():
            data, label = data.to(self.dev), label.to(self.dev)
            preds = self.net(data)
            preds = torch.argmax(preds, dim=1)
            sum_accu += (preds == label).float().mean().item()
            num += 1
        return sum_accu / num
    