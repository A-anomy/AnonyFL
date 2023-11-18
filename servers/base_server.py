import torch
class BaseServer(object):
    def __init__(self) -> None:
        pass
    def evaluate(self):
        pass
    def send_model(self, clients:list, client_id:list):

        for idx in client_id:
            clients[idx].set_parameter(self.parameter)
    def receive(self, para, size):
        self.size_cnt += size
        self.rec_list.append((para, size))

    def save(self):
        path = self.cfg["base_path"]+"model_save/save.pkl"
        torch.save(self.net.state_dict(), path)

    def get_model(self):
        return self.net