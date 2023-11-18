import numpy as np
import torch
import random
from opacus import PrivacyEngine

#distribute：pathological / practical, 
def create_non_iid_data_splits(images:np.array, labels:np.array, num_clients:int, distribute = "pra", alpha = 0.1, batch_size=10, class_per_client=0, balance = False):
    '''
    Split training data in non-iid way

    Inputs:
    images: training data
    labels: label data
    num_clients: how many clients will split the data
    distribute：non_iid type
    alpha: parameter of Dirichlet, the smaller alpha, the degree of non-iid higher.

    Returns
    data_splits = [[train_data1, label1],[train_data2, label2]...]
    '''
    data_splits = []

    least_sample = batch_size * 2 
    if distribute == "pathological" or distribute == "pat":
        data_splits = [[] for _ in range(num_clients)]
        unique_classes = np.unique(labels)
        num_class = unique_classes.__len__()
        idxs = np.array(range(len(labels)))
        idxs_each_class = []
        for i in range(num_class):
            idxs_each_class.append(idxs[labels == i])
        class_num_per_client = [class_per_client] * num_clients
        for i in range(num_class):
            selected = []
            for client_idx in range(num_clients):
                if class_num_per_client[client_idx] > 0:
                    selected.append(client_idx)
            selected = selected[:int(np.ceil(num_clients/num_class * class_per_client))] 

            sample_num = len(idxs_each_class[i])
            sample_per_client = sample_num / len(selected) 
            if balance == True:
                num_samples = [int(sample_per_client) for _ in range(len(selected))]
            else:
                num_samples = [int(sample_per_client) for _ in range(len(selected))]
                for idx in range(num_samples.__len__()//2):
                    delta = np.random.randint(-(num_samples[idx] // 1.5), num_samples[idx] // 1.5)
                    num_samples[idx] -= delta
                    num_samples[num_samples.__len__() - idx - 1] += delta

            idx = 0
            for client, num in zip(selected, num_samples):
                if data_splits[client] == []:
                    data_splits[client] = [images[idxs_each_class[i][idx:idx+num]], labels[idxs_each_class[i][idx:idx+num]]]
                else:
                    data_splits[client][0] = np.append(data_splits[client][0], images[idxs_each_class[i][idx:idx+num]], axis=0)
                    data_splits[client][1] = np.append(data_splits[client][1], labels[idxs_each_class[i][idx:idx+num]], axis=0)
                idx += num
                class_num_per_client[client] -= 1

    elif distribute == "practical" or distribute == "pra":
        min_size = 0
        num_samples = labels.shape[0]
        classes = np.unique(labels).__len__()
        unique_classes = np.unique(labels)
        while min_size < least_sample:
            idx_batch = [[] for _ in range(num_clients)]
            for cs in range(len(unique_classes)):
                idx_cs = np.where(labels == cs)[0]
                np.random.shuffle(idx_cs)
                proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
                proportions = np.array([p*(len(idx)<num_samples/num_clients) for p,idx in zip(proportions,idx_batch)])
                proportions = proportions/proportions.sum()
                proportions = (np.cumsum(proportions)*len(idx_cs)).astype(int)[:-1] 
                idx_batch = [idx + idx2.tolist() for idx,idx2 in zip(idx_batch,np.split(idx_cs,proportions))] 
                min_size = min([len(idx_j) for idx_j in idx_batch])
        for client in range(num_clients):
            data_splits.append([images[idx_batch[client]], labels[idx_batch[client]]])
    else:
        raise ValueError("wrong iid value")
    for client in data_splits:
        idx = list(range(client[1].__len__()))
        np.random.shuffle(idx)
        client[0] = client[0][idx]
        client[1] = client[1][idx]
    return data_splits


def create_iid_data_splits(images, labels, num_clients):
    num_samples = labels.shape[0]
    samples_per_client = num_samples / num_clients

    indices = np.random.permutation(num_samples)
    images = images[indices]
    labels = labels[indices]

    data_splits = []
    start = 0
    for i in range(num_clients):
        end = int(start + samples_per_client)
        client_images = images[start:end]
        client_labels = labels[start:end]
        data_splits.append([client_images, client_labels])
        start = end

    return data_splits

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True



MAX_GRAD_NORM = 0.1
DELTA = 1e-5

def initialize_dp(model, optimizer, data_loader, dp_sigma):
    privacy_engine = PrivacyEngine()
    model, optimizer, data_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=data_loader,
        noise_multiplier = dp_sigma, 
        max_grad_norm = MAX_GRAD_NORM,
    )

    return model, optimizer, data_loader, privacy_engine


def get_dp_params(privacy_engine):
    return privacy_engine.get_epsilon(delta=DELTA), DELTA

class VateerDataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.idx = list()
        for item in x:
            self.idx.append(item)
        pass

    def __getitem__(self, index):
        input_data = self.idx[index] 
        target = self.y[index]
        return input_data, target

    def __len__(self):
        return len(self.idx)



def change2torchloader(loader, bz):
    dataset = VateerDataset(loader.preprocess_data(np.array(loader.images[0])), loader.labels)
    return torch.utils.data.DataLoader(dataset, batch_size=1,shuffle=False,drop_last=True) 
