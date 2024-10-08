# import os

import sys
import flwr as fl
import GPUtil
from time import sleep
from pathlib import Path
import shutil
import glob
import argparse
import os
from collections import OrderedDict
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import neptune
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
parser = argparse.ArgumentParser()
parser.add_argument('--num_clients', type=int, default=3, required=False)
parser.add_argument('--num_rounds', type=int, default=10, required=False)
parser.add_argument('--participation_rate', type=float, default=1.0, required=False)
parser.add_argument('--batch_size', type=int, default=128, required=False)
parser.add_argument('--train_epochs', type=int, default=1, required=False)
parser.add_argument('--lr', type=float, default=0.3, required=False)
parser.add_argument('--momentum', type=float, default=0.9, required=False)
parser.add_argument('--knn_neighbors', type=int, default=10, required=False)
parser.add_argument('--noisy_frac', type=float, default=0.8, required=False)
parser.add_argument('--noise_level', type=float, default=0.4, required=False)
parser.add_argument('--noise_sparsity', type=float, default=0.7, required=False)
parser.add_argument('--distil_round', type=int, default=1, required=False)
parser.add_argument('--embeddings_dir', type=str, default='./data/features', required=False)
parser.add_argument('--embeddings_dims', type=int, default=512, required=False)
parser.add_argument('--dataset_name', type=str, default='eurosat', required=False)
parser.add_argument('--model_name', type=str, default='resnet20', required=False)
parser.add_argument('--temp_dir', type=str, default='./tmp', required=False)
parser.add_argument('--seed', type=int, default=42, required=False)
args = parser.parse_args()

if Path(args.temp_dir).exists() and Path(args.temp_dir).is_dir():
    shutil.rmtree(Path(args.temp_dir))

def load_available_datasets(train=True):
    import data
    return {
        # 'eurosat': data.load_eurosat if train else data.load_eurosat_test, 
        'cifar10': data.load_cifar if train else data.load_cifar_test,
    }

def load_available_models():
    import models
    return {
        'resnet20': models.load_resnet20_model,
        # 'cnn': models.load_cnn_model,
    }

def grab_gpu(memory_limit=0.91):
	return "0"
    # print("Grabbing gpu")
    # while len(GPUtil.getAvailable(order='memory', limit=len(GPUtil.getGPUs()), maxLoad=1.0, maxMemory=memory_limit)) == 0:
    #     sleep(1)
    # print("GPU grabbed")
    # cuda_device_ids = GPUtil.getAvailable(order='memory', limit=len(GPUtil.getGPUs()), maxLoad=1.0, maxMemory=memory_limit)
    # cuda_device_ids.extend("")  # Fix no gpu issue
    # return str(cuda_device_ids[0])

def create_client(cid):
    sleep(int(cid) * 0.75)
    os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = grab_gpu()
    sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    from utils.flwr_client import _Client as Client
    from utils.knn_relabel import estimate_noise_with_pretrained_knn
    from utils.distiller import Distiller
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    #############################################
    # Override Client to use Noise-Aware FedAvg #
    #############################################
    class AKDClient(Client):
        def __init__(self, dataset, distil_round, num_neighbors, temp_dir, *args, **kwargs):
            super(AKDClient, self).__init__(*args, **kwargs)
            self.distil_round = distil_round
            self.dataset = dataset
            self.num_neighbors = num_neighbors
            self.data_loader = kwargs['data_loader']
            self.num_clients = int(kwargs['num_clients'])
            self.embeddings_params = None
            self.temp_dir = temp_dir
            self.device = kwargs['device']
            print("client created")

        @property
        def noisy(self):
            return os.path.isfile(f'{self.temp_dir}/noise_{self.cid}.npy')

        def set_parameters(self, parameters, config):
            if not hasattr(self, 'model'):
                self.model = self.model_loader(input_shape=self.input_shape[1:], num_classes=self.num_classes, embeddings_dim=int(args.embeddings_dims)) # TODO: removed if self.noisy flag
            # self.optimizer = optim.Adam(self.model.parameters(), lr=config['lr'])
            self.optimizer = optim.SGD(self.model.parameters(), lr=config['lr'], momentum=config['momentum'])
            self.criterion = nn.CrossEntropyLoss()

            if parameters is not None:
                # if self.noisy:
                #     parameters.extend(self.model.get_weights()[-2:] if self.embeddings_params is None else self.embeddings_params)
                #TODO: implement above, combine state dict of current model and embedding layer separately
                # assuming that self.noisy is always set to be true for training, we always use the embedding layer
                # this seems to be a flag to prevent edge case (at initial loading of parameters maybe than a part of method
                params_dict = zip(self.model.state_dict().keys(), parameters)
                state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
                self.model.load_state_dict(state_dict, strict=True)
   
        def fit(self, parameters, config):
            if int(config['round']) == self.distil_round:
                _data, _, _, _labels, _idxs = self.data_loader(shard_id=int(self.cid), num_shards=self.num_clients, shuffle=False)
                # _data = _data.unbatch()
                # all_data = []
                # for batch in _data:
                #     images, labels = batch
                #     all_data.append((images, labels))
                # all_data = torch.cat([d[0] for d in all_data], dim=0)
                # breakpoint()
                with open(f'{args.embeddings_dir}/{self.dataset}.npy', 'rb') as f:
                    _features = np.squeeze(np.load(f)[_idxs])
                estimated_noise = estimate_noise_with_pretrained_knn(labels=_labels, features=_features, num_classes=self.num_classes, num_neighbors=self.num_neighbors, verbose=False)
                if not os.path.isdir(f"{args.temp_dir}"):
                    os.makedirs(f"{args.temp_dir}")
                if estimated_noise > 0.0:
                    with open(f'{args.temp_dir}/noise_{self.cid}.npy', 'wb') as f:
                        np.save(f, np.array([estimated_noise]))

            self.set_parameters(parameters, config)
   
            # if self.noisy:
            _data, _, _, _, _idxs = self.data_loader(shard_id=int(self.cid), num_shards=self.num_clients, shuffle=False)
            all_images, all_labels = [], []
            for batch in _data:
                images, labels = batch
                all_images.append(images)
                all_labels.append(labels)
            
            all_images = torch.cat(all_images, dim=0)
            all_labels = torch.cat(all_labels, dim=0)
            with open(f'{args.embeddings_dir}/{self.dataset}.npy', 'rb') as f:
                _features = np.squeeze(np.load(f)[_idxs])
            self.model = Distiller(model=self.model, features=_features, idxs=np.squeeze(_idxs), optimizer=self.optimizer)
            self.embeddings_dim = 512
            self.set_parameters(parameters=None, config=config)
            self.data = DataLoader(TensorDataset(all_images, all_labels, torch.tensor(_idxs)), batch_size=128, shuffle=True)

            self.model.train()
            self.model = self.model.to(self.device)
            for epoch in range(config['epochs']):
                for batch in self.data:
                    # images, labels = batch
                    # images, labels = images.to(self.device), labels.to(self.device)
                    # self.optimizer.zero_grad()
                    # outputs = self.model(images)
                    # loss = self.criterion(outputs, labels)
                    # loss.backward()
                    # self.optimizer.step()
                    assert len(batch) == 3, f"{len(batch)}, {batch[0].shape}"
                    if batch[0][0].shape == 1:
                        print("single batch detected, continue---------")
                        continue
                    h = self.model(batch)
                    student_loss = h['student_loss']
                    distillation_loss = h['distillation_loss']
                
                    
            #TODO: implement below line in a better way
            # self.embeddings_params = self.model.fc
            # metrics = {'accuracy': float(h.history['accuracy'][-1]), 'loss': float(h.history['loss'][-1])}
            metrics = {
                'accuracy': self.evaluate_accuracy(),
                # 'student_loss': student_loss,
                # 'distillation_loss': distillation_loss
            }
            return self.get_parameters(), self.num_samples, metrics
    #############################################

    load_model = load_available_models()[args.model_name]
    load_train_data = load_available_datasets()[args.dataset_name]
    kwargs = {'batch_size': int(args.batch_size), 'seed': int(args.seed), 'noisy_clients_frac': float(args.noisy_frac),
              'noise_lvl': float(args.noise_level), 'noise_sparsity': float(args.noise_sparsity), 'device': "cuda:" + os.environ['CUDA_VISIBLE_DEVICES'], 'lr': float(args.lr), 'momentum': float(args.momentum)}
    return AKDClient(dataset=args.dataset_name, distil_round=int(args.distil_round), num_neighbors=int(args.knn_neighbors),
                     temp_dir=args.temp_dir, cid=cid, num_clients=int(args.num_clients), model_loader=load_model, data_loader=load_train_data,
                     shuffle=False, **kwargs)

def create_server():
    run = neptune.init_run(
                        project="mtrip1056/fednl2",
                        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI1NDA5OTRhYi0zYjkzLTQ4NTEtOTAwYS1hNzZhYjQ2ZjBlN2UifQ==",
                        name = f"FedLN-AKD",
                    )  
    run['parameters'] = args
    run['parameters/model_name'] = "resnet18"
    from utils.flwr_server import _Server as Server
    load_model = load_available_models()[args.model_name]
    load_test_data = load_available_datasets(train=False)[args.dataset_name]
    kwargs = {'lr': float(args.lr), 'train_epochs': int(args.train_epochs), 'momentum': float(args.momentum)}
    return Server(num_rounds=int(args.num_rounds), num_clients=int(args.num_clients), participation=float(args.participation_rate),
                  model_loader=load_model, data_loader=load_test_data, run=run, **kwargs)

def run_simulation():
    server = create_server()
    print("Created Server, starting simulation")
    history = fl.simulation.start_simulation(client_fn=create_client, server=server, num_clients=int(args.num_clients),
                                             ray_init_args={"ignore_reinit_error": True, "num_cpus": int(args.num_clients), "_temp_dir": "/mnt/Enterprise2/ray_logs" },
                                             config=fl.server.ServerConfig(num_rounds=int(args.num_rounds), round_timeout=None), )
    if Path(args.temp_dir).exists() and Path(args.temp_dir).is_dir():
        shutil.rmtree(Path(args.temp_dir))
    return history

if __name__ == "__main__":
    print(run_simulation())
