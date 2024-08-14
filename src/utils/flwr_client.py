# import flwr as fl
# import tensorflow as tf

# class _Client(fl.client.NumPyClient):

# 	def __init__(self, cid, num_clients, model_loader, data_loader, shuffle=True, load_data=True, **kwargs):
# 		self.cid = cid
# 		self.shuffle = shuffle
# 		if load_data:
# 			self.data, self.num_classes, self.num_samples, self.labels, self.idxs = data_loader(shard_id=int(cid), num_shards=num_clients, shuffle=self.shuffle,
# 				batch_size=kwargs['batch_size'], seed=kwargs['batch_size'], noisy_clients_frac=kwargs['noisy_clients_frac'], noise_lvl=kwargs['noise_lvl'], noise_sparsity=kwargs['noise_sparsity'])
# 			if shuffle: del self.labels # labels order is not preserved for shuffle==true
# 		self.model_loader = model_loader
# 		if load_data:
# 			self.input_shape = self.data.element_spec[0].shape

# 	def set_parameters(self, parameters, config):
# 		""" Set model weights """
# 		if not hasattr(self, 'model'):
# 			self.model = self.model_loader(input_shape=self.input_shape[1:], num_classes=self.num_classes)

# 		self.model.compile(
# 			optimizer=tf.keras.optimizers.Adam(learning_rate=config['lr']),
# 			loss=tf.keras.losses.CategoricalCrossentropy(name='loss', from_logits=True),
# 			metrics=[tf.keras.metrics.CategoricalAccuracy(name='accuracy')]
# 		)

# 		if parameters is not None:
# 			self.model.set_weights(parameters)

# 	def get_parameters(self, config={}):
# 		""" Get model weights """
# 		return self.model.get_weights()

# 	def fit(self, parameters, config):
# 		# Set parameters
# 		self.set_parameters(parameters, config)
# 		# Client local update
# 		h = self.model.fit(self.data, epochs=config['epochs'], verbose=0)
# 		metrics = {
# 			'accuracy':float(h.history['accuracy'][-1]),
# 			'loss':float(h.history['loss'][-1])
# 		}
# 		return self.get_parameters(), self.num_samples, metrics

# 	def evaluate(self, parameters, config):
# 		raise NotImplementedError('Client-side evaluation is not implemented!')


import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from copy import deepcopy
from collections import OrderedDict
class _Client(fl.client.NumPyClient):

    def __init__(self, cid, num_clients, model_loader, data_loader, shuffle=True, load_data=True, **kwargs):
        self.cid = cid
        self.shuffle = shuffle
        if load_data:
            self.data, self.num_classes, self.num_samples, self.labels, self.idxs = data_loader(
                shard_id=int(cid), num_shards=num_clients, shuffle=self.shuffle,
                batch_size=kwargs['batch_size'], seed=kwargs['batch_size'],
                noisy_clients_frac=kwargs['noisy_clients_frac'], noise_lvl=kwargs['noise_lvl'],
                noise_sparsity=kwargs['noise_sparsity']
            )
            if shuffle:
                del self.labels  # labels order is not preserved for shuffle==true
        self.model_loader = model_loader
        if load_data:
            self.input_shape = self.data.dataset[0][0].shape

    def set_parameters(self, parameters, config):
        """ Set model weights """
        if not hasattr(self, 'model'):
            self.model = self.model_loader(input_shape=self.input_shape[1:], num_classes=self.num_classes)

        # self.optimizer = optim.Adam(self.model.parameters(), lr=config['lr'])
        self.optimizer = optim.SGD(self.model.parameters(), lr=config['lr'], momentum=config['momentum'])
        
        self.criterion = nn.CrossEntropyLoss()

        if parameters is not None:
            params_dict = zip(self.model.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            self.model.load_state_dict(state_dict, strict=True)

    def get_parameters(self, config={}):
        """ Get model weights """
        try:
            return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
        except:
            return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def fit(self, parameters, config):
        # Set parameters
        self.set_parameters(parameters, config)
        
        # Client local update
        self.model.train()
        for epoch in range(config['epochs']):
            for batch in self.data:
                images, labels = batch
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

        metrics = {
            'accuracy': self.evaluate_accuracy(),
            'loss': loss.item()
        }
        return self.get_parameters(), self.num_samples, metrics

    def evaluate(self, parameters, config):
        raise NotImplementedError('Client-side evaluation is not implemented!')

    def evaluate_accuracy(self):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in self.data:
                images, labels, idxs = batch
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model.forward_only(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return 100 * correct / total
