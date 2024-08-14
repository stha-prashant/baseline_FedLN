# import tensorflow as tf

# class Distiller(tf.keras.Model):

# 	def __init__(self, model, features, idxs):
# 		super(Distiller, self).__init__()
# 		self.model = model
# 		self.features = tf.convert_to_tensor(features, dtype=tf.float32)
# 		self.idxs = tf.cast(tf.constant(idxs), dtype=tf.int32)
# 		self.table = tf.lookup.experimental.DenseHashTable(key_dtype=tf.int32, value_dtype=tf.float32, default_value=tf.zeros(shape=(features.shape[1],), dtype=tf.float32), empty_key=-2, deleted_key=-3, name='table')
# 		self.table.insert(self.idxs, self.features)

# 	def compile(self, optimizer, loss, metrics, alpha=10.0, temperature=4.0,):
# 		super(Distiller, self).compile(optimizer=optimizer, metrics=metrics)
# 		self.student_loss_fn = loss
# 		self.distillation_loss_fn = tf.keras.losses.MeanAbsoluteError(name='mae_loss', reduction=tf.keras.losses.Reduction.AUTO)
# 		self.alpha = alpha
# 		self.temperature = temperature

# 	def train_step(self, data):
# 		((x, y), idxs) = data
# 		supervision_signal = self.table.lookup(tf.cast(idxs, dtype=tf.int32))
# 		with tf.GradientTape() as tape:
# 			preds, embeddings = self.model(x, training=True)
# 			loss = self.student_loss_fn(y, preds)
# 			distillation_loss = self.distillation_loss_fn(supervision_signal,embeddings)
# 			r_loss = tf.add_n(self.model.losses)
# 			loss  = loss + (self.alpha * distillation_loss) + r_loss
# 		# Compute gradients
# 		trainable_vars = self.model.trainable_variables
# 		gradients = tape.gradient(loss, trainable_vars)
# 		# Update weights
# 		self.optimizer.apply_gradients(zip(gradients, trainable_vars))
# 		# Update the metrics configured in `compile()`.
# 		self.compiled_metrics.update_state(y, preds)
# 		# Return a dict of performance
# 		return {"loss": loss, "accuracy": self.metrics[0].result(), "distil_loss": distillation_loss,}

# 	def test_step(self, data):
# 		x, y = data
# 		y_prediction = self.model(x, training=False)
# 		student_loss = self.student_loss_fn(y, y_prediction)
# 		self.compiled_metrics.update_state(y, y_prediction)
# 		results = {m.name: m.result() for m in self.metrics}
# 		results.update({"loss": student_loss})
# 		return results



	



import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pdb
class Distiller(nn.Module):
    def __init__(self, model, features, idxs, device='cuda:0', optimizer=None):
        super(Distiller, self).__init__()
        self.model = model
        self.features = torch.tensor(features, dtype=torch.float32).to(device)
        self.idxs = torch.tensor(idxs, dtype=torch.int32).to(device)
        self.device = "cuda:0" # TODO: hard coded here

        
        # Create a dictionary to simulate a lookup table
        self.table = {idx.item(): feature for idx, feature in zip(self.idxs, self.features)}
        self.compile(optimizer=optimizer, loss_fn=nn.CrossEntropyLoss(), metrics=None, alpha=10, temperature=4.0)

    def compile(self, optimizer, loss_fn, metrics, alpha=10.0, temperature=4.0):
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.metrics = metrics
        self.alpha = alpha
        self.temperature = temperature
        self.distillation_loss_fn = nn.L1Loss()

    def forward(self, data):
        # print("0000000000000000000000000000000000000000000000")
        x, y, idxs = data
        x, y, idxs = x.to(self.device), y.to(self.device), idxs.to(self.device)
        
        # Lookup supervision signal
        supervision_signal = torch.stack([self.table[idx.item()] for idx in idxs]).to(self.device)
        
        self.optimizer.zero_grad()
        preds, embeddings = self.model.forward_embeddings(x)
        
        # Calculate losses
        student_loss = self.loss_fn(preds, y)
        distillation_loss = self.distillation_loss_fn(supervision_signal, embeddings)
        r_loss = sum(self.model.losses) if hasattr(self.model, 'losses') else 0
        
        loss = student_loss + (self.alpha * distillation_loss) + r_loss
        loss.backward()
        self.optimizer.step()
        
        # Update the metrics
        # for metric in self.metrics:
        #     metric.update(preds, y)
        
        return {
            "student_loss": student_loss.item(),
            # "accuracy": self.metrics[0].compute(),
            "distillation_loss": distillation_loss.item(),
        }

    def forward_only(self, images):
        return self.model(images)

    def test_step(self, data):
        x, y = data
        x, y = x.to(self.device), y.to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(x)
            student_loss = self.loss_fn(y_pred, y)
        
        for metric in self.metrics:
            metric.update(y_pred, y)
        
        results = {metric.name: metric.compute() for metric in self.metrics}
        results.update({"loss": student_loss.item()})
        
        return results

# Example metric class
class Metric:
    def __init__(self, name):
        self.name = name
        self.reset()
    
    def update(self, preds, target):
        # Update metric computation here
        pass
    
    def compute(self):
        # Return the computed metric
        pass
    
    def reset(self):
        # Reset metric computation
        pass

# Usage example
# Assuming `model` is a PyTorch model, `features` is a numpy array, and `idxs` is a list of indices
# model = YourModel()
# features = np.array([...])
# idxs = np.array([...])
# distiller = Distiller(model, features, idxs)
# optimizer = optim.Adam(model.parameters(), lr=0.001)
# loss_fn = nn.CrossEntropyLoss()
# metrics = [Metric(name='accuracy')]

# distiller.compile(optimizer, loss_fn, metrics)
# for data in train_loader:
#     distiller.train_step(data)
# for data in test_loader:
#     distiller.test_step(data)
