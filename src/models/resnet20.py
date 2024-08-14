# import tensorflow as tf

# class ResNet20:

# 	@staticmethod
# 	def regularized_padded_conv(*args, **kwargs):
# 		return tf.keras.layers.Conv2D(*args, **kwargs, padding="same", kernel_regularizer=_regularizer, kernel_initializer="he_normal", use_bias=False)

# 	@staticmethod
# 	def bn_relu(x):
# 		x = tf.keras.layers.BatchNormalization()(x)
# 		return tf.keras.layers.ReLU()(x)

# 	@staticmethod
# 	def shortcut(x, filters, stride, mode):
# 		if x.shape[-1] == filters:
# 			return x
# 		elif mode == "B":
# 			return __class__.regularized_padded_conv(filters, 1, strides=stride)(x)
# 		elif mode == "B_original":
# 			x = __class__.regularized_padded_conv(filters, 1, strides=stride)(x)
# 			return tf.keras.layers.BatchNormalization()(x)
# 		elif mode == "A":
# 			return tf.pad(tf.keras.layers.MaxPool2D(1, stride)(x) if stride>1 else x, paddings=[(0, 0), (0, 0), (0, 0), (0, filters - x.shape[-1])])
# 		else:
# 			raise KeyError("Parameter shortcut_type not recognized!")

# 	@staticmethod
# 	def original_block(x, filters, stride=1, **kwargs):
# 		c1 = __class__.regularized_padded_conv(filters, 3, strides=stride)(x)
# 		c2 = __class__.regularized_padded_conv(filters, 3)(__class__.bn_relu(c1))
# 		c2 = tf.keras.layers.BatchNormalization()(c2)

# 		mode = "B_original" if _shortcut_type == "B" else _shortcut_type
# 		x = __class__.shortcut(x, filters, stride, mode=mode)
# 		x = tf.keras.layers.Add()([x, c2])
# 		return tf.keras.layers.ReLU()(x)

# 	@staticmethod
# 	def preactivation_block(x, filters, stride=1, preact_block=False):
# 		flow = __class__.bn_relu(x)
# 		if preact_block:
# 			x = flow
# 		c1 = __class__.regularized_padded_conv(filters, 3, strides=stride)(flow)
# 		if _dropout:
# 			c1 = tf.keras.layers.Dropout(_dropout)(c1)
# 		c2 = __class__.regularized_padded_conv(filters, 3)(__class__.bn_relu(c1))
# 		x = __class__.shortcut(x, filters, stride, mode=_shortcut_type)
# 		return x + c2

# 	@staticmethod
# 	def bootleneck_block(x, filters, stride=1, preact_block=False):
# 		flow = __class__.bn_relu(x)
# 		if preact_block:
# 			x = flow
# 		c1 = __class__.regularized_padded_conv(filters//_bootleneck_width, 1)(flow)
# 		c2 = __class__.regularized_padded_conv(filters//_bootleneck_width, 3, strides=stride)(__class__.bn_relu(c1))
# 		c3 = __class__.regularized_padded_conv(filters, 1)(__class__.bn_relu(c2))
# 		x = __class__.shortcut(x, filters, stride, mode=_shortcut_type)
# 		return x + c3

# 	@staticmethod
# 	def group_of_blocks(x, block_type, num_blocks, filters, stride, block_idx=0):
# 		global _preact_shortcuts
# 		preact_block = True if _preact_shortcuts or block_idx == 0 else False

# 		x = block_type(x, filters, stride, preact_block=preact_block)
# 		for i in range(num_blocks-1):
# 			x = block_type(x, filters)
# 		return x

# 	@staticmethod
# 	def Resnet(input_shape, n_classes, l2_reg=1e-4, group_sizes=(2, 2, 2), features=(16, 32, 64), strides=(1, 2, 2),
# 		shortcut_type="B", block_type="preactivated", first_conv={"filters": 16, "kernel_size": 3, "strides": 1},
# 		dropout=0, cardinality=1, bootleneck_width=4, preact_shortcuts=True, embeddings_dim=None):

# 		global _regularizer, _shortcut_type, _preact_projection, _dropout, _cardinality, _bootleneck_width, _preact_shortcuts
# 		_bootleneck_width = bootleneck_width
# 		_regularizer = tf.keras.regularizers.l2(l2_reg)
# 		_shortcut_type = shortcut_type
# 		_cardinality = cardinality
# 		_dropout = dropout
# 		_preact_shortcuts = preact_shortcuts

# 		block_types = {"preactivated": __class__.preactivation_block,
# 					"bootleneck": __class__.bootleneck_block,
# 					"original": __class__.original_block}

# 		selected_block = block_types[block_type]
# 		inputs = tf.keras.layers.Input(shape=input_shape)
# 		flow = __class__.regularized_padded_conv(**first_conv)(inputs)

# 		if block_type == "original":
# 			flow = __class__.bn_relu(flow)

# 		for block_idx, (group_size, feature, stride) in enumerate(zip(group_sizes, features, strides)):
# 			flow = __class__.group_of_blocks(flow, block_type=selected_block, num_blocks=group_size, block_idx=block_idx, filters=feature, stride=stride)

# 		if block_type != "original":
# 			flow = __class__.bn_relu(flow)

# 		flow = tf.keras.layers.GlobalAveragePooling2D()(flow)
# 		outputs = tf.keras.layers.Dense(n_classes, kernel_regularizer=_regularizer)(flow)

# 		if embeddings_dim is not None: 
# 			embeddings =  tf.keras.layers.Dense(embeddings_dim)(flow)
# 			return tf.keras.models.Model(inputs, [outputs, embeddings], name='audio_classifier')

# 		model = tf.keras.models.Model(inputs, outputs)
# 		return model


# def load_model(input_shape, num_classes, l2_reg=1e-4, shortcut_type="A", block_type="original", embeddings_dim=None):
# 	return ResNet20.Resnet(input_shape=input_shape, n_classes=num_classes, l2_reg=l2_reg, embeddings_dim=embeddings_dim,
# 							group_sizes=(3, 3, 3), features=(16, 32, 64), strides=(1, 2, 2),
# 							first_conv={"filters": 16, "kernel_size": 3, "strides": 1},
# 							shortcut_type=shortcut_type, block_type=block_type, preact_shortcuts=False)

# def load_initial_model_weights(input_shape, num_classes):
# 	return ResNet20.load_model(input_shape=input_shape, num_classes=num_classes).get_weights()

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from torchvision import models

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNet20(nn.Module):
    def __init__(self, input_shape, n_classes=10, l2_reg=1e-4, group_sizes=(2, 2, 2), features=(16, 32, 64),
                 strides=(1, 2, 2), shortcut_type="B", block_type="preactivated", first_conv={"filters": 16, "kernel_size": 3, "strides": 1},
                 dropout=0, cardinality=1, bootleneck_width=4, preact_shortcuts=True, embeddings_dim=None):
        super(ResNet20, self).__init__()

        self.shortcut_type = shortcut_type
        self.cardinality = cardinality
        self.dropout = dropout
        self.bootleneck_width = bootleneck_width
        self.preact_shortcuts = preact_shortcuts
        self.input_channels = input_shape[0]

        block_types = {
            "preactivated": self.preactivation_block,
            "bootleneck": self.bootleneck_block,
            "original": self.original_block
        }

        self.selected_block = block_types[block_type]
        self.conv1 = self.regularized_padded_conv(first_conv["filters"], first_conv["kernel_size"], stride=first_conv["strides"])

        if block_type == "original":
            self.bn_relu1 = self.bn_relu_block(first_conv["filters"])
        
        self.blocks = nn.ModuleList()
        for block_idx, (group_size, feature, stride) in enumerate(zip(group_sizes, features, strides)):
            block = self.group_of_blocks(self.selected_block, group_size, feature, stride, block_idx)
            self.blocks.append(block)

        if block_type != "original":
            self.bn_relu2 = self.bn_relu_block(features[-1])

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(features[-1], n_classes)

        if embeddings_dim is not None:
            self.embeddings = nn.Linear(features[-1], embeddings_dim)
            self.output_mode = 'embeddings'
        else:
            self.output_mode = 'classifier'

    def regularized_padded_conv(self, out_channels, kernel_size, stride=1):
        return nn.Conv2d(in_channels=self.input_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=1, bias=False)

    def bn_relu_block(self, num_features):
        return nn.Sequential(
            nn.BatchNorm2d(num_features),
            nn.ReLU()
        )

    def shortcut(self, x, filters, stride, mode):
        if x.shape[1] == filters:
            return x
        elif mode == "B":
            return self.regularized_padded_conv(filters, 1, stride=stride)(x)
        elif mode == "B_original":
            x = self.regularized_padded_conv(filters, 1, stride=stride)(x)
            return nn.BatchNorm2d(filters)(x)
        elif mode == "A":
            # Correct padding logic for mode "A"
            pad = filters - x.shape[1]
            out = F.avg_pool2d(x, 1, stride)
            return F.pad(out, (0, 0, 0, 0, 0, pad))
        else:
            raise KeyError("Parameter shortcut_type not recognized!")

    def original_block(self, x, filters, stride=1):
        c1 = self.regularized_padded_conv(filters, 3, stride=stride)(x)
        c2 = self.regularized_padded_conv(filters, 3)(self.bn_relu_block(c1.shape[1])(c1))
        c2 = nn.BatchNorm2d(filters)(c2)

        mode = "B_original" if self.shortcut_type == "B" else self.shortcut_type
        x = self.shortcut(x, filters, stride, mode=mode)
        x = torch.add(x, c2)
        return F.relu(x)

    def preactivation_block(self, x, filters, stride=1, preact_block=False):
        flow = self.bn_relu_block(x.shape[1])(x)
        if preact_block:
            x = flow
        c1 = self.regularized_padded_conv(filters, 3, stride=stride)(flow)
        if self.dropout:
            c1 = nn.Dropout(self.dropout)(c1)
        c2 = self.regularized_padded_conv(filters, 3)(self.bn_relu_block(c1.shape[1])(c1))
        x = self.shortcut(x, filters, stride, mode=self.shortcut_type)
        return x + c2

    def bootleneck_block(self, x, filters, stride=1, preact_block=False):
        flow = self.bn_relu_block(x.shape[1])(x)
        if preact_block:
            x = flow
        c1 = self.regularized_padded_conv(filters // self.bootleneck_width, 1)(flow)
        c2 = self.regularized_padded_conv(filters // self.bootleneck_width, 3, stride=stride)(self.bn_relu_block(c1.shape[1])(c1))
        c3 = self.regularized_padded_conv(filters, 1)(self.bn_relu_block(c2.shape[1])(c2))
        x = self.shortcut(x, filters, stride, mode=self.shortcut_type)
        return x + c3

    def group_of_blocks(self, block_type, num_blocks, filters, stride, block_idx=0):
        layers = []
        preact_block = self.preact_shortcuts or block_idx == 0
        layers.append(block_type(filters, stride=stride, preact_block=preact_block))
        for i in range(1, num_blocks):
            layers.append(block_type(filters))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)

        if hasattr(self, 'bn_relu1'):
            x = self.bn_relu1(x)

        for block in self.blocks:
            x = block(x)

        if hasattr(self, 'bn_relu2'):
            x = self.bn_relu2(x)

        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        if self.output_mode == 'embeddings':
            x = self.embeddings(x)
        else:
            x = self.fc(x)
        return x


# def load_model(input_shape, num_classes, l2_reg=1e-4, shortcut_type="A", block_type="original", embeddings_dim=None):
#     return ResNet20(input_shape=input_shape, n_classes=num_classes, l2_reg=l2_reg, embeddings_dim=embeddings_dim,
#                     group_sizes=(3, 3, 3), features=(16, 32, 64), strides=(1, 2, 2),
#                     first_conv={"filters": 16, "kernel_size": 3, "strides": 1},
#                     shortcut_type=shortcut_type, block_type=block_type, preact_shortcuts=False)
class ResNet50(nn.Module):
    def __init__(self, num_classes=10, embeddings_dim=512):
        super(ResNet50, self).__init__()
        self.backbone = models.resnet18()
        # self.backbone = ResNet20()

        self.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        # if embeddings_dim is not None:
        self.embeddings = nn.Linear(self.backbone.fc.in_features, embeddings_dim) # hardcoded here
        self.backbone.fc = nn.Identity()
        

    def forward(self, input):
        return self.fc(self.backbone(input))

    def forward_embeddings(self, input):
        # breakpoint()
        x = self.backbone(input)
        return self.fc(x), self.embeddings(x)


from copy import deepcopy
def load_model(input_shape, num_classes, l2_reg=1e-4, shortcut_type="A", block_type="original", embeddings_dim=None):
    # model = models.resnet18()
    # model.fc = nn.Linear(model.fc.in_features, num_classes)
    # if embeddings_dim is not None:
    #     model.embeddings = nn.Linear(model.fc.in_features, embeddings_dim)
    print('-----numclasses: ', num_classes, embeddings_dim)
    return ResNet50(num_classes=num_classes)
    # return model

def load_initial_model_weights(input_shape, num_classes):
    model = load_model(input_shape, num_classes)
    return deepcopy(model.state_dict())

if __name__ == "__main__":
	# Example usage
	input_shape = (3, 32, 32)  # CIFAR-10 images shape
	num_classes = 10

	model = load_model(input_shape, num_classes)
	summary(model, input_shape)
