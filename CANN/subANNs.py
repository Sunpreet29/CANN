# Standard imports
import torch
import torch.nn as nn


class DirLayers(nn.Module):
	def __init__(self, numExtra):
		super().__init__()
		self.numExtra = numExtra
		self.nn_layer1 = nn.Linear(self.numExtra, 8)
		self.nn_layer2 = nn.Linear(8, 8)
		self.activation = nn.Tanh()

	def forward(self, extra):
		dir = self.activation(self.nn_layer1(extra))
		dir = self.activation(self.nn_layer2(dir))
		return dir


class WLayers(nn.Module):
	def __init__(self, numExtra):
		super().__init__()
		self.numExtra = numExtra
		self.nn_layer1 = nn.Linear(self.numExtra, 8)
		self.nn_layer2 = nn.Linear(8, 8)
		self.activation = nn.Sigmoid()

	def forward(self, extra):
		w = self.activation(self.nn_layer1(extra))
		w = self.activation(self.nn_layer2(w))
		return w


class PsiLayers(nn.Module):
	def __init__(self, inDim):
		super().__init__()
		self.inDim = inDim
		self.nn_layerI1 = nn.Linear(self.inDim, 8)
		self.nn_layerI2 = nn.Linear(8, 8)
		self.nn_layerJ1 = nn.Linear(self.inDim, 8)
		self.nn_layerJ2 = nn.Linear(8, 8)
		self.activation = nn.Softplus()

	def forward(self, I, J, III_C):
		I = self.activation(self.nn_layerI1(I))
		I = self.activation(self.nn_layerI2(I))

		J = self.activation(self.nn_layerJ1(J))
		J = self.activation(self.nn_layerJ2(J))

		Psi = torch.cat((I, J), dim=1)
		return Psi