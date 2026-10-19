# Standard imports
import torch
import torch.nn as nn

# Own files
from CANN import ContinuumMechanics as CM
from CANN.subANNs import DirLayers, WLayers, PsiLayers

def unitVector(dir, numDir):
	"""
	Transforms the flat tensor that results from the directional subANN to a properly shaped tensor which is normalized (each vector is of length 1).

	Parameters
	----------
	dir : tensor [shape: (?, numDir * 3)]
		Flat tensor which results from the directional subANN layers.

	Returns
	-------
	dir : tensor [shape: (?, numDir, 3)]
		Reshaped tensor that represents the directional vectors in normalized form (each direction vector is a unit vector).
		
	"""
	dir = torch.reshape(dir, (-1, numDir, 3))		# -1 preserves the batch size, not required in Tensorflow
	length = torch.norm(dir, p=2, dim=2, dtype=dir.dtype)	# Eucledean norm
	dir = dir / torch.tile(length.unsqueeze(2), (1,1,3))

	return dir

def summation(w, numDir, numTens):
	"""
	Transforms the flat tensor that results from the weighting subANN to a properly shaped tensor which is normalized (the weight in each grou sum to 1).
	
	Parameters
	----------
	w : tensor [shape: (?, numTens * (numDir+1))]
		Flat tensor which results from the weighting factor subANN layers.
	
	Returns
	-------
	w : tensor [shape: (?, numTens, numDir+1)]
		Reshaped tensor that represents the weighting factors in normalized form (the weight in each grou sum to 1).
		
	"""

	w = torch.reshape(w, (-1, numTens, numDir+1))	# -1 preserves the batch size, not required in Tensorflow
	sum = torch.norm(w, dim=2, p=1)
	w = w / torch.tile(sum.unsqueeze(2), (1,1,numDir+1))
	
	return w


class Dir_subANN(nn.Module):
	def __init__(self, numDir, numExtra):
		super().__init__()
		self.numDir = numDir
		self.numExtra = numExtra
		self.dirlayer = DirLayers(numExtra)
		self.output_layer = nn.Linear(8, self.numDir*3, bias=False)
		self.activation = nn.Tanh()

	def forward(self, extra):
		dir = self.dirlayer(extra)
		dir = self.activation(self.output_layer(dir))
		dir = unitVector(dir, self.numDir)
		return dir


class W_subANN(nn.Module):
	def __init__(self, numTens, numDir, numExtra):
		super().__init__()
		self.numTens = numTens
		self.numDir = numDir
		self.numExtra = numExtra
		self.wlayer = WLayers(numExtra)
		self.output_layer = nn.Linear(8, self.numTens*(self.numDir+1), bias=False)
		self.activation = nn.Sigmoid()

	def forward(self, extra):
		w = self.wlayer(extra)
		w = self.activation(self.output_layer(w))
		w = summation(w, self.numDir, self.numTens)
		return w


class Psi_subANN(nn.Module):
	def __init__(self, computePsiOnly, numTens, numExtra, Psi=None):
		super().__init__()
		self.computePsiOnly = computePsiOnly
		self.numTens = numTens
		self.numExtra = numExtra
		if computePsiOnly:
			self.psilayer = Psi                 # analytical strain-energy function (not the ANN)
			self.output_layer = None            # unused in computePsiOnly mode
		else:
			self.psilayer = PsiLayers(numTens + numExtra)
			self.output_layer = nn.Linear(16, 1, bias=False)

	def forward(self, I, J, III_C, extra=False):
		if self.computePsiOnly:
			return self.psilayer(I, J, III_C)

		if self.numExtra > 0:
			I = torch.concat([I, extra], dim=1)
			J = torch.concat([J, extra], dim=1)
			III_C = torch.concat([III_C, extra], dim=1)

		Psi = self.psilayer(I, J, III_C)
		Psi = self.output_layer(Psi)
		return Psi


class CANN(nn.Module):
	def __init__(self, numTens, numDir, numExtra, computePsiOnly, incomp, Psi=None):
		super().__init__()
		self.numTens = numTens
		self.numDir = numDir
		self.numExtra = numExtra
		self.computePsiOnly = computePsiOnly
		self.incomp = incomp

		self.psi_ann = Psi_subANN(computePsiOnly, numTens, numExtra, Psi=Psi)

		if numDir > 0:
			if numExtra == 0:
				print('\n\n\n\nNo extra variables are provided; numDir cannot be non-zero')
				exit()

			self.dir_ann = Dir_subANN(numDir, numExtra)
			self.w_ann = W_subANN(numTens, numDir, numExtra)

		else:
			self.dir_ann = None
			self.w_ann = None

	def forward(self, F, extra=None):
		F.requires_grad_(True)		# enable ten2_P
		F, extra, C, invariants_I, invariants_J, invariant_III_C, F_isoRef, C_isoRef, invariants_I_isoRef, invariants_J_isoRef, invariants_III_C_isoRef = CM.pre_Psi(F, extra, self.numExtra, self.numTens, self.numDir, self.w_ann, self.dir_ann)

		Psi = self.psi_ann(invariants_I, invariants_J, invariant_III_C, extra)
		Psi_isoRef = self.psi_ann(invariants_I_isoRef, invariants_J_isoRef, invariants_III_C_isoRef, extra)
		Psi = Psi - Psi_isoRef                           # Psi=0 in reference config

		if self.incomp:
			P11, P, S, sigma = CM.post_Psi_incomp(Psi, Psi_isoRef, F, F_isoRef)
		else:
			P11, P, S, sigma = CM.post_Psi(Psi, F)
		return P11, Psi, P, S, sigma                     # use P11 for the loss