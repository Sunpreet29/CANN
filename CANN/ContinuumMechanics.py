# Standard imports
import torch
import numpy as np

"""
	NOTE:
	All functions in this file are directly adopted from continuum mechanics fundamentals
	and hence do not need further introduction. In order to not inflate this file
	unnecessarily with comments (and sacrificing readability), we decided to mostly omit
	the type of comment blocks we put in all other functions of this project. These 
	comment blocks are only present when we deemed them necessary.
	
	Explanations to the continuum mechanics basics utilized can be found in the article 
	and/or standard textbooks on the subject.
	
	The functions in this file can be sorted into three groups:
	- basic continuum mechanics functions
	- collective continuum mechanics functions (that chain basic functions in a meaningful 
	  order)
	- Simple analytical strain-energy functions (for test runs or as artificial data 
	  sources)
"""

##########################################################################################
##########################################################################################
###### BASIC CONTINUUM MECHANICS FUNCTIONS ###############################################
##########################################################################################
##########################################################################################


def wrapper(numTens, numDir):
	"""
	Returns all basic continuum mechanics functions which have a direct dependency on the 
	number of generalized structural tensors or preferred directions to use. The returned
	functions are tailored to the desired specifications.

	Parameters
	----------
	numTens : int
		Number of generalized structural tensors to use (at least 1).
	numDir : int
		Number of preferred directions to use (0 for isotropy, more than 0 for anisotropy).

	Returns
	-------
	ten2_H : function
		A function for generalized structural tensors.
	invariants_I : function
		A function for generalized invariants I.
	invariants_J : function
		A function for generalized invariants J.
		
	"""
	
	def ten2_H(L, w): # Generalized structural tensors: H_r = \sum_i w_ri * L_i [?,numTens,3,3]
		batchSize = w.shape[0]

		# Create L_0 and add it to L
		eye = torch.eye(3, dtype=w.dtype, device=w.device)
		L_0 = torch.tile(eye.unsqueeze(0).unsqueeze(0), (batchSize, 1, 1, 1)) / 3.0
		if numDir > 0:
			L = torch.concat([L_0, L], dim=1)
		else:
			L = L_0

		# Expand L (to get one for each numTens)
		L = torch.tile(L.unsqueeze(1), (1, numTens, 1, 1, 1))

		# Expand w
		w = torch.tile(w.unsqueeze(3), (1, 1, 1, 3))
		w = torch.tile(w.unsqueeze(4), (1, 1, 1, 1, 3))
	
		# Multiply L with weights
		L_weighted = L * w
	
		# Sum them up for the corresponding H
		H = torch.sum(L_weighted, dim=2)
	
		return H
		
	def invariants_I(C, H): # Generalized invariants I: I_r = trace(C*H_r) [?,numTens]
		C_tile = torch.tile(C.unsqueeze(1), (1, numTens, 1, 1))

		return (C_tile @ H).diagonal(offset=0, dim1=-2, dim2=-1).sum(dim=-1)

	def invariants_J(C, H): # Generalized invariants J: J_r = trace(cofactor(C)*H_r) [?,numTens]		
		C_tile = torch.tile(C.unsqueeze(1), (1, numTens, 1, 1))
		
		detC_tile = torch.linalg.det(C_tile)
		detC_tile = torch.tile(detC_tile.unsqueeze(2), (1, 1, 3))
		detC_tile = torch.tile(detC_tile.unsqueeze(3), (1, 1, 1, 3))
		
		invTransC = torch.linalg.inv(torch.transpose(C_tile, 3, 2))
		
		mul = torch.multiply(detC_tile, invTransC)
		matmul = mul @ H
		
		return matmul.diagonal(offset=0, dim1=-2, dim2=-1).sum(dim=-1)
		
	return ten2_H, invariants_I, invariants_J

def defGrad_ut(lam): # Deformation gradient for incompressible uniaxial tension loading [?,3,3]
	F = np.zeros([len(lam), 3, 3])
	F[:,0,0] = lam
	F[:,1,1] = 1.0/(np.sqrt(lam))
	F[:,2,2] = 1.0/(np.sqrt(lam))
	return F

def defGrad_bt(lam): # Deformation gradient for incompressible equi-biaxial loading [?,3,3]
	F = np.zeros([len(lam), 3, 3])
	F[:,0,0] = lam
	F[:,1,1] = lam
	F[:,2,2] = 1.0/lam**2
	return F

def defGrad_ps(lam): # Deformation gradient for incompressible pure shear loading [?,3,3]
	F = np.zeros([len(lam), 3, 3])
	F[:,0,0] = lam
	F[:,1,1] = 1/lam
	F[:,2,2] = 1.0
	return F

def ten2_C(F): # Right Cauchy-Green tensor: C = F^T * F [?,3,3]
	return F.transpose(-2, -1) @ F

def ten2_F_isoRef(F): # Deformation gradient in reference configuration [?,3,3]
	# In Order for the other formulae to work we need the correct dimension required to produce enough eye matrices/tensors 
	eye = torch.eye(3, dtype=F.dtype, device=F.device).unsqueeze(0)
	F_isoRef = torch.tile(eye, (F.shape[0],1,1))
	
	return F_isoRef

def ten2_L(dir): # Structural tensor L_i = l_i (x) l_i [?,numDir,3,3]
	dir = dir.unsqueeze(3)
	dir_t = torch.transpose(dir, 3, 2)
	L = dir @ dir_t
	
	return L

def invariant_I3(C): # Third invariant of a tensor C: I3 = det(C) [?,1]
	return (torch.linalg.det(C)).unsqueeze(1)

def invariants2principalStretches(I1_arr, I2_arr, I3_arr): # Calculates the principal stretches based on invariants of C [only used for one specific kind of plot]
	# Itskov, 2015, Tensor Algebra and Tensor Analysis for Engineers, 4th edition, p. 103-104
	
	dim = I1_arr.shape
	eig = np.empty((dim[0],3,), dtype=np.complex_)
	eig[:,:] = np.NaN

	for i in range(dim[0]):
		I1 = I1_arr[i]
		I2 = I2_arr[i]
		I3 = I3_arr[i]
		if np.abs(np.power(I1,2)-3.*I2) > 1e-6:
			nom = 2.*np.power(I1,3) - 9.*np.multiply(I1,I2) + 27.*I3
			denom = 2.*np.power(np.power(I1,2) - 3*I2,1.5)
			theta = np.arccos(nom/denom)

			for k in [1, 2, 3]:
				eig[i,k-1] = (I1 + 2*np.sqrt(np.power(I1,2)-3.*I2)*np.cos((theta+2*np.pi*(k-1.))/3.))/3.
		else:
			for k in [1, 2, 3]:
				eig[i,k-1] = I1/3. + 1./3.*np.power(27.*I3-np.power(I1,3), 1./3.) * (np.cos(2./3.*np.pi*k) + (0+1j)*np.sin(2./3.*np.pi*k))

	principalStretch = np.sqrt(eig)

	return principalStretch

def ten2_P(Psi, F):        # First Piola-Kirchhoff stress: P = dPsi/dF  [?,3,3]
    der = torch.autograd.grad(
        Psi, F,
        grad_outputs=torch.ones_like(Psi),   # (1) Psi is [b,1], not scalar
        create_graph=True,                    # (2) so P stays differentiable for training
        retain_graph=True,
        materialize_grads=True,               # (3) ≈ unconnected_gradients='zero'
    )[0]
    return der
	
def ten2_P_lagMul(P_iso, F, lagMul): # Lagrange multiplier for incompressibility [?,1]
	FtransInv = torch.linalg.inv(torch.transpose(F, dim0=2, dim1=1))
	lagMul = torch.tile(lagMul.unsqueeze(2), (1,3,3))
	lastTerm = torch.multiply(lagMul, FtransInv)

	return P_iso - lastTerm

def ten2_S(P, F): # Second Piola Kirchhoff stress tensor: S = F^-1 * P [?,3,3]
	return torch.linalg.inv(F) @ P

def ten2_sigma(P, F, J): # Cauchy stress tensor: sigma = J^-1 * P * F^T [?,3,3]
	OneOverJ = torch.tile((1.0 / J).unsqueeze(2), (1,3,3))
	return torch.multiply(OneOverJ, P @ torch.transpose(F, dim0=2, dim1=1))


##########################################################################################
##########################################################################################
###### COLLECTIVE CONTINUUM MECHANICS FUNCTIONS ##########################################
##########################################################################################
##########################################################################################


def pre_Psi(numExtra, numTens, numDir, w_model, dir_model): # Deals with everything before the strain-energy is used (deformation measures, structural tensors, invariants)
	ten2_H, invariants_I, invariants_J = wrapper(numTens, numDir)
	
	if numExtra == 0:
		extra = []
	else:
		extra = k.layers.Input(shape=(numExtra,), name='extra') # INPUT
	
	# Deformation measures
	F = k.layers.Input(shape=(3,3,), name='F') # INPUT
	C = k.layers.Lambda(lambda F: ten2_C(F), name='C' )(F)
	
	# Directions and structure tensors
	if numDir == 0:
		dir = [] # we do not need directions (and hence their sub-ANN) at all
		w = tf.ones([tf.shape(F)[0],numTens,1]) # we do not need a sub-ANN to get the weights
		L = []
	else:
		dir = dir_model(extra)
		w = w_model(extra)
		L = k.layers.Lambda(lambda dir: ten2_L(dir), name='L')(dir)

	# Generalized structure tensors
	H = k.layers.Lambda(lambda x: ten2_H(x[0], x[1]), name='H')([L, w])

	# Generalized invariants
	inv_I     = k.layers.Lambda(lambda x: invariants_I(x[0], x[1]), name='invariants_I'   )([C,H])
	inv_J     = k.layers.Lambda(lambda x: invariants_J(x[0], x[1]), name='invariants_J'   )([C,H])
	inv_III_C = k.layers.Lambda(lambda C: invariant_I3(C)         , name='invariant_III_C')(C)
	
	# Determination of the eact reference configuration
	F_isoRef         = k.layers.Lambda(lambda F: ten2_F_isoRef(F), output_shape=(None,3,3), name='F_isoRef'              )(F)
	C_isoRef         = k.layers.Lambda(lambda F: ten2_C(F)                                , name='C_isoRef'              )(F_isoRef)
	inv_I_isoRef     = k.layers.Lambda(lambda x: invariants_I(x[0], x[1])                 , name='invariants_I_isoRef'   )([C_isoRef,H])
	inv_J_isoRef     = k.layers.Lambda(lambda x: invariants_J(x[0], x[1])                 , name='invariants_J_isoRef'   )([C_isoRef,H])
	inv_III_C_isoRef = k.layers.Lambda(lambda C_isoRef: invariant_I3(C_isoRef)            , name='invariant_III_C_isoRef')(C_isoRef)

	return F, extra, C, inv_I, inv_J, inv_III_C, F_isoRef, C_isoRef, inv_I_isoRef, inv_J_isoRef, inv_III_C_isoRef

def post_Psi(Psi, F): # Deals with everything after the strain-energy is used [variant for compressible materials] (stresses)
	P = k.layers.Lambda(lambda x: ten2_P(x[0], x[1]), name='P'    )([Psi, F])

	return post_Psi_both(Psi, P, F)

def post_Psi_incomp(Psi, Psi_isoRef, F, F_isoRef): # Deals with everything after the strain-energy is used [variant for incompressible materials] (stresses)
	P_iso    = k.layers.Lambda(lambda x: ten2_P(x[0], x[1])                      , name='P_iso'   )([Psi, F])
	P_isoRef = k.layers.Lambda(lambda x: ten2_P(x[0], x[1])                      , name='P_isoRef')([Psi_isoRef, F_isoRef])
	lagMul   = k.layers.Lambda(lambda P: tf.keras.backend.expand_dims(P[:,0,0],1), name='lagMul'  )(P_isoRef)
	P        = k.layers.Lambda(lambda x: ten2_P_lagMul(x[0], x[1], x[2])         , name='P'       )([P_iso, F, lagMul])

	return post_Psi_both(Psi, P, F)

def post_Psi_both(Psi, P, F): # Common parts from post_Psi & post_Psi_incomp
	S     = k.layers.Lambda(lambda x: ten2_S(x[0], x[1])                              , name='S'    )([P, F])
	J     = k.layers.Lambda(lambda F: tf.keras.backend.expand_dims(tf.linalg.det(F),1), name='J'    )(F)
	sigma = k.layers.Lambda(lambda x: ten2_sigma(x[0], x[1], x[2])                    , name='sigma')([P, F, J])
	P11   = k.layers.Lambda(lambda P: tf.keras.backend.expand_dims(P[:,0,0],1)        , name='P11'  )(P)

	return P11, P, S, sigma


##########################################################################################
##########################################################################################
###### ANALYTICAL STRAIN ENERGY DENSITY FUNCTIONS ########################################
##########################################################################################
##########################################################################################


def MooneyRivlin6term_wrapper(c10, c20, c30, c01, c02, c03):
	def MooneyRivlin6term(I, J, I3):
		I1 = I*3.0
		I2 = J*3.0

		Psi = k.layers.Lambda(lambda x: c10*(x[0]-3.0) + c20*(x[0]-3.0)**2 + c30*(x[0]-3.0)**3 + c01*(x[1]-3.0) + c02*(x[1]-3.0)**2 + c03*(x[1]-3.0)**3, name='Psi')([I1, I2, I3])

		return Psi
	return MooneyRivlin6term

def NeoHookean_wrapper(c):
	def NeoHookean(I, J, I3):
		I1 = I*3.0
		I2 = J*3.0
	
		Psi = k.layers.Lambda(lambda x: c*(x[0]-3.0), name='Psi')([I1, I2, I3])

		return Psi
	return NeoHookean

def MooneyRivlin_wrapper(c1, c2):
	def MooneyRivlin(I, J, I3):
		I1 = I*3.0
		I2 = J*3.0
	
		Psi = k.layers.Lambda(lambda x: c1*(x[0]-3.0) + c2*(x[1]-3.0), name='Psi')([I1, I2, I3])

		return Psi
	return MooneyRivlin