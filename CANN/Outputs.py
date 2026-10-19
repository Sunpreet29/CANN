# Standard imports
import numpy as np
import torch
from torchview import draw_graph
import graphviz
import scipy.io
import matplotlib.pyplot as plt
import os
import pandas as pd
from contextlib import redirect_stdout

# Own files
from CANN import ContinuumMechanics as CM



def prepareOutputFolder(baseFolder, problemName):
	"""
	Creates all necessary output folders.

	Parameters
	----------
	baseFolder : string
		Name of the base folder for all exports.
	problemName : string
		Name of the current problem (will be used as sub folder name).
		
	Returns
	-------
	outputFolder : string
		Complete path to problem specific output folder (with filename prefix).
		
	"""
	if not os.path.exists(baseFolder):
		os.makedirs(baseFolder)
		
	outputFolder = baseFolder + problemName+'/'
	
	if not os.path.exists(outputFolder):
		os.makedirs(outputFolder)
		
	outputFolder = outputFolder + problemName+'_'
	
	return outputFolder




#####
## Plot/save main curves
#####

def saveMainCurves(ds, outputFolder):
	"""
	Saves the three main curves (uniaxial tension, pure shear, equi-biaxial tension) into a MATLAB file.

	Parameters
	----------
	ds : DataSource
		DataSource that contains curve information.
	outputFolder : string
		Output folder (and filename prefix).

	"""
	
	scipy.io.savemat(outputFolder+'result.mat', mdict={
		'train_ut_lam': ds.train_ut.lam,
		'train_ut_P11': ds.train_ut.P11,
		'train_ut_P11_pred': ds.train_ut.P11_pred,
		'plot_ut_lam': ds.plot_ut.lam,
		'plot_ut_P11': ds.plot_ut.P11,
		'plot_ut_P11_pred': ds.plot_ut.P11_pred,
#
		'train_bt_lam': ds.train_bt.lam,
		'train_bt_P11': ds.train_bt.P11,
		'train_bt_P11_pred': ds.train_bt.P11_pred,
		'plot_bt_lam': ds.plot_bt.lam,
		'plot_bt_P11': ds.plot_bt.P11,
		'plot_bt_P11_pred': ds.plot_bt.P11_pred,
#
		'train_ps_lam': ds.train_ps.lam,
		'train_ps_P11': ds.train_ps.P11,
		'train_ps_P11_pred': ds.train_ps.P11_pred,
		'plot_ps_lam': ds.plot_ps.lam,
		'plot_ps_P11': ds.plot_ps.P11,
		'plot_ps_P11_pred': ds.plot_ps.P11_pred,		
		})
		
		
		
		
def plotMainCurves(ds, outputFolder=False):
	"""
	Plots the three main curves (uniaxial tension, pure shear, equi-biaxial tension) and saves the figure if an outputFolder is given.

	Parameters
	----------
	ds : DataSource
		DataSource that contains curve information.
	outputFolder : string, optional
		Output folder (and filename prefix).

	"""
	
	plt.plot(ds.train_ut.lam, ds.train_ut.P11    , color='blue' , linestyle='None', marker='o')
	plt.plot(ds.plot_ut.lam , ds.plot_ut.P11_pred, color='blue')
	plt.plot(ds.train_bt.lam, ds.train_bt.P11    , color='green', linestyle='None', marker='o')
	plt.plot(ds.plot_bt.lam , ds.plot_bt.P11_pred, color='green')
	plt.plot(ds.train_ps.lam, ds.train_ps.P11    , color='red'  , linestyle='None', marker='o')
	plt.plot(ds.plot_ps.lam , ds.plot_ps.P11_pred, color='red')

	plt.legend(['Original UT', 'Prediction UT', 'Original BT', 'Prediction BT', 'Original PS', 'Prediction PS'], loc='upper left')
	plt.grid(visible=True, which='major', color='#CCCCCC', linestyle='-')
	if isinstance(outputFolder, str):
		plt.savefig(outputFolder+'result')
	plt.show()



#####
## Error plots
#####

def calculate2dErrorFields(ds, model_full, maxI1, maxI2, stepSizeI1=0.5, stepSizeI2=0.5):
	"""
	Calculates 2D error fields within the I1-I2 invariant plane for the strain energy density
	and the P_1 stress. The error is represented as relative error.
	
	Note: This function can be used for a very special case (isotropy, no extra features given,
	incompressible, material reference data is artificially based on an analytical strain energy
	function) only. It is designed to create the data for specific plots presented in the article.
	If used in other scenarios, it may not produce a proper representation of the overall error.

	Parameters
	----------
	ds : DataSource
		DataSource that contains analytical strain energy information.
	model_full : keras model
		Fitted keras model
	maxI1 : float
		Maximum value for the first invariant, defining the bounding box of the plot.
	maxI2 : float
		Maximum value for the second invariant, defining the bounding box of the plot.
	stepSizeI1 : float, optional
		Step size for the I1 discretization of the grid within the invariant plane.
	stepSizeI2 : float, optional
		Step size for the I2 discretization of the grid within the invariant plane. 
		
	Returns
	-------
	I1_mat : float matrix
		Used I1 values in grid form.
	I2_mat : float matrix
		Used I2 values in grid form.
	Psi_rel_error : float matrix
		Calculated relative error of the strain energy in grid form.
	P_rel_error : float matrix
		Calculated relative error of the P_1 stress in grid form.

	"""
	#####
	## Create grid
	#####
	# Define number of grid points
	nI1 = int(np.round((maxI1-3.0)/stepSizeI1+1))
	nI2 = int(np.round((maxI2-3.0)/stepSizeI2+1))

	# Create grid of I1, I2 values
	I1 = np.linspace(3.0, maxI1, nI1)
	I2 = np.linspace(3.0, maxI2, nI2)
	I2_mat, I1_mat = np.meshgrid(I2, I1)

	# Reshape grid into vector form for ANN input
	I1_ann = np.reshape(I1_mat, (nI1*nI2, 1)) / 3.0 # divided by three, because ANN uses generalized invariants
	I2_ann = np.reshape(I2_mat, (nI1*nI2, 1)) / 3.0 # divided by three, because ANN uses generalized invariants
	I3_ann = np.ones(I1_ann.shape)

	#####
	## PSI: Predict and calculate error
	#####
	# Predict model response (ANN and strain-energy reference) [PyTorch]
	Psi_ann = model_full.psi_ann(torch.tensor(I1_ann, dtype=torch.float32), torch.tensor(I2_ann, dtype=torch.float32), torch.tensor(I3_ann, dtype=torch.float32)).detach().numpy()
	Psi_ref = ds.model.psi_ann(torch.tensor(I1_ann, dtype=torch.float32), torch.tensor(I2_ann, dtype=torch.float32), torch.tensor(I3_ann, dtype=torch.float32)).detach().numpy()

	# Correct Psi to zero reference configuration
	_one = torch.ones((1, 1), dtype=torch.float32)
	Psi_ann_refConfig = model_full.psi_ann(_one, _one, _one).detach().numpy()
	Psi_ref_refConfig = ds.model.psi_ann(_one, _one, _one).detach().numpy()
	Psi_ann = Psi_ann - Psi_ann_refConfig
	Psi_ref = Psi_ref - Psi_ref_refConfig

	# Reshape to matrix form
	Psi_ann = np.reshape(Psi_ann, [nI1, nI2])
	Psi_ref = np.reshape(Psi_ref, [nI1, nI2])

	# Calculate relative error
	Psi_rel_error = np.abs((Psi_ann-Psi_ref) / Psi_ref) * 100.

	#####
	## P: Predict and calculate error
	#####
	# Derivatives of Psi w.r.t. I and J, via autograd [PyTorch]
	def _dPsi(psi_module):
		I_t   = torch.tensor(I1_ann, dtype=torch.float32, requires_grad=True)
		J_t   = torch.tensor(I2_ann, dtype=torch.float32, requires_grad=True)
		III_t = torch.tensor(I3_ann, dtype=torch.float32)
		Psi = psi_module(I_t, J_t, III_t)
		dI, dJ = torch.autograd.grad(Psi.sum(), (I_t, J_t))
		return dI.detach().numpy(), dJ.detach().numpy()

	dPsi_dI_ann, dPsi_dJ_ann = _dPsi(model_full.psi_ann)
	dPsi_dI_ref, dPsi_dJ_ref = _dPsi(ds.model.psi_ann)
	dPsi_dI_ann = np.squeeze(dPsi_dI_ann) / 3.
	dPsi_dJ_ann = np.squeeze(dPsi_dJ_ann) / 3.
	dPsi_dI_ref = np.squeeze(dPsi_dI_ref) / 3.
	dPsi_dJ_ref = np.squeeze(dPsi_dJ_ref) / 3.
	
	# Calculate principal stretches
	ps = CM.invariants2principalStretches(I1_ann*3., I2_ann*3., I3_ann)
	ps = np.sort(ps)
	ps1 = ps[:,2]
	ps2 = ps[:,1]
	
	# Calculate principal stresses (sigma)
	sigma1_ann = 2.*(np.multiply(ps1,ps1) - 1/np.multiply(np.multiply(ps1,ps1),np.multiply(ps2,ps2))) * (dPsi_dI_ann + np.multiply(np.multiply(ps2,ps2),dPsi_dJ_ann))
	sigma2_ann = 2.*(np.multiply(ps2,ps2) - 1/np.multiply(np.multiply(ps1,ps1),np.multiply(ps2,ps2))) * (dPsi_dI_ann + np.multiply(np.multiply(ps1,ps1),dPsi_dJ_ann))
	sigma1_ref = 2.*(np.multiply(ps1,ps1) - 1/np.multiply(np.multiply(ps1,ps1),np.multiply(ps2,ps2))) * (dPsi_dI_ref + np.multiply(np.multiply(ps2,ps2),dPsi_dJ_ref))
	sigma2_ref = 2.*(np.multiply(ps2,ps2) - 1/np.multiply(np.multiply(ps1,ps1),np.multiply(ps2,ps2))) * (dPsi_dI_ref + np.multiply(np.multiply(ps1,ps1),dPsi_dJ_ref))

	# Calculate principal stresses (P)
	P1_ann = np.divide(sigma1_ann, ps1)
	P2_ann = np.divide(sigma2_ann, ps2)
	P1_ref = np.divide(sigma1_ref, ps1)
	P2_ref = np.divide(sigma2_ref, ps2)

	# Reshape to matrix form
	P1_ann = np.real(np.reshape(P1_ann, [nI1, nI2]))
	P1_ref = np.real(np.reshape(P1_ref, [nI1, nI2]))
	P2_ann = np.real(np.reshape(P2_ann, [nI1, nI2]))
	P2_ref = np.real(np.reshape(P2_ref, [nI1, nI2]))
	
	# Calculate relative error
	P_rel_error = np.abs((P1_ann-P1_ref) / P1_ref) * 100.
	
	#####
	## Get rid of values which are physically impossible to reach (otherwise: garbage in, garbage out)
	#####
	# Calculate uniaxial tension and biaxial tension 
	lam = np.linspace(1.0, 8.0, 100000)
	I1_ut = lam**2 + 2/lam 
	I2_ut = 0.5*((lam**2 + 2/lam)**2 - (lam**4 + 2/lam**2))
	I1_bt = 2*lam**2 + 1/lam**4
	I2_bt = 0.5*((2*lam**2 + 1/lam**4)**2 - (2*lam**4 + 1/lam**8))

	# Put values into DataFrames so that the comparison can be done at one
	I1_killVal = pd.DataFrame(np.interp(I2, I2_bt, I1_bt))
	I1_killVal = I1_killVal.T
	I2_killVal = pd.DataFrame(np.interp(I1, I1_ut, I2_ut))

	I1_mat_df = pd.DataFrame(I1_mat)
	I2_mat_df = pd.DataFrame(I2_mat)

	# Get rid of unnecessary error values
	Psi_rel_error[I1_killVal.values > I1_mat_df.values] = np.nan
	Psi_rel_error[I2_killVal.values > I2_mat_df.values] = np.nan
	P_rel_error[I1_killVal.values > I1_mat_df.values] = np.nan
	P_rel_error[I2_killVal.values > I2_mat_df.values] = np.nan
	
	Psi_rel_error[np.isinf(Psi_rel_error)] = np.nan
	P_rel_error[np.isinf(P_rel_error)] = np.nan
	
	return I1_mat, I2_mat, Psi_rel_error, P_rel_error



def saveAndPlotErrors(ds, model_full, outputFolder, maxI1, maxI2, stepSizeI1=0.5, stepSizeI2=0.5):
	"""
	Plots and saves 2D error fields within the I1-I2 invariant plane for the strain energy density
	and the P_1 stress. The error is represented as relative error.
	
	Note: This function can be used for a very special case (isotropy, no extra features given,
	incompressible, material reference data is artificially based on an analytical strain energy
	function) only. It is designed to create the data for specific plots presented in the article.
	If used in other scenarios, it may not produce a proper representation of the overall error.

	Parameters
	----------
	ds : DataSource
		DataSource that contains analytical strain energy information.
	model_full : keras model
		Fitted keras model
	outputFolder : string, optional
		Output folder (and filename prefix).
	maxI1 : float
		Maximum value for the first invariant, defining the bounding box of the plot.
	maxI2 : float
		Maximum value for the second invariant, defining the bounding box of the plot.
	stepSizeI1 : float, optional
		Step size for the I1 discretization of the grid within the invariant plane.
	stepSizeI2 : float, optional
		Step size for the I2 discretization of the grid within the invariant plane. 

	"""
		
	I1_mat, I2_mat, Psi_rel_error, P_rel_error = calculate2dErrorFields(ds, model_full, maxI1, maxI2, stepSizeI1=0.5, stepSizeI2=0.5)

	#####
	## Plotting and saving
	#####
	# Create plot
	plt.pcolormesh(I1_mat, I2_mat, Psi_rel_error)
	plt.colorbar(format='%.2f')
	plt.savefig(outputFolder+'PsiError')
	plt.title('Rel. error Psi')
	plt.show()
	
	# Create plot
	plt.pcolormesh(I1_mat, I2_mat, P_rel_error)
	plt.colorbar(format='%.2f')
	plt.savefig(outputFolder+'PError')
	plt.title('Rel. error P_1')
	plt.show()

	# Save to Matlab
	scipy.io.savemat(outputFolder+'error.mat', mdict={
		'Psi_rel_error': Psi_rel_error,
		'P_rel_error': P_rel_error,
		'I1_mat': I1_mat,
		'I2_mat': I2_mat,
		})




#####
## Save/plot loss
#####

def saveLoss(his, outputFolder):
	"""
	Saves the loss to MATLAB.

	Parameters
	----------
	his : keras history object
		History object obtained during training.
	outputFolder : string
		Output folder (and filename prefix).

	"""
	scipy.io.savemat(outputFolder+'loss.mat', mdict={
		'loss': his['loss'],
		'val_loss': his['val_loss']
		})



def plotLoss(his, outputFolder=False):
	"""
	Plots the loss and, if an outputFolder is specified, saves the figure.

	Parameters
	----------
	his : keras history object
		History object obtained during training.
	outputFolder : string, optional
		Output folder (and filename prefix).

	"""
	
	plt.subplot(211)
	plt.plot(his['loss'])
	plt.plot(his['val_loss'])
	plt.ylabel('Loss')
	plt.xlabel('Epoch')
	plt.legend(['Training data', 'Validation data'], loc='upper right')
	plt.grid(visible=True, which='major', color='#CCCCCC', linestyle='-')
	
	plt.subplot(212)
	plt.plot(his['loss'])
	plt.plot(his['val_loss'])
	plt.ylabel('Loss')
	plt.xlabel('Epoch')
	plt.legend(['Training data', 'Validation data'], loc='upper right')
	plt.yscale('log')
	plt.grid(visible=True, which='major', color='#CCCCCC', linestyle='-')

	if isinstance(outputFolder, str):
		plt.savefig(outputFolder+'loss')
	plt.show()
	
#####
## Save model and statistics
#####

def saveTrainingValidationSplit(train_indices, val_indices, outputFolder):
	"""
	Saves the training/validation split into MATLAB.

	Parameters
	----------
	train_indices : float array
		Original indices that have been put into the training data set.
	val_indices : float array
		Original indices that have been put into the validation data set.
	outputFolder : string
		Output folder (and filename prefix).

	"""
	
	scipy.io.savemat(outputFolder+'trainingValidation.mat', mdict={
		'train_indices': train_indices,
		'val_indices': val_indices
		})



def showModelSummary(model, numDir=0, outputFolder=False):
	"""Prints the module structure for the whole model and each sub-ANN. If an outputFolder is given, each is also written to a text file."""
	def _dump(obj, suffix):
		print('\n\n\n')
		print(obj)
		if isinstance(outputFolder, str):
			with open(outputFolder + suffix, 'w') as f:
				f.write(str(obj))

	_dump(model, 'modelSummary_Full.txt')
	_dump(model.psi_ann, 'modelSummary_sub_Psi.txt')
	if numDir > 0:
		_dump(model.w_ann, 'modelSummary_sub_w.txt')
		_dump(model.dir_ann, 'modelSummary_sub_dir.txt')


def plotModelGraph(model, outputFolder, numDir=0):
	"""Saves PNG plots of the model graph (whole model + sub-ANNs) using torchview + graphviz."""

	numExtra = model.numExtra
	numTens = model.numTens
	device = next(model.parameters()).device
	b = 4  # dummy batch size for tracing

	def _save(m, input_data, name, expand=False):
		try:
			g = draw_graph(m, input_data=input_data, expand_nested=expand, device=device, graph_name=name)
			g.visual_graph.render(filename=outputFolder + name, format='png', cleanup=True)
		except Exception as e:
			print('plotModelGraph: could not render %s (%s)' % (name, e))

	# Dummy inputs (on the model's device)
	extra = torch.randn(b, numExtra, device=device) if numExtra > 0 else None

	# Note: the whole-model graph cannot be rendered with torchview, because the model's
	# forward differentiates internally (ten2_P via torch.autograd.grad) and torchview's
	# tracer cannot follow that. Only the learnable sub-ANNs (below) are graphed.

	# Psi sub-ANN (plain MLP, traces cleanly)
	I = torch.randn(b, numTens, device=device)
	J = torch.randn(b, numTens, device=device)
	III_C = torch.randn(b, 1, device=device)
	if numExtra > 0:
		_save(model.psi_ann, [I, J, III_C, extra], 'modelGraph_sub_Psi')
	else:
		_save(model.psi_ann, [I, J, III_C], 'modelGraph_sub_Psi')

	if numDir > 0:
		_save(model.dir_ann, extra, 'modelGraph_sub_dir')
		_save(model.w_ann, extra, 'modelGraph_sub_w')
