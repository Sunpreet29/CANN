# Standard imports
import numpy as np
import time
import torch
import torch.nn as nn

# Own files
from CANN import CANN
from CANN import Outputs
from CANN import Inputs
from CANN import Helpers as help


#####
##
#####

# Switch to read pre-defined model from HDD (attention: when using a noisy data set the data import will create new/different noise)
readModelFromDisk = False

# Switch to change problem
problem = 2
"""
	0: Treloar
	1: Generalized Mooney Rivlin
	2: Generalized Mooney Rivlin with noise
"""

#####
## General
#####

# Device (PyTorch is eager by default, so no disable_eager_execution needed)
device = torch.device('cpu')  # tiny model

# Obtain input data and problem specific settings
F, P11, extra, ds, problemName, numTens, numDir, batchSize, epochs, incomp = Inputs.defineProblem(problem)

# Create output folder
outputFolder = Outputs.prepareOutputFolder('output/', problemName)

#####
## Create model and statistics
#####
if type(extra) is np.ndarray:
    numExtra = extra.shape[1]
else:
    numExtra = 0

# Create model (one nn.Module replaces the two Keras models model_fit / model_full)
model = CANN.CANN(numTens=numTens, numDir=numDir, numExtra=numExtra, computePsiOnly=False, incomp=incomp).to(device)

# Output debugging information
Outputs.showModelSummary(model, numDir=numDir, outputFolder=outputFolder)
Outputs.plotModelGraph(model, outputFolder, numDir=numDir)


def to_inputs(x):
    # trainValidationSplit returns F alone (numExtra==0) or [F, extra] (numExtra>0)
    if numExtra == 0:
        return torch.tensor(x, dtype=torch.float32, device=device), None
    return (torch.tensor(x[0], dtype=torch.float32, device=device),
            torch.tensor(x[1], dtype=torch.float32, device=device))


if not readModelFromDisk:
    #####
    ## Train model
    #####
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    # Split data into training and validation
    train_in, train_out, val_in, val_out = help.trainValidationSplit(F, P11, extra, numExtra, trainSize=0.8, outputFolder=outputFolder)

    F_tr, extra_tr = to_inputs(train_in)
    y_tr = torch.tensor(train_out, dtype=torch.float32, device=device)
    F_val, extra_val = to_inputs(val_in)
    y_val = torch.tensor(val_out, dtype=torch.float32, device=device)

    nTrain = F_tr.shape[0]
    history = {'loss': [], 'val_loss': []}
    best_val = float('inf')

    startTime = time.time()
    for epoch in range(epochs):
        # --- training ---
        model.train()
        perm = torch.randperm(nTrain)
        running = 0.0
        for i in range(0, nTrain, batchSize):
            idx = perm[i:i + batchSize]
            Fb = F_tr[idx]
            yb = y_tr[idx]
            eb = extra_tr[idx] if extra_tr is not None else None

            optimizer.zero_grad()
            pred = model(Fb, eb)[0]          # P11 is the first return value
            loss = loss_fn(pred, yb)
            loss.backward()                  # backprops THROUGH ten2_P's dPsi/dF (double backward)
            optimizer.step()
            running += loss.item() * Fb.shape[0]
        train_loss = running / nTrain

        # validation (No torch.no_grad(): P11 itself needs autograd)
        model.eval()
        val_loss = loss_fn(model(F_val, extra_val)[0], y_val).item()

        history['loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        print('Epoch %d/%d  loss: %.6e  val_loss: %.6e' % (epoch + 1, epochs, train_loss, val_loss))

        # Checkpoint: save best weights only (replaces Keras ModelCheckpoint)
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), outputFolder + 'modelWeights.pth')
            print('   val_loss improved -> weights saved')

    endTime = time.time()
    print('\n\n\n       FITTING TIME: ' + help.timeString(endTime - startTime) + '\n\n')

    # Plot/save loss
    Outputs.plotLoss(history, outputFolder)
    Outputs.saveLoss(history, outputFolder)

# Reload best weights
model = help.loadWeights(model, outputFolder)

#####
## Visualization of results
#####
ds.predict(model)
Outputs.saveMainCurves(ds, outputFolder)
Outputs.plotMainCurves(ds, outputFolder)

if problem == 1 or problem == 2:  # Cannot be done for Treloar, since it's not based on an analytical strain energy
    Outputs.saveAndPlotErrors(ds, model, outputFolder, 55, 300, stepSizeI1=0.2, stepSizeI2=1.)
