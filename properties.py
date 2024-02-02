import models
import torch
import torch.nn as nn
import os

#model
Bilinear = True         # Use bilinear upsampling
N_channels  = 1
N_classes = 1
Model = models.UNet(N_channels,N_classes,Bilinear)
Loss_function = nn.L1Loss()

#run
Epochs = 20
Batch_size = 1
Learning_rate = 0.0001
Val_percent = 0.1      # % of samples used for validation
Num_samples = 100

#optics 
#microscope = brightfield (change in ParticleDataset.py)
Image_size = 256
Particle_range = 1
Noise_value = 0.0001
Radius_factor = 1


#other
Load_model = False      # Use an existing model
SAVE_PATH = os.getcwd() # Path for saving model
Save_checkpoint = True
Amp  = False            # Use automated mixed precision
Device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


