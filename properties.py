import models
import torch
import torch.nn as nn
import os

#model
N_channels  = 1         # n_channels=1 for grayscale images
N_classes = 1

Model = models.UNet_pl(N_channels,N_classes)
Model_name = "UNet_pl"
Loss_function = nn.BCELoss(torch.tensor(5).to('cuda:0'))

#run
Epochs = 20
Batch_size = 1
Learning_rate = 0.0001
Val_percent = 0.1       # % of samples used for validation
Num_samples = 1000

#optics 
Microscope = "brightfield" #(change in ParticleDataset.py)
Image_size = 128
Particle_range = 1
Noise_value = 0.0001
Radius_factor = 1


#other
Load_model = False      # Use an existing model
SAVE_PATH = os.getcwd() # Path for saving model
Device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')





# Models: 
# models.UNet(N_channels,N_classes, True), 
# models.UNet_jr(N_channels,N_classes), 
# models.UNet_pl(N_channels,N_classes)

# Loss_functions:
# Loss_function = nn.BCELoss(torch.tensor(5).to('cuda:0'))