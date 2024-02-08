import logging
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl

from torch.utils.data import DataLoader, random_split
from pathlib import Path
from datetime import datetime
from ParticleDataset import ParticleDataset
from properties import *
from tqdm import tqdm


############################################################################################
# model
class UNet(pl.LightningModule):
    def __init__(self, N_channels, N_classes):
        super(UNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(N_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, N_classes, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )

    def forward(self, x):
        x1 = self.encoder(x)
        x2 = self.decoder(x1)
        return x2

    # def configure_optimizers(self):
    #     return optim.Adam(self.parameters(), lr=Learning_rate)

    # def training_step(self, batch):
    #     x, y = batch
    #     logits = self(x)
    #     loss = nn.MSELoss()(logits, y)  # Example loss for image reconstruction
    #     self.log('train_loss', loss)
    #     return loss
############################################################################################
    

model = UNet(N_channels, N_classes)
model_name = "Unet_pl"

def train_model():
    # 1. Create dataset
    dataset = ParticleDataset(num_samples=Num_samples, image_size=Image_size, particle_range=Particle_range, noise_value=Noise_value, radius_factor=Radius_factor)


    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * Val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. Create data loaders
    train_loader = DataLoader(train_set, 
                              batch_size=Batch_size, 
                              shuffle=True, 
                              drop_last=True) 


    val_loader = DataLoader(val_set, 
                            batch_size=Batch_size, 
                            shuffle=True, 
                            drop_last=True) 

    # (Initialize logging)
    experiment = wandb.init(project= model_name, resume='allow', anonymous='must')
    experiment.config.update(
        dict(model_name = model_name,
                microscope = Microscope,
                epochs=Epochs, 
                batch_size=Batch_size, 
                learning_rate=Learning_rate,
                val_percent=Val_percent, 
                num_samples=Num_samples,
                image_size = Image_size,
                particle_range = Particle_range,
                noise_value = Noise_value,
                radius_factor = Radius_factor
                )
    )
    # # Initialize Lightning Trainer
    # trainer = pl.Trainer(max_epochs=Epochs, gpus=1)
    # # Create and train the model
   
    # # trainer.fit(model, train_loader, val_loader)

    # 4. Set up the optimizer, the loss and the learning rate scheduler
    optimizer = optim.Adam(model.parameters(),
                              lr=Learning_rate)
    criterion = Loss_function
    global_step = 0

    # 5. Begin training
    for epoch in range(1, Epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{Epochs}', unit='img') as pbar:
            for image, label in train_loader:

                #send the input to the Device
                image = image.to(device=Device)
                label = label.to(device=Device)

                # perform a forward pass and calculate the training loss
                pred = model(image)
                loss = criterion(pred, label)

                # zero out the gradients, perform the backpropagation step,
                # and update the weights
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # add the loss to the total training loss so far and
                # calculate the number of correct predictions
                epoch_loss += loss

                # update the progress bar
                pbar.set_postfix(**{'loss (batch)': loss.item()})
                pbar.update(image.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                experiment.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Evaluation round
                model.eval()

                # Wandb stuff
                division_step = (n_train // (5 * Batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        histograms = {}
                        for tag, value in model.named_parameters():
                            tag = tag.replace('/', '.')
                            if not (torch.isinf(value) | torch.isnan(value)).any():
                                histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                            if not (torch.isinf(value.grad) | torch.isnan(value.grad)).any():
                                histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                        for image, label in val_loader:
                            image = image.to(device=Device)
                            label = label.to(device=Device)
                            with torch.no_grad():
                                pred = model(image)
                                loss = criterion(pred, label)
                                val_loss = loss.item()
                                images = image
                                true = label
                                pred = pred
                        # val_score = evaluate(model, val_loader, Device, amp)
                        # scheduler.step(val_score)

                        logging.info('Validation score: {}'.format(val_loss))
                        try:
                            experiment.log({
                                'learning rate': optimizer.param_groups[0]['lr'],
                                'validation loss': val_loss,
                                'images': wandb.Image(images[0].cpu()),
                                'masks': {
                                    'true': wandb.Image(true[0].float().cpu()),
                                    'pred': wandb.Image(pred[0].float().cpu()),
                                },
                                'step': global_step,
                                'epoch': epoch,
                                **histograms
                            })
                        except:
                            pass






if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logging.info(f'Using Device {Device}')

    
    model = model
    model = model.to(memory_format=torch.channels_last)

    #uncomment when using UNet
    if model_name == "UNet":
        logging.info(f'Network:\n'
                        f'\t{model.n_channels} input channels\n'
                        f'\t{model.n_classes} output channels (classes)\n'
                        f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')

    model.to(device=Device)
    
    train_model()

    # get current date and time
    now = datetime.now()
    now = now.strftime("%Y-%m-%d_%H-%M-%S")

    torch.save(model, SAVE_PATH + f'/malte{now}.pth')