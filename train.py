import argparse
import logging
import os
import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, random_split

from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from Unet_pytorch import UNet
from ParticleDataset import ParticleDataset

dir_img = Path('./data/imgs/')
dir_mask = Path('./data/masks/')
dir_checkpoint = Path('./checkpoints/')


def train_model(
        model,
        device,
        epochs: int = 6,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        val_percent: float = 0.1,
        save_checkpoint: bool = True,
        img_scale: float = 1,
        amp: bool = True,
):

    # 1. Create dataset
    dataset = ParticleDataset(num_samples=1000, image_size=128, particle_range=1, noise_value=0.0, radius_factor=2)

    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. Create data loaders
    # loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
    train_loader = DataLoader(train_set, batch_size=batch_size)
    val_loader = DataLoader(val_set, drop_last=True)

    # (Initialize logging)
    experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')
    experiment.config.update(
        dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
             val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale, amp=amp)
    )

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.Adam(model.parameters(),
                              lr=learning_rate)
    criterion = nn.BCELoss()
    global_step = 0


    # 5. Begin training
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for image, label in train_loader:

                assert image.shape[1] == model.n_channels, \
                    f'Network has been defined with {model.n_channels} input channels, ' \
                    f'but loaded images have {image.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                #send the input to the device
                image = image.to(device=device)
                label = label.to(device=device)

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
                division_step = (n_train // (5 * batch_size))
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
                            image = image.to(device=device)
                            label = label.to(device=device)
                            with torch.no_grad():
                                pred = model(image)
                                loss = criterion(pred, label)
                                val_loss = loss.item()
                                val_score = val_loss #potentially val_score = 1 - val_loss

                                images = image
                                true = label
                                pred = pred
                        # val_score = evaluate(model, val_loader, device, amp)
                        # scheduler.step(val_score)

                        logging.info('Validation score: {}'.format(val_score))
                        try:
                            experiment.log({
                                'learning rate': optimizer.param_groups[0]['lr'],
                                'validation score': val_score,
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


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and labels')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=6, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=1, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=True, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=True, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=1, help='Number of classes')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # n_channels=3 for RGB images, else n_channels=1 for grayscale images
    model = UNet(n_channels=1, n_classes=1, bilinear=args.bilinear)
    model = model.to(memory_format=torch.channels_last)

    #uncomment when using UNet
    logging.info(f'Network:\n'
                 f'\t{model.n_channels} input channels\n'
                 f'\t{model.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')

    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        del state_dict['mask_values']
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {args.load}')

    model.to(device=device)
    try:
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp
        )


    except torch.cuda.OutOfMemoryError:
        logging.error('Detected OutOfMemoryError! '
                      'Enabling checkpointing to reduce memory usage, but this slows down training. '
                      'Consider enabling AMP (--amp) for fast and memory efficient training')
        torch.cuda.empty_cache()
        model.use_checkpointing()
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp
        )

        # save model
        current_dir = os.getcwd()
        # get current date and time
        now = datetime.datetime.now()
        now = now.strftime("%Y-%m-%d_%H-%M-%S")

    torch.save(model, current_dir + f'/malte{now}.pth')
