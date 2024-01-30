import torch
from ParticleGenerator import generate_particles

class ParticleDataset(torch.utils.data.Dataset):
    def __init__(self, image_size, particle_range, noise_value, num_samples, radius_factor):
        self.image_size = image_size
        self.particle_range = particle_range
        self.noise_value = noise_value
        self.num_samples = num_samples
        self.radius_factor = radius_factor

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        image, label, _ = generate_particles(self.image_size, self.particle_range, self.noise_value, self.radius_factor)
        image = torch.tensor(image).float().permute(2, 0, 1)
        label = torch.tensor(label).unsqueeze(-1).float().permute(2, 0, 1)
        return image, label