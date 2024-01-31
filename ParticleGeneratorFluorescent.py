import deeptrack as dt
import numpy as np
import matplotlib.pyplot as plt

max_nr_particles = 1

particle = dt.PointParticle(                                         
    intensity=100,
    position=lambda: np.random.rand(2) * 128
)

fluorescence_microscope = dt.Fluorescence(
    NA=0.7,                
    resolution=1e-6,     
    magnification=10,
    wavelength=680e-9,
    output_region=(0, 0, 128, 128)
)


offset = dt.Add(
    value=lambda: np.random.rand()*1
)

poisson_noise = dt.Poisson(
    snr=lambda: np.random.rand()*7 + 3,
    background=offset.value
)

num_particles = lambda: np.random.randint(1, max_nr_particles+1)

image_features = fluorescence_microscope(particle^num_particles) >> offset >> poisson_noise

for i in range(4):
    image_features.update()
    output_image = image_features.plot(cmap="gray")

plt.show()

