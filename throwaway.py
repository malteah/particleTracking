import deeptrack as dt
import matplotlib.pyplot as plt
import numpy as np


#Define parameters of the particles
particle = dt.Sphere(
    radius=0.5e-6,  # Radius of the particle in meters
    position=(128, 128),  # Position of the particle in the field of view
    refractive_index=1.5  # Refractive index of the particle
)

# particle=dt.MieSphere(  position=lambda: (np.random.uniform(22,42), np.random.uniform(22,42)),
#                         radius=lambda: np.random.uniform(.05,.08)*1e-6,
#                         refractive_index=lambda: np.random.uniform(1.35,1.6),
#                         z=lambda: np.random.uniform(-30,30),
#                         position_objective=(np.random.uniform(-250,250)*1e-6,
#                         np.random.uniform(-250,250)*1e-6,np.random.uniform(-15,15)*1e-6))



# Define fluorescence optics
optics = dt.Fluorescence(
    NA=1.4,  # Numerical Aperture
    wavelength=520e-9,  # Emission wavelength in meters (green light)
    magnification=60,  # Magnification
    resolution=200e-7,  # Resolution in meters
    refractive_index_medium=1.33,  # Refractive index (e.g., for water)
    output_region=(0, 0, 256, 256),
    pupil=None,
    illumination=None,
    upscale=1
)

# Function for altering the relative phase of the scattered light compared to the reference light
def phase_adder(ph):
    def inner(image):
        image=image-1
        image=image*np.exp(1j*ph)
        image=image+1
        return np.abs(image)
    return inner

phadd=dt.Lambda(phase_adder,ph=lambda: np.random.uniform(0,2*np.pi))

image_features=optics(particle) #Apply the optics to the particle
sample=image_features>>phadd #Randomly change the relative phase of scattered light and reference light

# sample=(sample>>dt.Gaussian(sigma=0.001)) #Add noise to the images

for i in range(4):
    image_features.update()
    output_image = image_features.plot(cmap="gray")


# im=sample.update()() #Create an image of a particle (now, all the lambda-functions in the parameters above are explicitly called, and a new set of parameters is generated)
# plt.imshow(np.abs(im)) #Plot the image of a particle
# plt.colorbar()
# plt.show()

# print(im.get_property("radius"))