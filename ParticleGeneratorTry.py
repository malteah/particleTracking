#%%
import deeptrack as dt
import numpy as np
#%%

def generate_particles(IMAGE_SIZE: int = 128, PARTICLE_RANGE: int = 8, NOISE_VALUE: float=0.001, RADIUS_FACTOR = 1)->(np.ndarray, np.ndarray, np.ndarray):
    #Skapar partiklar
    if PARTICLE_RANGE == 0:
        nan = np.zeros((IMAGE_SIZE, IMAGE_SIZE))
        return nan,nan,nan
    elif PARTICLE_RANGE == 1:
        NR_OF_PARTICLES = 1
    elif PARTICLE_RANGE > 1: 
        NR_OF_PARTICLES = np.random.randint(1,PARTICLE_RANGE)

    
    particle = dt.Sphere(
        position=lambda: np.random.rand(2) * 128,
        z=lambda: -10 + np.random.rand() * 20,
        position_unit="pixel",
        radius=lambda: 400e-9 + np.random.rand() * 100e-9,
        refractive_index=lambda: 1.45 + (0.02j + np.random.rand() * 0.01j),
        particle_type = 1
    )

    # Define the optics
    spectrum = np.linspace(400e-9, 700e-9, 3)

    illumination_gradient = dt.IlluminationGradient(gradient=lambda: np.random.randn(2) * 0.0002)

    fluorescense_microscope = [dt.Fluorescence(
                                wavelength=wavelength,
                                NA=1,
                                resolution=1e-6,
                                magnification=10,
                                refractive_index_medium=1.33,
    #                             illumination=illumination_gradient,
                                upsample=2,
                                output_region=(0, 0, 128, 128))
                            for wavelength 
                            in spectrum]

    # Define optical abberrations in the system
    noise = dt.Poisson(snr=lambda: 50 + np.random.rand() * 50)

    # Combine the features and normalize
    sample = particle

    incoherently_illuminated_sample = sum([fluorescense_microscope_one_wavelegth(sample) 
                                        for fluorescense_microscope_one_wavelegth 
                                        in fluorescense_microscope])

    augmented_image = incoherently_illuminated_sample

    im = augmented_image >> noise >> dt.NormalizeMinMax()

    im = sample.update()()
    positions = im.get_property('position', get_one=False)
    radii = im.get_property('radius', get_one=False)
    
    label = np.zeros((IMAGE_SIZE, IMAGE_SIZE))

    for i in range(len(radii)):
        x, y = np.meshgrid(np.arange(IMAGE_SIZE), np.arange(IMAGE_SIZE))
        gauss_blob = np.exp(-((x-positions[i][1])**2+(y-positions[i][0])**2)/(2*(radii[i]*1e8 * RADIUS_FACTOR)**2))
        label += gauss_blob

    # for _ in range(8):
    #     incoherently_illuminated_sample.update()
    #     incoherently_illuminated_sample.plot(cmap="gray")
    incoherently_illuminated_sample.update()

    #types: deeptrack.image.Image, np.ndarray,list
    # return im, label, positions
    return incoherently_illuminated_sample


# im,label,positions = generate_particles()
a = generate_particles()
a.plot(cmap="gray")

# print(type(im))
# print(type(label))
# print(type(positions))
