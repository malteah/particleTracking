#%%
import deeptrack as dt
import matplotlib.pyplot as plt
import numpy as np
#%%

def generate_particles(IMAGE_SIZE: int = 128, PARTICLE_RANGE: int = 8, NOISE_VALUE: float=0.001, RADIUS_FACTOR = 1)->(np.ndarray, np.ndarray, np.ndarray):
    #Skapar partiklar
    if PARTICLE_RANGE == 0:
        return np.zeros((IMAGE_SIZE, IMAGE_SIZE)), np.zeros((IMAGE_SIZE, IMAGE_SIZE)), np.zeros((IMAGE_SIZE, IMAGE_SIZE))
    elif PARTICLE_RANGE == 1:
        NR_OF_PARTICLES = 1
    elif PARTICLE_RANGE > 1: 
        NR_OF_PARTICLES = np.random.randint(1,PARTICLE_RANGE)


    particle=dt.MieSphere(position=lambda: (np.random.uniform(0,IMAGE_SIZE),np.random.uniform(0,IMAGE_SIZE)),radius=lambda: np.random.uniform(.05,.08)*1e-6,refractive_index=lambda: np.random.uniform(1.35,1.6), z=lambda: np.random.uniform(-30,30),position_objective=(np.random.uniform(-250,250)*1e-6,np.random.uniform(-250,250)*1e-6,np.random.uniform(-15,15)*1e-6))^NR_OF_PARTICLES
    
    #Define optical abberrations in the system
    args=dt.Arguments(hccoeff=lambda: np.random.uniform(-100,100))
    pupil=dt.HorizontalComa(coefficient=args.hccoeff)

    #Define parameters of the microscope
    optics=dt.Brightfield(NA=1.1,working_distance=.2e-3,aberration=pupil,wavelength=660e-9,resolution=.15e-6,magnification=1,output_region=(0,0,IMAGE_SIZE,IMAGE_SIZE),return_field=True,illumination_angle=np.pi) 
    #Function for altering the relative phase of the scattered light compared to the reference light
    def phase_adder(ph):
        def inner(image):
            image=image-1
            image=image*np.exp(1j*ph)
            image=image+1
            return np.abs(image)
        return inner


    phadd=dt.Lambda(phase_adder,ph=lambda: np.random.uniform(0,2*np.pi))


    s0=optics(particle) #Apply the optics to the particle

    sample=s0>>phadd #Randomly change the relative phase of scattered light and reference light


    sample = (sample>>dt.Gaussian(sigma=NOISE_VALUE)) #Add noise to the images

    sample = (sample >> dt.NormalizeMinMax(0, 1)) # Normalize each image between 0 and 1.

    im = sample.update()()
    positions = im.get_property('position', get_one=False)
    radii = im.get_property('radius', get_one=False)
    
    label = np.zeros((IMAGE_SIZE, IMAGE_SIZE))
    

    for i in range(len(radii)):
        x, y = np.meshgrid(np.arange(IMAGE_SIZE), np.arange(IMAGE_SIZE))
        gauss_blob = np.exp(-((x-positions[i][1])**2+(y-positions[i][0])**2)/(2*(radii[i]*1e8 * RADIUS_FACTOR)**2))
        label += gauss_blob

    #types: deeptrack.image.Image, np.ndarray,list
    return im, label, positions

im,label,positions = generate_particles()