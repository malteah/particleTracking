import deeptrack as dt

import numpy as np
import matplotlib.pyplot as plt


# Define the particle
particle = dt.PointParticle(                                         
    intensity=100,
    position=lambda: np.random.rand(2) * 256
)


# Define the optical system
fluorescence_microscope = dt.Fluorescence(
    NA=0.7,                
    resolution=1e-6,     
    magnification=10,
    wavelength=680e-9,
    output_region=(0, 0, 256, 256)
)


# Define noises
offset = dt.Add(
    value=lambda: np.random.rand()*1
)

poisson_noise = dt.Poisson(
    snr=lambda: np.random.rand()*7 + 3,
    background=offset.value
)


# Define the image features
num_particles = lambda: np.random.randint(1, 11)

image_features = fluorescence_microscope(particle^num_particles)


# Plot example images
for i in range(4):
    image_features.update()
    output_image = image_features.plot(cmap="gray")


# Create the target images
    
    # Creates an image with circles of radius two at the same position 
    # as the particles in the input image.
CIRCLE_RADIUS = 3

def get_target_image(image_of_particles):
    target_image = np.zeros(image_of_particles.shape)
    X, Y = np.meshgrid(
        np.arange(0, image_of_particles.shape[0]), 
        np.arange(0, image_of_particles.shape[1])
    )

    for property in image_of_particles.properties:
        if "position" in property:
            position = property["position"]

            distance_map = (X - position[1])**2 + (Y - position[0])**2
            target_image[distance_map < CIRCLE_RADIUS**2] = 1
    
    return target_image

for i in range(4):
    image_features.update()
    image_of_particles = image_features.resolve()

    target_image = get_target_image(image_of_particles)

    # plt.subplot(1,2,1)
    # plt.imshow(np.squeeze(image_of_particles), cmap="gray")
    # plt.title("Input Image")
    
    # plt.subplot(1,2,2)
    # plt.imshow(np.squeeze(target_image), cmap="gray")
    # plt.title("Target image")
    
    # plt.show()


# Define image generator
generator = dt.generators.ContinuousGenerator(
    image_features, 
    get_target_image,
    batch_size=8,
    min_data_size=256,
    max_data_size=512
)

# Define the neural network model
model = dt.models.unet(
    (256, 256, 1), 
    conv_layers_dimensions=[8, 16, 32],
    base_conv_layers_dimensions=[32, 32], 
    loss=dt.losses.weighted_crossentropy((10, 1)),
    output_activation="sigmoid"
)

# model.summary()

with generator:
    model.fit(
        generator, 
        epochs=50,
    )

# generator[0] grabs a single batch from the generator
input_image, target_image = generator[0]

for i in range(input_image.shape[0]):
    
    predicted_image = model.predict(input_image)
    plt.figure(figsize=(15, 5))
    plt.subplot(1,3,1)
    plt.imshow(np.squeeze(input_image[i, :, :, 0]), cmap="gray")
    plt.title("Input Image")

    plt.subplot(1,3,2)
    plt.imshow(np.squeeze(predicted_image[i, :, :, 0]), cmap="gray")
    plt.title("Predicted Image")
    
    plt.subplot(1,3,3)
    plt.imshow(np.squeeze(target_image[i, :, :, 0] > 0.5), cmap="gray")
    plt.title("Target image")

    plt.show()