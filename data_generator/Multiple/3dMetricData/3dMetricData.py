# Import libraries
from vpython import *
import numpy as np
import povexport
import sys

scene = canvas(width=640, height=640)
# Set up scene
scene.scale = 1

scene.camera.pos = vector(0, 0, 20)
[distant_light(direction=vector(0.22, 0.44, 0.88), color=color.gray(0.8)),
 distant_light(direction=vector(-0.88, -0.22, -0.44), color=color.gray(0.5))]

# Define objects
Objects = [
    pyramid(size=vector(7,7,7), visible=False),
    box(size=vector(7,7,7), visible=False),
    ellipsoid(size=vector(12,6,6), visible=False)
]

def create_image(colour, shape, orientation, pos, i, k):
    # Sample object
    obj = Objects[shape]

    obj.visible = True

    # Sample colour
    obj.color = vector(colour[0], colour[1], colour[2])

    # Sample position
    obj.pos = vector(pos[0], pos[1], pos[2])

    # Sample orientation
    obj.rotate(angle=orientation[0], axis=vector(1,0,0), origin=obj.pos)
    obj.rotate(angle=orientation[1], axis=vector(0,1,0), origin=obj.pos)
    obj.rotate(angle=orientation[2], axis=vector(0,0,1), origin=obj.pos)

    # POV-Export
    povexport.export(filename="./MetricData{0}/3dMetricData%06d.pov".format(k) % i)

    # Make object invisible again
    obj.visible = False

params_sampler = [
    lambda: np.random.rand(), # red
    lambda: np.random.rand(), # green
    lambda: np.random.rand(), # blue
    lambda: np.random.randint(3), # shape
    lambda: np.random.rand() * 2 * np.pi - np.pi, # rx
    lambda: np.random.rand() * 2 * np.pi - np.pi, # ry
    lambda: np.random.rand() * 2 * np.pi - np.pi, # rz
    lambda: 7 * np.random.rand(), # pos_x
    lambda: 7 * np.random.rand(), # pos_y
    lambda: 7 * np.random.rand() # pos_z
]

def sample_params():
    params = []
    for p in range(len(params_sampler)):
        params.append(params_sampler[p]())
    return params

sample_size = 400
vote_size = 80
no_params = len(params_sampler)

i = 0
k = int(sys.argv[1])
print(k)
latents = []

if not os.path.isdir('./MetricData{0}'.format(k)):
    print('Path does not exist. Creating path... ./MetricData{0}'.format(k))
    os.makedirs(
        os.path.join('./MetricData{0}'.format(k)))    

for vote in range(vote_size):
    params_k = params_sampler[k]()
    
    for s in range(sample_size):
        params = sample_params()
        params[k] = params_k

        i += 1
        create_image(
            colour=(params[0],
                    params[1],
                    params[2]),
            shape=params[3],
            orientation=(
                params[4],
                params[5],
                params[6]),
            pos=(
                params[7],
                params[8],
                params[9]),
            i=i,
            k=k
        )
        latents.append(
            {
                'vote': vote,
                'latent_index': k,
                'sample_index': s
            }
        )
np.save('Latents_{0}'.format(k), latents)
