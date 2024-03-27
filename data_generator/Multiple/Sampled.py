# Import libraries
from vpython import *
import numpy as np
import povexport

# Positions
x_min, y_min, z_min = -10, -10, -10
x_max, y_max, z_max = 10, 10, 10

# Rotations
rx_min, ry_min, rz_min = -pi, -pi, -pi
rx_max, ry_max, rz_max = pi, pi, pi

scene = canvas(width=640, height=640)
# Set up scene
scene.scale = 1

scene.camera.pos = vector(0, 0, 20)
[distant_light(direction=vector(0.22, 0.44, 0.88), color=color.gray(0.8)),
 distant_light(direction=vector(-0.88, -0.22, -0.44), color=color.gray(0.5))]

# Define objects
Objects = [
    pyramid(size=vector(5,5,5), visible=False),
    box(size=vector(5,5,5), visible=False),
    sphere(radius=2.5, visible=False)
]

# Sample size
sample_size = 100000

for i in range(sample_size):
    scene.scale = 1
    scene.camera.pos = vector(0, 0, 20)

    ind = np.random.randint(3)
    # Sample object
    obj = Objects[ind]

    if ind == 2:
        obj.radius = 2.5
    else:
        obj.size = vector(5, 5, 5)
    
    obj.visible = True

    # Sample colour
    obj.color = vector(np.random.rand(), np.random.rand(), np.random.rand())

    # Sample position
    x = np.random.uniform(x_min, x_max)
    y = np.random.uniform(y_min, y_max)
    z = np.random.uniform(z_min, z_max)
    obj.pos = vector(x, y, z)
    
    # Sample orientation
    rx = np.random.uniform(rx_min, rx_max)
    obj.rotate(angle=rx, axis=vector(1,0,0), origin=obj.pos)    
    ry = np.random.uniform(ry_min, ry_max)
    obj.rotate(angle=ry, axis=vector(0,1,0), origin=obj.pos)    
    rz = np.random.uniform(rz_min, rz_max)
    obj.rotate(angle=rz, axis=vector(0,0,1), origin=obj.pos)
    
    # POV-Export
    count = i + 1
    povexport.export(filename="./Data/3dSprites%06d.pov" % count)
    
    # Make object invisible again
    obj.visible = False
