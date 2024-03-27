# Import libraries
from vpython import *
import numpy as np
import povexport
import os
import sys

if len(sys.argv) == 2:
    n = sys.argv[1]
else:
    n = 0
    while os.path.isdir('./data%d' % (n+1)):
        n += 1
if os.path.isfile('./data%d/3dSprites100000.pov' % n):
    n += 1
    m = 1
elif len(sys.argv) == 3:
    m = sys.argv[2]
else:
    m = 1
    while os.path.isfile('./data%d/3dSprites%06d.pov' % (n, m+1)):
        m += 1

print('Creating from folder \'./data%d\' and from file \'3dSprites%06d.pov\'' % (n, m))

# Positions
x_min, y_min, z_min = -7, -7, -7
x_max, y_max, z_max = 7, 7, 7

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
    pyramid(size=vector(7,7,7), visible=False),
    box(size=vector(7,7,7), visible=False),
    ellipsoid(size=vector(12,6,6), visible=False)
]

# Sample size
sample_size = 100000


path = 'data{0}'.format(n)
if not os.path.isdir(path):
    print('Creating path... ./'+path)
    os.makedirs(path)
else:
    print('Path ./'+path+' already exists...')

for i in range(m, sample_size+1):
    ind = np.random.randint(3)
    # Sample object
    obj = Objects[ind]

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
    povexport.export(filename="./"+path+"/3dSprites%06d.pov" % i)

    # Make object invisible again
    obj.visible = False
