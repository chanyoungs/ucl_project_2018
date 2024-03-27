import cv2
import numpy as np
import matplotlib.pyplot as plt

def create_image(colour=None, shape=0, scale=1, orientation=0, pos_x=0, pos_y=0, plot=False):
  size = 10
  border = 16
  
  # Deal with BW vs Colour
  if colour:
    background = np.zeros((64,64,3))
  else:
    background = np.zeros((64,64), np.uint8)
    colour = 255
  
  # Choose shape
  if shape == 0: # Ellipse
    img_sha = cv2.ellipse(background, (32, 32), (int(scale*1.4*size), int(scale*0.6*size)), 0, 0, 360, colour, -1)
  elif shape == 1: # Square
    img_sha = cv2.rectangle(background, (int(32-scale*size), int(32-scale*size)), (int(32+scale*size), int(32+scale*size)), colour, -1)
  elif shape == 2: # Triangle
    size *= 1.3
    pts = np.array(
        [
            [32, 32 - scale * size],
            [32 - scale * size * np.sqrt(3) * 0.5, 32 + scale * size * 0.5],
            [32 + scale * size * np.sqrt(3) * 0.5, 32 + scale * size * 0.5]
        ], np.int32
    ).reshape((-1, 1, 2))    
    img_sha = cv2.fillPoly(background, [pts], colour)
  
  # Rotate
  M_rot = cv2.getRotationMatrix2D((32, 32), orientation * 180 / np.pi,1)
  img_rot = cv2.warpAffine(img_sha, M_rot, (64, 64))
  
  # Translate
  M_tra = np.float32([[1, 0, border - 32 + pos_x * 2 * (32 - border)],
                      [0, 1, border - 32 + pos_y * 2 * (32 - border)]])
  img_tra = cv2.warpAffine(img_rot, M_tra, (64, 64))

  if plot:
    plt.figure(facecolor='grey')
    plt.imshow(img_tra)
    plt.grid('off')
    plt.show()
  
  return img_tra

total = 1000000
Data = np.zeros((total, 64, 64, 3))
Latents = np.zeros((total, 8))
percent = 0

for i in range(total):
  red = np.random.rand()
  blue = np.random.rand()
  green = np.random.rand()
  shape = np.random.randint(3)
  scale = np.random.rand() / 2 + 0.5
  orientation = np.random.rand() * 2 * np.pi
  pos_x = np.random.rand()
  pos_y = np.random.rand()
  
  Data[i] = create_image(colour=(red, green, blue), shape=shape, scale=scale, orientation=orientation, pos_x=pos_x, pos_y=pos_y)
  Latents[i] = [red, blue, green, shape, scale, orientation, pos_x, pos_y]
  if i/total * 100 > percent:
    percent += 1
    print("{0}% complete...".format(percent))

np.save('2dSprites', Data)
print('2dSprites saved!')
np.save('Latents', Latents)
print('Latents saved!')
