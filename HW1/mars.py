import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

#read image
mars_gray = mpimg.imread('mars.png')


#convert to luminence
lum_img = mars_gray[:, :, 0]

#colormap
cdict = {'red':   [[0.0,  0.0, 0.0],
                   [0.2,  0.0, 0.14],
                   [0.75,  1.0, 1.0],
                   [1.0,  0.7, 0.7]],
         'green': [[0.0,  0.0, 0.14],
                   [0.24, 0.14, 0.24],
                   [0.3, 0.7, 0.7],
                   [0.4,  0.3, 0.0],
                   [1.0, 0.0, 0.0]],
         'blue':  [[0.0,  1.0, 1.0],
                   [0.28,  0.0, 0.0],
                   [1.0,  0.0, 0.0]]}
mars_colormap = LinearSegmentedColormap('mars', segmentdata=cdict, N=256)

#apply colormap
mars_red = mars_colormap(lum_img)

plt.matshow(mars_red)
plt.show()