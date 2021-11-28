import matplotlib.image as mpimg
import matplotlib.pyplot as plt

import numpy as np

mario = mpimg.imread("mario_big.png")

#out = np.where(np.absolute(mario - np.array([1,0,0,1]))[0] < 0.01, [0,1,0,1], mario)
"""
print(len(mario))
print(mario.shape)
for i in range(mario.shape[0]):
    for j in range(mario.shape[1]):
        if np.allclose(mario[i][j], np.array([1,0,0,1]), atol = 0.05):
            mario[i][j] = np.array([0,1,0,1])
"""

luigi = np.where(np.allclose(mario, np.array([1,0,0,1]), atol = 0.05), [0,1,0,1], mario)

plt.imshow(luigi)
plt.show()