from skimage import io, color, feature, transform, filters
import matplotlib.pyplot as plt
import numpy as np
import time


"""
Mathematical convolution, slide kernel over image
"""
def oneD_convolution(kernel, row):
    return np.convolve(kernel, row, mode='same')
        
    
"""
Sobel Filter
"""
def sobel(gray_image):
    #Sobel in the x direction, use two separable 1d convolutions instead of 3x3 2d convolution for performance purposes
    sobel_x = np.zeros_like(gray_image)
    for i in range(gray_image.shape[0]):
        sobel_x[i] = oneD_convolution([1,0,-1], gray_image[i])
    for j in range(gray_image.shape[1]):
        sobel_x[:,j] = oneD_convolution([1,2,1], sobel_x[:,j])

    
    #Sobel in the y direction
    sobel_y = np.zeros_like(gray_image)
    for i in range(gray_image.shape[0]):
        sobel_y[i] = oneD_convolution([1,2,1], gray_image[i])
    for j in range(gray_image.shape[1]):
        sobel_y[:,j] = oneD_convolution([1,0,-1], sobel_y[:,j])

    #calculate gradient magnitude and angle
    magnitude = np.sqrt(sobel_x * sobel_x + sobel_y * sobel_y)
    angle = np.arctan2(sobel_y, sobel_x)
    
    return (magnitude, angle)


"""
Guassian Filter
"""
def guassian_filter(gray_image):
    #Sobel in the x direction, use two separable 1d convolutions instead of 3x3 2d convolution for performance purposes
    filtered = np.zeros_like(gray_image)
    for i in range(gray_image.shape[0]):
        filtered[i] = oneD_convolution([.25,.5,.25], gray_image[i])
    for j in range(gray_image.shape[1]):
        filtered[:,j] = oneD_convolution([.25,.5,.25], filtered[:,j])
        
    return filtered
    
"""
Non-maximum Suppression
"""
def non_max_suppression(magnitude, angle):
    suppressed = np.zeros_like(magnitude)
    for i in range(1,angle.shape[0]-1):
        for j in range(1,angle.shape[1]-1):
        
            #get direction of gradient
            x = int(np.rint(np.cos(angle[i][j])))
            y = int(np.rint(np.sin(angle[i][j])))
                        
            #get neighborhood of gradient
            neighborhood = [magnitude[i][j]]
            #if (0 <= i+x < magnitude.shape[0]) and (0 <= j+y < magnitude.shape[1]):
            neighborhood.append(magnitude[i+y, j+x])
            #if (0 <= i-x < magnitude.shape[0]) and (0 <= j-y < magnitude.shape[1]):
            neighborhood.append(magnitude[i-y, j-x])
                
            #check if i,j is local maximum. If not set its magnitude to zero
            if magnitude[i][j] == max(neighborhood):
                suppressed[i][j] = magnitude[i][j]
    
    return suppressed      


def hysteresis_thresholding(gradient):
    #set low and high thresholds for hysteresis
    gradient_max = np.amax(gradient)
    high_thres = .09 * gradient_max
    low_thres = .05 * high_thres
    
    #determine strong and weak edges
    for i in range(1,gradient.shape[0]-1):
        for j in range(1,gradient.shape[1]-1):
            #strong edge
            if gradient[i][j] > high_thres:
                gradient[i][j] = 255
            
            #discard this edge
            elif gradient[i][j] < low_thres:
                gradient[i][j] = 0
            
            #weak edge check its neighbors
            else:
                if np.any(np.array([gradient[i-1][j-1], gradient[i-1][j], gradient[i-1][j+1], gradient[i][j+1], 
                gradient[i+1][j+1], gradient[i+1][j], gradient[i+1][j-1], gradient[i][j-1]]) > high_thres):
                    gradient[i][j] = 255
                else:
                    gradient[i][j] = 0
                    
                    
    return gradient
                
    

#Read image
print("Reading Image...")
current = time.time()   
image = io.imread('vessel_1.jpg')
gray_image = color.rgb2gray(image)
print("Time Elapsed {0}\n".format(time.time()-current))

#downscale image
print("Downscaling Image...")
current = time.time()   
scaling_factor = np.sqrt(500000/(gray_image.shape[0] * gray_image.shape[1]))
downscaled = transform.rescale(gray_image, scaling_factor)
print("Time Elapsed {0}\n".format(time.time()-current))

###Run Canny algorithm
## Noise Reduction
print("Applying Guassian Filter...")
current = time.time() 
noise_reduced = guassian_filter(downscaled)
#noise_reduced = filters.gaussian(downscaled)
print("Time Elapsed {0}\n".format(time.time()-current))

## Intensity gradient
print("Applying Sobel Filter...")
current = time.time() 
gradient_magnitude, gradient_angle = sobel(noise_reduced)
print("Time Elapsed {0}\n".format(time.time()-current))

## Non-maximum suppression
print("Applying Non-max Suppression...")
current = time.time()
suppressed = non_max_suppression(gradient_magnitude, gradient_angle)
print("Time Elapsed {0}\n".format(time.time()-current))

## Hysteresis Thresholding
print("Applying Hysteresis Thresholding...")
current = time.time()
intensity = hysteresis_thresholding(suppressed)
print("Time Elapsed {0}\n".format(time.time()-current))

## Plot results
fig, axes = plt.subplots(1, 3, figsize=(8, 4))
ax = axes.ravel()

ax[0].imshow(downscaled, cmap=plt.cm.gray)
ax[0].set_title("Grayscale")
ax[1].imshow(intensity, cmap=plt.cm.gray)
ax[1].set_title("My Sobel")
ax[2].imshow(feature.canny(downscaled), cmap=plt.cm.gray)
ax[2].set_title("Scikit Image Sobel")

fig.tight_layout()
plt.show()
