# Loading all necessary packages

import os
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
import io as python_io
import matplotlib.patches as patches
import cv2
import torch
import lpips
import requests
import itertools
import sys

from io import BytesIO
from itertools import product
from numba import njit
from pickle import TUPLE3
from PIL import Image
from pylops import Gradient, BlockDiag
from scipy.ndimage import gaussian_filter, median_filter, convolve
from skimage import img_as_float, io, util, color
from skimage.color import rgb2gray
from skimage.io import imread
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.restoration import estimate_sigma
from skimage.restoration import denoise_nl_means
from skimage.util import img_as_ubyte
from torchvision import transforms



# Load the LPIPS model, using GPU if available
# Parameter net='alex' is faster and smaller; net='vgg' is more accurate.

device = 'cuda' if torch.cuda.is_available() else 'cpu'
loss_fn = lpips.LPIPS(net='alex').to(device)
print("LPIPS import successful")



# Function for conversion of an image to a form readable by LPIPS (3 channels, rescaled to [-1,1])

def conversion_for_lpips(image):

    image_lpips_v1 = np.stack([image] * 3, axis=0)     # Shape becomes (3, H, W)
    image_lpips_v2 = torch.from_numpy(image_lpips_v1).float()   # Type conversion
    image_lpips_v2 = image_lpips_v2.unsqueeze(0)       # Shape becomes (1, 3, H, W)
    image_lpips_v2 = (image_lpips_v2 * 2.0) - 1.0      # Normalisation to [-1,1]
    image_lpips_final = image_lpips_v2.to(device)
    return image_lpips_final




# Loading clear images

# Add proper directory path

folder_path = '/clear_images'


if not os.path.isdir(folder_path):
    print("Error: Folder not found.")
else:

    all_files = os.listdir(folder_path)

    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff'}
    image_files = [f for f in all_files if os.path.splitext(f)[1].lower() in image_extensions]

    print(f"Found {len(image_files)} image files in the folder.")
    print("Image files:", image_files)

# Initialisation for a list of images
all_images = []

print("Loading images...")

for image_file in image_files:
    # Construct the full path to the image file
    full_path = os.path.join(folder_path, image_file)

    try:
        # Load the image
        image = io.imread(full_path)
        all_images.append(image)
        #print(f"Successfully loaded: {image_file} | Shape: {image.shape}")

    except Exception as e:
        print(f"Could not load {image_file}. Reason: {e}")

number_of_images = len(all_images)
print(f"Total images successfully loaded: {number_of_images}")




# Definition of helper functions for ROF and dpROF

# Helper functions: Gradient, Divergence, Norm

# Compute Gradient Magnitude
def compute_gradient_magnitude(img):
    gx = np.zeros_like(img)
    gy = np.zeros_like(img)
    gx[:, :-1] = img[:, 1:] - img[:, :-1]
    gy[:-1, :] = img[1:, :] - img[:-1, :]
    return np.sqrt(gx**2 + gy**2)

@njit
def gradient(u):
    grad_u = np.zeros(u.shape + (2,))
    grad_u[:-1,:,0] = u[1:,:] - u[:-1,:]
    grad_u[:,:-1,1] = u[:,1:] - u[:,:-1]
    return grad_u

@njit
def divergence(p):
    div = np.zeros(p.shape[:2])
    div[:-1,:] += p[:-1,:,0]
    div[1:,:]  -= p[:-1,:,0]
    div[:,:-1] += p[:,:-1,1]
    div[:,1:]  -= p[:,:-1,1]
    return div

@njit
def norm2(p):
    return np.sqrt(np.sum(p**2, axis=-1) + 1e-12)

# Original Chambolle-Pock, acclerated v1
@njit
def chambolle_pock(image, tau, lam=0.2, max_iter=10000, tol=1.0e-5):
    # Initialise the variables

    m, n = image.shape                      # Take the dimensions of the image
    p = np.zeros((m, n, 2))                 # The vector field
    g = np.zeros((m, n, 2))                 # The gradient
    x = image                               # Input image
    x_bar = x.copy()                        # Again the input image, because Chambolle-Pock tracks two copies of it

    # Set the values of auxiliary constants

    L = np.sqrt(8)
    sigma_cp = 1 / (tau * L**2)
    theta = 1

    # Start the algorithm

    for i in range(max_iter):

        # Compute the gradient g and update the vector field p

        g = gradient(x_bar)
        p_new = (p + sigma_cp * g)
        norm_p_1 = norm2(p_new)
        norm_p_2 = np.maximum(1.0,norm_p_1[..., None])
        p = (p_new / norm_p_2)

        # Compute divergence and update x

        div_p = divergence(p)
        x_prev = x.copy()
        x = ( x + tau * div_p + (tau / lam) * image  )/(1 + (tau / lam))

        # Update theta, tau, sigma and x_bar
        theta = 1 / np.sqrt(1 + tau /(2*lam))
        tau *= theta
        sigma_cp /= theta
        x_bar = x + theta * (x - x_prev)

        # Check convergence

        if np.linalg.norm((x - x_prev)) < tol:
            print ( f" Converged in { i + 1 } iterations. - classical ROF model (acc) " )
            break

    return x


#Original Chambolle-Pock, accelerated v2
@njit
def chambolle_pock_v(image, tau, lam=0.2, max_iter=10000, tol=1.0e-5):
    # Initialise the variables

    m, n = image.shape                      # Take the dimensions of the image
    p = np.zeros((m, n, 2))                 # The vector field
    g = np.zeros((m, n, 2))                 # The gradient
    x = image                               # Input image
    x_bar = x.copy()                        # Again the input image, because Chambolle-Pock tracks two copies of it

    # Set the values of auxiliary constants

    L = np.sqrt(8)
    sigma_cp = 1 / (tau * L**2)
    theta = 1

    # Start the algorithm

    for i in range(max_iter):

        # Compute the gradient g and update the vector field p

        g = gradient(x_bar)
        p_new = (p + sigma_cp * g)
        norm_p_1 = norm2(p_new)
        norm_p_2 = np.maximum(1.0,norm_p_1[..., None]/lam)
        p = (p_new / norm_p_2)

        # Compute divergence and update x

        div_p = divergence(p)
        x_prev = x.copy()
        x = ( x + tau * (div_p + image) )/(1 + tau)

        # Update theta, tau, sigma and x_bar
        theta = 1 / np.sqrt(1 + tau/2)
        tau *= theta
        sigma_cp /= theta
        x_bar = x + theta * (x - x_prev)

        # Check convergence

        if np.linalg.norm((x - x_prev)) < tol:
            #print ( f" Converged in { i + 1 } iterations. - classical ROF model (acc) " )
            break

    return x

# Average on ball kernel for mollification
def ball_kernel(radius):
    y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
    mask = x**2 + y**2 <= radius**2
    kernel = mask.astype(float)
    kernel /= np.sum(kernel)
    return kernel

def mollify_function(func, radius=2):
    kernel = ball_kernel(radius)
    padded_func = np.pad(func, pad_width=radius, mode='reflect')
    mollified = convolve(padded_func, kernel, mode='reflect')
    return mollified[radius:-radius, radius:-radius]

# Custom Resolvent
def custom_resolvent(p_tilde, a, sigma):
    norm_p = norm2(p_tilde)
    p = np.zeros_like(p_tilde)

    mask_zero = (a == 0)
    mask_small = (a > 0) & (norm_p <= 1)
    mask_large = (a > 0) & (norm_p > 1)

    p[mask_zero] = p_tilde[mask_zero] / np.maximum(1, norm_p[mask_zero])[..., None]
    p[mask_small] = p_tilde[mask_small]
    s = norm_p[mask_large]
    a_vals = a[mask_large]
    factor = (a_vals * s + sigma) / (a_vals * s + sigma * s)
    p[mask_large] = (factor[..., None]) * p_tilde[mask_large]

    return p

# Modified Chambolle-Pock, accelerated (adaptive double-phase) v1
def chambolle_pock_modified(image, a_weight, tau, lam=0.2, max_iter=10000, tol=1e-5):
    m, n = image.shape
    p = np.zeros((m, n, 2))
    g = np.zeros((m, n, 2))
    x = image.copy()
    x_bar = x.copy()


    # Set the values of auxiliary constants
    L = np.sqrt(8)
    sigma_cp = 1 / (tau * L**2)
    theta = 1


    for i in range(max_iter):

        # Compute the gradient g and update the vector field p
        g = gradient(x_bar)
        p_new = p + sigma_cp * g
        p = custom_resolvent(p_new, a_weight, sigma_cp)


        # Compute divergence and update x
        div_p = divergence(p)
        x_prev = x.copy()
        x = (x + tau * div_p + (tau / lam) * image) / (1 + (tau / lam))


        # Update theta, tau, sigma and x_bar

        theta = 1 / np.sqrt(1 + tau / (2*lam))
        tau *= theta
        sigma_cp /= theta
        x_bar = x + theta * (x - x_prev)

        if np.linalg.norm(x - x_prev) < tol:
            print(f"Converged in {i+1} iterations. - adaptive double-phase ROF model (acc)")
            break

    return x

# Custom Resolvent for dpROF
def custom_resolvent_v(p_tilde, a, sigma, lam):
    norm_p = norm2(p_tilde)
    p = np.zeros_like(p_tilde)

    mask_zero = (a == 0)
    mask_small = (a > 0) & (norm_p <= lam)
    mask_large = (a > 0) & (norm_p > lam)

    p[mask_zero] = p_tilde[mask_zero] / np.maximum(1, norm_p[mask_zero]/lam)[..., None]
    p[mask_small] = p_tilde[mask_small]
    s = norm_p[mask_large]
    a_vals = a[mask_large]
    factor = (sigma/s + a_vals) / (sigma/lam + a_vals)
    p[mask_large] = p_tilde[mask_large] * (factor[..., None])

    return p

# Modified Chambolle-Pock, accelerated (adaptive double-phase) v2
def chambolle_pock_modified_v(image, a_weight, tau, lam=0.2, max_iter=10000, tol=1e-5):
    m, n = image.shape
    p = np.zeros((m, n, 2))
    g = np.zeros((m, n, 2))
    x = image.copy()
    x_bar = x.copy()


    # Set the values of auxiliary constants

    L = np.sqrt(8)
    sigma_cp = 1 / (tau * L**2)
    theta = 1

    for i in range(max_iter):



        # Compute the gradient g and update the vector field p

        g = gradient(x_bar)
        p_new = p + sigma_cp * g
        p = custom_resolvent_v(p_new, a_weight, sigma_cp, lam)


        # Compute divergence and update x

        div_p = divergence(p)
        x_prev = x.copy()
        x = ( x + tau * (div_p + image) )/(1 + tau)

        # Update theta, tau, sigma and x_bar

        theta = 1 / np.sqrt(1 + tau/2)
        tau *= theta
        sigma_cp /= theta
        x_bar = x + theta * (x - x_prev)

        if np.linalg.norm(x - x_prev) < tol:
            #print(f"Converged in {i+1} iterations. - adaptive double-phase ROF model (acc)")
            break

    return x



# Huber ROF, accelerated
def chambolle_pock_huber_accelerated(image, alpha, tau, lam=0.2, max_iter=10000, tol=1e-8):

    # Initialise the variables

    m, n = image.shape
    p = np.zeros((m, n, 2))
    g = np.zeros((m, n, 2))
    x = image.copy()
    x_bar = x.copy()

    # Set the values of auxiliary constants

    L = np.sqrt(8)
    sigma = 1 / (2*tau * L**2)
    theta = 1

    for i in range(max_iter):

        # Compute the gradient g and update the vector field p

        g = gradient(x_bar)
        p_new = (p + sigma * g)
        norm_p_1 = norm2(p_new)
        norm_p_2 = (1 + sigma * alpha)*np.maximum(1.0,(1/(1 + sigma * alpha))*norm_p_1[..., None])
        p = (p_new / norm_p_2)

        # Compute divergence and update x

        div_p = divergence(p)
        x_prev = x.copy()
        x = (x + tau * div_p + (tau / lam) * image) / (1 + (tau / lam))

        # Update theta, tau, sigma and x_bar

        theta = 1/(np.sqrt(1 + 2*tau/lam))
        tau *= theta
        sigma /= theta

        x_bar = (1+theta)*x - theta*x_prev

        # Check convergence

        if np.linalg.norm(x - x_prev) < tol:
            print(f"Huber ROF (acc): {i+1} iterations")
            break

    return x


# Setting the values of tested parameters

# Noise level

given_variance = 0.001

# Initial ROF settings

lambda_initial_ROF = 0.02
given_precision_ROF = 1e-2
max_iter_ROF = 3000

# dpROF settings

given_precision_dpROF = 1e-4
max_iter_dpROF = 3000

# Tested values of lambda

set_of_lambdas = [0.015]

# Tested values of a

set_of_a = [70]

# Tested values of b/a

set_of_ba = [100]





# Initialisation of variables

global_time_start = time.time()

product_set = list(product(set_of_lambdas,set_of_a,set_of_ba))

number_of_parameters = len(set_of_lambdas)*len(set_of_a)*len(set_of_ba)

noise_technical_sum = 0
lpips_technical_sum_noisy_image = [[[0 for _ in range(len(set_of_ba))] for _ in range(len(set_of_a))] for _ in range(len(set_of_lambdas))]
lpips_technical_sum_noisy_ROF = [[[0 for _ in range(len(set_of_ba))] for _ in range(len(set_of_a))] for _ in range(len(set_of_lambdas))]
lpips_technical_sum_ROF_image = [[[0 for _ in range(len(set_of_ba))] for _ in range(len(set_of_a))] for _ in range(len(set_of_lambdas))]
lpips_technical_sum_dpROF_image = [[[0 for _ in range(len(set_of_ba))] for _ in range(len(set_of_a))] for _ in range(len(set_of_lambdas))]

ssim_sum_ROF = [[[0 for _ in range(len(set_of_ba))] for _ in range(len(set_of_a))] for _ in range(len(set_of_lambdas))]
psnr_sum_ROF = [[[0 for _ in range(len(set_of_ba))] for _ in range(len(set_of_a))] for _ in range(len(set_of_lambdas))]
ssim_sum_dpROF = [[[0 for _ in range(len(set_of_ba))] for _ in range(len(set_of_a))] for _ in range(len(set_of_lambdas))]
psnr_sum_dpROF = [[[0 for _ in range(len(set_of_ba))] for _ in range(len(set_of_a))] for _ in range(len(set_of_lambdas))]

iterations_number = 0

# Starting the automated loop for images

for (i,current_lambda,a_const,ba) in itertools.product(range(number_of_images), set_of_lambdas, set_of_a, set_of_ba):

  # Initialisation step: loading clear images and adding Gaussian noise

  image = all_images[i]

  # Handle different image formats
  if image.ndim == 2:
     pass  # Already grayscale
  elif image.ndim == 3:
      if image.shape[2] == 4:
         image = color.rgb2gray(color.rgba2rgb(image))  # Convert RGBA -> RGB -> grayscale
      elif image.shape[2] == 3:
          image = color.rgb2gray(image)  # Convert RGB -> grayscale
      else:
         raise ValueError(f"Unsupported channel format: image.shape = {image.shape}")
  else:
     raise ValueError(f"Unsupported image format: image.ndim = {image.ndim}")

  image = img_as_float(image)

  noisy = util.random_noise(image, mode='gaussian', var=given_variance)

  noise_level_estimate = estimate_sigma(noisy)

  lambdas_index = set_of_lambdas.index(current_lambda)
  a_index = set_of_a.index(a_const)
  b_index = set_of_ba.index(ba)

  noise_technical_sum += noise_level_estimate / (number_of_images*number_of_parameters)

  # Step 1: Initial classical ROF

  start = time.time()
  denoised_v1v = chambolle_pock_v(noisy, tau=0.25, lam=lambda_initial_ROF, max_iter=max_iter_ROF, tol=given_precision_ROF)
  t1 = time.time() - start


  # Step 2: Compute weight from gradient norm

  mollified_v1 = mollify_function(denoised_v1v, radius=2)
  grad_v1 = gradient(mollified_v1)
  grad_norm = norm2(grad_v1)

  a_weight = np.maximum(0, a_const - ba * a_const * np.maximum(grad_norm, a_const / (2 * ba * a_const)))

  # Step 3: Apply modified Chambolle-Pock algorithm for adaptive double-phase ROF model

  start = time.time()
  denoised_v2v = chambolle_pock_modified_v(noisy, a_weight, tau=0.25, lam=current_lambda, max_iter=max_iter_dpROF, tol=given_precision_dpROF)
  t2 = time.time() - start





  # Computing LPIPS for all relevant images

  # Normalisation for LPIPS (different image dimensions, rescaled to [-1,1])

  image_lpips = conversion_for_lpips(image)

  ROF_lpips = conversion_for_lpips(denoised_v1v)

  dpROF_lpips = conversion_for_lpips(denoised_v2v)



  # Applying LPIPS for ROF

  with torch.no_grad():
     distance = loss_fn(image_lpips, ROF_lpips)

  score = distance.item()
  lpips_technical_sum_ROF_image[lambdas_index][a_index][b_index] += score / number_of_images


  # Applying LPIPS for dpROF

  with torch.no_grad():
     distance = loss_fn(image_lpips, dpROF_lpips)

  score = distance.item()
  lpips_technical_sum_dpROF_image[lambdas_index][a_index][b_index] += score / number_of_images


  # Now SSIM and PSNR

  ssim_val = ssim(image, denoised_v1v, data_range=1)
  psnr_val = psnr(image, denoised_v1v, data_range=1)

  ssim_sum_ROF[lambdas_index][a_index][b_index] += float(ssim_val) / number_of_images
  psnr_sum_ROF[lambdas_index][a_index][b_index] += float(psnr_val) / number_of_images

  ssim_val = ssim(image, denoised_v2v, data_range=1)
  psnr_val = psnr(image, denoised_v2v, data_range=1)

  ssim_sum_dpROF[lambdas_index][a_index][b_index] += float(ssim_val) / number_of_images
  psnr_sum_dpROF[lambdas_index][a_index][b_index] += float(psnr_val) / number_of_images

  iterations_number += 1

  if iterations_number % 10 == 0:
            print(f"{iterations_number}/{number_of_images*len(set_of_lambdas)*len(set_of_a)*len(set_of_ba)} iterations.")

print()

global_time_total = time.time() - global_time_start

print(f"Total elapsed time: {global_time_total:.2f}s \n")

print()

print(product_set)

print()

print(f"Average noise level: {noise_technical_sum}")

print()

print(f"Average LPIPS, ROF-image: {lpips_technical_sum_ROF_image}")

print()

print(f"Average LPIPS, dpROF-image: {lpips_technical_sum_dpROF_image}")

print()

print(f"Average SSIM, ROF: {ssim_sum_ROF}")

print()

print(f"Average SSIM, dpROF: {ssim_sum_dpROF}")

print()

print(f"Average PSNR, ROF: {psnr_sum_ROF}")

print()

print(f"Average PSNR, dpROF: {psnr_sum_dpROF}")

print()



def best_for_lpips_dpROF(params):
    current_lambda, a_const, ba = params

    lambdas_index = set_of_lambdas.index(current_lambda)
    a_index = set_of_a.index(a_const)
    b_index = set_of_ba.index(ba)

    score = lpips_technical_sum_dpROF_image[lambdas_index][a_index][b_index]
    return score

best_lpips = min(product_set, key=best_for_lpips_dpROF)
best_lpips_value = best_for_lpips_dpROF(best_lpips)

print(f"Optimal parameters using LPIPS: {best_lpips}")
print(f"Optimal value of LPIPS: {best_lpips_value}")
print()


def best_for_ssim_dpROF(params):
    current_lambda, a_const, ba = params

    lambdas_index = set_of_lambdas.index(current_lambda)
    a_index = set_of_a.index(a_const)
    b_index = set_of_ba.index(ba)

    score = ssim_sum_dpROF[lambdas_index][a_index][b_index]
    return score

best_ssim = max(product_set, key=best_for_ssim_dpROF)
best_ssim_value = best_for_ssim_dpROF(best_ssim)

print(f"Optimal parameters using SSIM: {best_ssim}")
print(f"Optimal value of SSIM: {best_ssim_value}")
print()



def best_for_psnr_dpROF(params):
    current_lambda, a_const, ba = params

    lambdas_index = set_of_lambdas.index(current_lambda)
    a_index = set_of_a.index(a_const)
    b_index = set_of_ba.index(ba)

    score = psnr_sum_dpROF[lambdas_index][a_index][b_index]
    return score

best_psnr = max(product_set, key=best_for_psnr_dpROF)
best_psnr_value = best_for_psnr_dpROF(best_psnr)

print(f"Optimal parameters using PSNR: {best_psnr}")
print(f"Optimal value of PSNR: {best_psnr_value}")
print()



# Adding outputs to a log file

# Add proper file path

log_file = '/dpROF_optimisation.log'

with open(log_file, 'a') as f:
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    f.write(f"TIME: [{timestamp}]\n\n")

    f.write(f"Total elapsed time: {global_time_total:.2f}s \n\n")

    f.write(f"Prescribed noise variance: {given_variance}\n")
    f.write(f"Estimated average noise level: {noise_technical_sum}\n\n")

    f.write(f"Considered lambdas: {set_of_lambdas}\n")
    f.write(f"Considered a: {set_of_a}\n")
    f.write(f"Considered b: {set_of_ba}\n")

    f.write(f"Optimal parameters using LPIPS: {best_lpips}\n")
    f.write(f"Optimal value of LPIPS: {best_lpips_value}\n\n")

    f.write(f"Optimal parameters using SSIM: {best_ssim} \n")
    f.write(f"Optimal value of SSIM: {best_ssim_value} \n\n")

    f.write(f"Optimal parameters using PSNR: {best_psnr} \n")
    f.write(f"Optimal value of PSNR: {best_psnr_value} \n\n")

    f.write("======================================\n\n")

print(f"Successfully added to {log_file}")





# Adding extended outputs to a supplemental log file

# Add proper file path

log_file = '/dpROF_optimisation_extendedlog.log'

with open(log_file, 'a') as f:
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    f.write(f"TIME: [{timestamp}]\n\n")

    f.write(f"Total elapsed time: {global_time_total:.2f}s \n\n")

    f.write(f"{product_set} \n")

    f.write(f"Average noise level: {noise_technical_sum} \n")

    f.write(f"Average LPIPS, ROF-image: {lpips_technical_sum_ROF_image} \n")

    f.write(f"Average LPIPS, dpROF-image: {lpips_technical_sum_dpROF_image} \n")

    f.write(f"Average SSIM, ROF: {ssim_sum_ROF} \n")

    f.write(f"Average SSIM, dpROF: {ssim_sum_dpROF} \n")

    f.write(f"Average PSNR, ROF: {psnr_sum_ROF} \n")

    f.write(f"Average PSNR, dpROF: {psnr_sum_dpROF}\n\n")

    f.write(f"Prescribed noise variance: {given_variance}\n")
    f.write(f"Estimated average noise level: {noise_technical_sum}\n\n")

    f.write(f"Considered lambdas: {set_of_lambdas}\n")
    f.write(f"Considered a: {set_of_a}\n")
    f.write(f"Considered b: {set_of_ba}\n")

    f.write(f"Optimal parameters using LPIPS: {best_lpips}\n")
    f.write(f"Optimal value of LPIPS: {best_lpips_value}\n\n")

    f.write(f"Optimal parameters using SSIM: {best_ssim} \n")
    f.write(f"Optimal value of SSIM: {best_ssim_value} \n\n")

    f.write(f"Optimal parameters using PSNR: {best_psnr} \n")
    f.write(f"Optimal value of PSNR: {best_psnr_value} \n\n")

    f.write("======================================\n\n")

print(f"Successfully added to {log_file}")



