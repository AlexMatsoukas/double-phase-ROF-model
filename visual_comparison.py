# Installations of specific versions/packages required for the part of the code concerning Chen-Levine-Rao and total generalised variation

#!pip install -q --upgrade pip
#!pip install -q "numpy>=2.3,<2.4" "scipy>=1.15,<1.17" matplotlib scikit-image pylops odl

# Add proper directory path before installation

#!rm -rf /content/recon
#!git clone -q https://github.com/lucasplagwitz/recon.git /content/recon



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



# Loading the previously installed git repository

# Add proper directory path

REPO_ROOT = "/content/recon"
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

for mod in list(sys.modules):
    if mod == "recon" or mod.startswith("recon.") or mod == "terms" or mod.startswith("terms."):
        del sys.modules[mod]

from recon.terms import IndicatorL2, DatanormL2, DatanormL2Bregman



# Load the LPIPS model, using GPU if available
# Parameter net='alex' is faster and smaller; net='vgg' is more accurate.

device = 'cuda' if torch.cuda.is_available() else 'cpu'
loss_fn = lpips.LPIPS(net='alex').to(device)
print("LPIPS import successful")


# Function for conversion of a float image to a form readable by LPIPS (3 channels, rescaled to [-1,1])

def conversion_for_lpips(image):

    image_lpips_v1 = np.stack([image] * 3, axis=0)     # Shape becomes (3, H, W)
    image_lpips_v2 = torch.from_numpy(image_lpips_v1).float()   # Type conversion
    image_lpips_v2 = image_lpips_v2.unsqueeze(0)       # Shape becomes (1, 3, H, W)
    image_lpips_v2 = (image_lpips_v2 * 2.0) - 1.0      # Normalisation to [-1,1]
    image_lpips_final = image_lpips_v2.to(device)
    return image_lpips_final




# Loading a clear image

# Add proper directory path

folder_path = '/clear_images'


if not os.path.isdir(folder_path):
    print("Error: Folder not found.")
else:

    all_files = os.listdir(folder_path)

    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff'}
    image_files = [f for f in all_files if os.path.splitext(f)[1].lower() in image_extensions]
    image_files = sorted(image_files)

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





# Definition of helper functions for all the considered models

# Functions for total generalised variation

class PdHgmTGV:
    def __init__(
        self,
        lam=1.0,
        alpha=(1.0, 1.0),
        tol=1e-4,
        mode="tv",
        pk=None,
        prox_param=1 / np.sqrt(12),
        max_iter=3000,
        verbose=False,
    ):
        self.lam = lam
        self.alpha = alpha
        self.tol = tol
        self.mode = mode
        self.pk = pk
        self.sigma = prox_param
        self.tau = prox_param
        self.max_iter = max_iter
        self.verbose = verbose
        self.k = 1

    def solve(self, f):
        self.k = 1

        if f.ndim != 2:
            raise ValueError(f"Only 2D images supported. Got shape {f.shape}")

        n, m = f.shape
        N = n * m

        grad = Gradient(
            dims=(n, m),
            dtype="float64",
            edge=True,
            kind="backward",
        )
        grad_v = BlockDiag([grad, grad])

        u = f.ravel().astype(np.float64).copy()
        u_bar = u.copy()

        v = np.zeros(2 * N, dtype=np.float64)
        v_bar = v.copy()

        p = np.zeros_like(grad * u_bar, dtype=np.float64)
        q = np.zeros_like(grad_v * v_bar, dtype=np.float64)

        proj_p = IndicatorL2((n, m), upper_bound=self.alpha[0])
        proj_q = IndicatorL2((2 * n, m), upper_bound=self.alpha[1])

        if self.mode == "tv":
            dataterm = DatanormL2(
                image_size=f.shape,
                data=f.ravel(),
                prox_param=self.tau,
                lam=self.lam,
            )
        else:
            dataterm = DatanormL2Bregman(
                image_size=f.shape,
                data=f.ravel(),
                prox_param=self.tau,
                lam=self.lam,
            )
            dataterm.pk = self.pk
            dataterm.bregman_weight_alpha = self.alpha[0]

        sens = np.inf

        while (sens > self.tol or self.k == 1) and (self.k <= self.max_iter):
            p = proj_p.prox(p + self.sigma * (grad * u_bar - v_bar))
            q = proj_q.prox(q + self.sigma * (grad_v * v_bar))

            u_old = u.copy()
            v_old = v.copy()

            u = dataterm.prox(u - self.tau * (grad.H * p))
            u_bar = 2 * u - u_old

            v = v + self.tau * (p - grad_v.H * q)
            v_bar = 2 * v - v_old

            if self.k % 300 == 0:
                u_gap = u - u_old
                v_gap = v - v_old

                denom_u = max(np.linalg.norm(u, 2), 1e-12)
                denom_v = max(np.linalg.norm(v, 2), 1e-12)

                sens = 0.5 * (
                    np.linalg.norm(u_gap - self.tau * (grad.H * v_gap), 2) / denom_u
                    + np.linalg.norm(v_gap - self.sigma * (grad * u_gap), 2) / denom_v
                )

                if self.verbose:
                    print(f"iter={self.k:4d}, sens={sens:.6e}")

            self.k += 1

        return u.reshape(n, m)



# Functions for Chen-Levine-Rao

# Forward / backward finite differences (homogeneous Neumann boundary conditions)
def Dpx(u):
    d = np.roll(u, -1, axis=1) - u
    d[:, -1] = 0
    return d

def Dmx(u):
    d = u - np.roll(u, 1, axis=1)
    d[:, 0] = 0
    return d

def Dpy(u):
    d = np.roll(u, -1, axis=0) - u
    d[-1, :] = 0
    return d

def Dmy(u):
    d = u - np.roll(u, 1, axis=0)
    d[0, :] = 0
    return d


# Minmod operator
def minmod(a, b):
    return 0.5 * (np.sign(a) + np.sign(b)) * np.minimum(np.abs(a), np.abs(b))


def minmodgradients(u):
    u_xp = Dpx(u)
    u_yp = Dpy(u)
    u_xm = Dmx(u)
    u_ym = Dmy(u)

    Dx = np.sqrt(u_xp**2 + minmod(u_yp, u_ym)**2 + 1e-8)
    Dy = np.sqrt(u_yp**2 + minmod(u_xp, u_xm)**2 + 1e-8)

    return Dx, Dy, u_xp, u_yp


# Chen-Levine-Rao exponent from mollified noisy datum
def exponent_qx(f, sigma=0.5, K=0.0075):
    mollified_datum = gaussian_filter(f, sigma)
    gx = Dpx(mollified_datum)
    gy = Dpy(mollified_datum)
    g2 = np.sqrt(gx**2 + gy**2)
    q = 1.0 + 1.0 / (1.0 + K * (np.abs(g2) ** 2))
    return q


def clr_split_divergence2(u, q, beta, eps=1e-8):
    Dx, Dy, u_xp, u_yp = minmodgradients(u)
    g = np.sqrt(u_xp**2 + u_yp**2 + eps)

    # Threshold for gradient magnitude
    mask_threshold = (g <= beta)

    # piecewise flux
    vx = np.where(mask_threshold, u_xp / (Dx**(2.0 - q)), u_xp / Dx)
    vy = np.where(mask_threshold, u_yp / (Dy**(2.0 - q)), u_yp / Dy)

    # divergence
    return Dmx(vx) + Dmy(vy), mask_threshold


def clr_split_flow2(noisy, tau=0.05, lam=0.05, K=0.0075, sigma=0.5,
                    beta=50.0, max_iter=2000, tol=1e-4, eps=1e-8):

    f = noisy
    q = exponent_qx(f, sigma=sigma, K=K)

    u = f.copy()
    for it in range(max_iter):
        u_prev = u.copy()

        div, mask_threshold = clr_split_divergence2(u, q, beta=beta, eps=eps)

        # explicit update
        u = u + tau * (div - lam * (u - f))

        reldif = np.linalg.norm(u - u_prev) / (np.linalg.norm(u))
        if reldif < tol:
            #print(f"CLR2-split converged at iteration {it+1}")
            break

    return u, q



# Definitions of functions used for ROF, dpROF and Huber-ROF models


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

# Original Chambolle-Pock for ROF, acclerated v1
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

# Custom Resolvent for dpROF
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
            #print(f"Huber ROF (acc): {i+1} iterations")
            break

    return x







# Setting the values of tested parameters

# Noise level

given_variance = 0.001

# Parameters for ROF

lambda_ROF = 0.02

# Parameters for dpROF

lambda_dpROF = 0.6*np.sqrt(given_variance)
lambda_initial_ROF = 0.8*np.sqrt(given_variance)
a_dpROF = 60
b_dpROF = 3600

# Parameters for dpROF (v2)

lambda_initial_ROF_b = 0.7*np.sqrt(given_variance)
lambda_dpROF_b = 0.5*np.sqrt(given_variance)
a_dpROF_b = 60
b_dpROF_b = 4800

# Parameters for Huber-ROF

lambda_huber = 0.02
alpha_huber = 0.005

# Parameters for TGV

a1_TGV = 0.02
a0_TGV = 0.4

# Precision levels

given_precision_ROF = 1e-4
given_precision_ROF_for_dp = 1e-2
given_precision_dpROF = 1e-4
given_precision_tgv = 1e-4
given_precision_clr = 1e-4
given_precision_huber = 1e-4




# Initialisation of variables

global_time_start = time.time()

noise_technical_sum = 0

ROF_time_sum = 0
dpROF_time_sum = 0
TGV_time_sum = 0
CLR_time_sum = 0
Huber_time_sum = 0
ROF_small_time_sum = 0
dpROF_small_time_sum = 0
dpROF_b_small_time_sum = 0
dpROF_noisy_time_sum = 0
dpROF_edge_time_sum = 0
dpROF_edge2_time_sum = 0
nl_means_time_sum = 0





iterations_number = 0

# Starting the automated loop for images

for i in range(number_of_images):

  # Initialisation step: loading images, first clear then noisy

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

  # Handle different image formats
  if noisy.ndim == 2:
     pass  # Already grayscale
  elif noisy.ndim == 3:
      if noisy.shape[2] == 4:
         noisy = color.rgb2gray(color.rgba2rgb(noisy))  # Convert RGBA -> RGB -> grayscale
      elif noisy.shape[2] == 3:
          noisy = color.rgb2gray(noisy)  # Convert RGB -> grayscale
      else:
         raise ValueError(f"Unsupported channel format: noisy.shape = {noisy.shape}")
  else:
     raise ValueError(f"Unsupported image format: noisy.ndim = {noisy.ndim}")

  noisy = img_as_float(noisy)

  noise_level_estimate = estimate_sigma(noisy)


  # Step 1: Classical ROF model

  start = time.time()
  denoised_v1v = chambolle_pock_v(noisy, tau=0.25, lam=lambda_ROF, max_iter=20000, tol=given_precision_ROF)
  t1 = time.time() - start

  ROF_time_sum += t1


  # Step 2: dpROF
 
  mollified_v1 = mollify_function(denoised_v1v, radius=2)
  grad_v1 = gradient(mollified_v1)
  grad_norm = norm2(grad_v1)

  a_const = a_dpROF
  b_const = b_dpROF

  a_weight = np.maximum(0, a_const - b_const * np.maximum(grad_norm, a_const / (2 * b_const)))

  start = time.time()
  denoised_v2v = chambolle_pock_modified_v(noisy, a_weight, tau=0.25, lam=lambda_dpROF, max_iter=20000, tol=given_precision_dpROF)
  t2 = time.time() - start

  dpROF_time_sum += t2


  # Step 3: TGV

  solver = PdHgmTGV(
    lam=1.0,
    alpha=(a1_TGV,a0_TGV),
    tol=given_precision_tgv,
    max_iter=1200
  )

  start = time.time()
  denoised_v3v = solver.solve(noisy)
  t3 = time.time() - start

  TGV_time_sum += t3

  # Step 4: Apply the Chen-Levine-Rao algorithm

  noisy_clr = 255 * noisy

  start = time.time()
  clr_denoised, q_map = clr_split_flow2(noisy_clr, tau=0.05, lam=0.3125, K=0.01, sigma=0.5, beta=30.0, max_iter=3000, tol=given_precision_clr, eps=1e-8)
  t4 = time.time() - start

  CLR_time_sum += t4

  denoised_v4v = img_as_float(clr_denoised/255)
  

  # Step 5: Apply the Huber-ROF

  start = time.time()
  denoised_v5v = chambolle_pock_huber_accelerated(noisy, alpha = alpha_huber, tau = 0.25, lam=lambda_huber, max_iter=10000, tol=given_precision_huber)
  t5 = time.time() - start

  Huber_time_sum += t5



  # Step 6: The double-phase ROF model with reduced accuracy of the initial ROF calculation

  start = time.time()
  denoised_v6va = chambolle_pock_v(noisy, tau=0.25, lam=lambda_initial_ROF, max_iter=20000, tol=given_precision_ROF_for_dp)
  t6a = time.time() - start

  ROF_small_time_sum += t6a


  # Step 2: Compute weight from gradient norm
  mollified_v6va = mollify_function(denoised_v6va, radius=2)
  grad_v1 = gradient(mollified_v6va)
  grad_norm = norm2(grad_v1)

  # Use weight 1

  a_const = a_dpROF
  b_const = b_dpROF

  a_weight = np.maximum(0, a_const - b_const * np.maximum(grad_norm, a_const / (2 * b_const)))

  start = time.time()
  denoised_v6v = chambolle_pock_modified_v(noisy, a_weight, tau=0.25, lam=lambda_dpROF, max_iter=20000, tol=given_precision_dpROF)
  t6 = time.time() - start

  dpROF_small_time_sum += t6



  # Step 8: The double-phase ROF model with lower b for reduction of staircasing

  start = time.time()
  denoised_v8va = chambolle_pock_v(noisy, tau=0.25, lam=lambda_initial_ROF_b, max_iter=20000, tol=given_precision_ROF_for_dp)
  t8a = time.time() - start

  ROF_small_time_sum += t8a


  # Step 2: Compute weight from gradient norm
  mollified_v8va = mollify_function(denoised_v8va, radius=2)
  #mollified_v1 = mollify_function(noisy, radius=1)
  grad_v1 = gradient(mollified_v8va)
  #grad_v1 = gradient(denoised_v8va)
  grad_norm = norm2(grad_v1)

  # Use weight 1

  a_const = a_dpROF_b
  b_const = b_dpROF_b

  a_weight = np.maximum(0, a_const - b_const * np.maximum(grad_norm, a_const / (2 * b_const)))

  start = time.time()
  denoised_v8v = chambolle_pock_modified_v(noisy, a_weight, tau=0.25, lam=lambda_dpROF_b, max_iter=20000, tol=given_precision_dpROF)
  t8 = time.time() - start

  dpROF_small_time_sum += t8

  # Step 8: The double-phase ROF model with lower b for reduction of staircasing *and precise ROF*

  start = time.time()
  denoised_v9va = chambolle_pock_v(noisy, tau=0.25, lam=lambda_initial_ROF_b, max_iter=20000, tol=given_precision_dpROF)
  t9a = time.time() - start

  ROF_small_time_sum += t9a


  # Step 2: Compute weight from gradient norm
  mollified_v9va = mollify_function(denoised_v9va, radius=2)
  grad_v1 = gradient(mollified_v9va)
  grad_norm = norm2(grad_v1)

  a_const = a_dpROF_b
  b_const = b_dpROF_b

  a_weight = np.maximum(0, a_const - b_const * np.maximum(grad_norm, a_const / (2 * b_const)))

  start = time.time()
  denoised_v9v = chambolle_pock_modified_v(noisy, a_weight, tau=0.25, lam=lambda_dpROF_b, max_iter=20000, tol=given_precision_dpROF)
  t9 = time.time() - start

  dpROF_small_time_sum += t9

  # Step 10: The NL-means algorithm

  start = time.time()
  denoised_v10v = denoise_nl_means(noisy, patch_size = 3, patch_distance = 21, h = 0.6 * np.sqrt(given_variance), sigma =np.sqrt(given_variance), channel_axis = None, fast_mode = True)
  t10 = time.time() - start

  nl_means_time_sum += t10


  start = time.time()
  denoised_v11v = denoise_nl_means(noisy, patch_size = 3, patch_distance = 21, h = 0.6 * np.sqrt(given_variance), sigma =np.sqrt(given_variance), channel_axis = None, fast_mode = True)
  t11 = time.time() - start

  nl_means_time_sum += t11


  # Plots of denoised versions


  images1 = [image, denoised_v1v, denoised_v9v, denoised_v5v]
  titles1 = ['Original', 'ROF', 'dpROF', 'Huber-ROF']

  images2 = [noisy, denoised_v4v, denoised_v3v, denoised_v10v]
  titles2 = ['Noisy', 'CLR', 'TGV', 'NL-means']

  fig, axes = plt.subplots(2, 4, figsize=(12, 8))

  axes = axes.ravel()

  images = images1 + images2
  titles = titles1 + titles2

  for ax, img, title in zip(axes, images, titles):
    ax.imshow(img, cmap='gray')
    ax.set_title(title)
    ax.axis('off')

  plt.tight_layout()
  plt.show()

start = time.time()
  denoised_v11v = denoise_nl_means(noisy, patch_size = 3, patch_distance = 21, h = 0.6 * np.sqrt(given_variance), sigma =np.sqrt(given_variance), channel_axis = None, fast_mode = True)
  #denoised_v10v = denoise_nl_means(noisy, h = 0.65 * noise_level_estimate, channel_axis = None, fast_mode = True)
  t11 = time.time() - start

  nl_means_time_sum += t11

images1 = [image, denoised_v1v, denoised_v9v, denoised_v5v]
  titles1 = ['Original', 'ROF', 'dpROF', 'Huber-ROF']

  images2 = [noisy, denoised_v4v, denoised_v3v, denoised_v11v]
  titles2 = ['Noisy', 'CLR', 'TGV', 'NL-means']

  fig, axes = plt.subplots(2, 4, figsize=(12, 8))

  axes = axes.ravel()

  images = images1 + images2
  titles = titles1 + titles2

  for ax, img, title in zip(axes, images, titles):
    ax.imshow(img, cmap='gray')
    ax.set_title(title)
    ax.axis('off')

  plt.tight_layout()
  plt.show()

print('ROF:', ssim(image, denoised_v1v, data_range=1.0))
print('dpROF:', ssim(image, denoised_v9v, data_range=1.0))
print('Huber-ROF:', ssim(image, denoised_v5v, data_range=1.0))
print('CLR:', ssim(image, denoised_v4v, data_range=1.0))
print('TGV:', ssim(image, denoised_v3v, data_range=1.0))
print('NLM:', ssim(image, denoised_v11v, data_range=1.0))

def preprocess_image_for_lpips(img_np):
    # Convert numpy array to torch tensor
    img_tensor = torch.from_numpy(img_np).float()
    # Add channel dimension (C=1) and batch dimension (N=1)
    # Shape becomes (1, 1, H, W)
    img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)
    # Replicate the single channel to 3 channels for LPIPS (expects RGB-like input)
    # Shape becomes (1, 3, H, W)
    img_tensor = img_tensor.repeat(1, 3, 1, 1)
    # Move to the appropriate device (CPU/GPU)
    img_tensor = img_tensor.to(device)
    return img_tensor

print('ROF:', loss_fn(preprocess_image_for_lpips(image), preprocess_image_for_lpips(denoised_v1v)).item())
print('dpROF:', loss_fn(preprocess_image_for_lpips(image), preprocess_image_for_lpips(denoised_v9v)).item())
print('Huber-ROF:', loss_fn(preprocess_image_for_lpips(image), preprocess_image_for_lpips(denoised_v5v)).item())
print('CLR:', loss_fn(preprocess_image_for_lpips(image), preprocess_image_for_lpips(denoised_v4v)).item())
print('TGV:', loss_fn(preprocess_image_for_lpips(image), preprocess_image_for_lpips(denoised_v3v)).item())
print('NLM:', loss_fn(preprocess_image_for_lpips(image), preprocess_image_for_lpips(denoised_v11v)).item())

denoised_images = [
    denoised_v1v, denoised_v9v, denoised_v5v,
    denoised_v4v, denoised_v3v, denoised_v11v
]

denoised_titles = [
    'ROF', 'dpROF', 'Huber-ROF',
    'CLR', 'TGV', 'NLM'
]

fig, axes = plt.subplots(2, 3, figsize=(12, 8))
axes = axes.ravel()

for ax, img, title in zip(axes, denoised_images, denoised_titles):
    error_map = np.abs(img - image)
    #error_map = np.sqrt((img - image)**2)
    #error_map = (img - image)**2

    im = ax.imshow(error_map, cmap='hot', vmin=0.0, vmax=0.05)
    ax.set_title(f'Error map - {title}', fontsize=10)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

axes[-1].axis('off')

plt.tight_layout()
plt.show()

denoised_images = [
    denoised_v1v, denoised_v9v, denoised_v5v,
    denoised_v4v, denoised_v3v, denoised_v11v
]

denoised_titles = [
    'ROF', 'dpROF', 'Huber-ROF',
    'CLR', 'TGV', 'NLM'
]

fig, axes = plt.subplots(2, 3, figsize=(12, 8))
axes = axes.ravel()

for ax, img, title in zip(axes, denoised_images, denoised_titles):
    error_diff = np.abs(denoised_v9v- image) - np.abs(img - image)
    #error_map = np.sqrt((img - image)**2)
    #error_map = (img - image)**2

    im = ax.imshow(error_diff, cmap='plasma', vmin=-0.01, vmax=0.01)
    ax.set_title(f'Error diff: dpROF - {title}', fontsize=10)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

axes[-1].axis('off')

plt.tight_layout()
plt.show()