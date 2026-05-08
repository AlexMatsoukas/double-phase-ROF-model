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




# Loading clear images

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



# Option: loading noisy images from a separate folder (second option: creating them from clear images during the loop)

# Add proper directory path

folder_path = '/noisy_images'


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
all_images_noisy = []

print("Loading images...")

for image_file in image_files:
    # Construct the full path to the image file
    full_path = os.path.join(folder_path, image_file)

    try:
        # Load the image
        image = io.imread(full_path)
        all_images_noisy.append(image)
        #print(f"Successfully loaded: {image_file} | Shape: {image.shape}")

    except Exception as e:
        print(f"Could not load {image_file}. Reason: {e}")

number_of_images = len(all_images_noisy)
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

# Parameters for classical ROF

lambda_ROF = 0.02
given_precision_ROF = 1e-4
max_iter_ROF = 3000

# Parameters for dpROF

# These parameters can be computed either using given_variance or the estimated noise level

lambda_initial_ROF = 0.7*np.sqrt(given_variance)
lambda_dpROF = 0.5*np.sqrt(given_variance)

a_dpROF = 60
b_dpROF = 6000

given_precision_initial_ROF = 1e-2
given_precision_dpROF = 1e-4
max_iter_initial_ROF = 3000
max_iter_dpROF = 3000

# Parameters for TGV

tgv_alpha1 = 0.02
tgv_alpha2 = 0.4
given_precision_tgv = 1e-4
max_iter_tgv = 3000

# Parameters for Chen-Levine-Rao

clr_lambda = 0.3125
clr_K = 0.01
clr_beta = 30
given_precision_clr = 1e-4
max_iter_clr = 3000

# Parameters for Huber-ROF

lambda_huber = 0.02
alpha_huber = 0.005
given_precision_huber = 1e-4
max_iter_huber = 3000

# Parameters for dpROF-LPIPS

lambda_ROF_LPIPS = 0.02
lambda_dpROF_LPIPS = 0.0175

a_dpROF_LPIPS = 40
b_dpROF_LPIPS = 4000

# Parameters for dpROF-SSIM

lambda_ROF_SSIM = 0.02
lambda_dpROF_SSIM = 0.015

a_dpROF_SSIM = 70
b_dpROF_SSIM = 7000

# Parameters for NL-means

nl_means_h = 0.6
size_1 = 3
size_2 = 21








# Initialisation of variables

global_time_start = time.time()

noise_technical_sum = 0

ROF_time_sum = 0
ROF_initial_time_sum = 0
dpROF_time_sum = 0
TGV_time_sum = 0
CLR_time_sum = 0
Huber_time_sum = 0
ROF_LPIPS_time_sum = 0
dpROF_LPIPS_time_sum = 0
dpROF_noisy_time_sum = 0
dpROF_edge_time_sum = 0
dpROF_SSIM_time_sum = 0
nl_means_time_sum = 0

ROF_LPIPS_time_sum = 0
ROF_SSIM_time_sum = 0

lpips_ROF_image = [0 for _ in range(number_of_images)]
lpips_dpROF_image = [0 for _ in range(number_of_images)]
lpips_TGV_image = [0 for _ in range(number_of_images)]
lpips_CLR_image = [0 for _ in range(number_of_images)]
lpips_Huber_image = [0 for _ in range(number_of_images)]
lpips_dpROF_LPIPS_image = [0 for _ in range(number_of_images)]
lpips_dpROF_noisy_image = [0 for _ in range(number_of_images)]
lpips_dpROF_edge_image = [0 for _ in range(number_of_images)]
lpips_dpROF_SSIM_image = [0 for _ in range(number_of_images)]
lpips_nl_means_image = [0 for _ in range(number_of_images)]

ssim_ROF_image = [0 for _ in range(number_of_images)]
ssim_dpROF_image = [0 for _ in range(number_of_images)]
ssim_TGV_image = [0 for _ in range(number_of_images)]
ssim_CLR_image = [0 for _ in range(number_of_images)]
ssim_Huber_image = [0 for _ in range(number_of_images)]
ssim_dpROF_LPIPS_image = [0 for _ in range(number_of_images)]
ssim_dpROF_noisy_image = [0 for _ in range(number_of_images)]
ssim_dpROF_edge_image = [0 for _ in range(number_of_images)]
ssim_dpROF_SSIM_image = [0 for _ in range(number_of_images)]
ssim_nl_means_image = [0 for _ in range(number_of_images)]

psnr_ROF_image = [0 for _ in range(number_of_images)]
psnr_dpROF_image = [0 for _ in range(number_of_images)]
psnr_TGV_image = [0 for _ in range(number_of_images)]
psnr_CLR_image = [0 for _ in range(number_of_images)]
psnr_Huber_image = [0 for _ in range(number_of_images)]
psnr_dpROF_LPIPS_image = [0 for _ in range(number_of_images)]
psnr_dpROF_noisy_image = [0 for _ in range(number_of_images)]
psnr_dpROF_edge_image = [0 for _ in range(number_of_images)]
psnr_dpROF_SSIM_image = [0 for _ in range(number_of_images)]
psnr_nl_means_image = [0 for _ in range(number_of_images)]


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

  # Creating noisy images

  noisy = util.random_noise(image, mode='gaussian', var=given_variance)

  # Alternatively: using noisy images from a given folder

  #noisy = all_images_noisy[i]

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

  # Alternative definition of parameters using the estimated noise level

  #lambda_initial_ROF = 0.7*noise_level_estimate
  #lambda_dpROF = 0.5*noise_level_estimate

  # Step 1: Classical ROF

  start = time.time()
  denoised_v1v = chambolle_pock_v(noisy, tau=0.25, lam=lambda_ROF, max_iter=max_iter_ROF, tol=given_precision_ROF)
  t1 = time.time() - start

  ROF_time_sum += t1


  # Step 2: dpROF

  start = time.time()
  denoised_v1va = chambolle_pock_v(noisy, tau=0.25, lam=lambda_initial_ROF, max_iter=max_iter_initial_ROF, tol=given_precision_initial_ROF)
  t1a = time.time() - start

  ROF_initial_time_sum += t1a

  mollified_v1 = mollify_function(denoised_v1va, radius=2)
  grad_v1 = gradient(mollified_v1)
  grad_norm = norm2(grad_v1)

  a_const = a_dpROF
  b_const = b_dpROF

  a_weight = np.maximum(0, a_const - b_const * np.maximum(grad_norm, a_const / (2 * b_const)))

  start = time.time()
  denoised_v2v = chambolle_pock_modified_v(noisy, a_weight, tau=0.25, lam=lambda_dpROF, max_iter=max_iter_dpROF, tol=given_precision_dpROF)
  t2 = time.time() - start

  dpROF_time_sum += t2


  # Step 3: TGV

  solver = PdHgmTGV(
    lam=1.0,
    alpha=(tgv_alpha1,tgv_alpha2),
    tol=given_precision_tgv,
    max_iter=max_iter_tgv
  )

  start = time.time()
  denoised_v3v = solver.solve(noisy)
  t3 = time.time() - start

  TGV_time_sum += t3


  # Step 4: Chen-Levine-Rao

  noisy_clr = img_as_float(noisy) * 255.0

  start = time.time()
  clr_denoised, q_map = clr_split_flow2(noisy_clr, tau=0.05, lam=clr_lambda, K=clr_K, sigma=0.5, beta=clr_beta, max_iter=max_iter_clr, tol=given_precision_clr, eps=1e-8)
  t4 = time.time() - start

  CLR_time_sum += t4

  denoised_v4v = img_as_float(clr_denoised/255)


  # Step 5: Huber-ROF

  start = time.time()
  denoised_v5v = chambolle_pock_huber_accelerated(noisy, alpha = alpha_huber, tau = 0.25, lam=lambda_huber, max_iter=max_iter_huber, tol=given_precision_huber)
  t5 = time.time() - start

  Huber_time_sum += t5



  # Step 6: dpROF-LPIPS

  start = time.time()
  denoised_v6va = chambolle_pock_v(noisy, tau=0.25, lam=lambda_ROF_LPIPS, max_iter=max_iter_initial_ROF, tol=given_precision_initial_ROF)
  t6a = time.time() - start

  ROF_LPIPS_time_sum += t6a


  mollified_v6va = mollify_function(denoised_v6va, radius=2)
  grad_v1 = gradient(mollified_v6va)
  grad_norm = norm2(grad_v1)

  a_const = a_dpROF_LPIPS
  b_const = b_dpROF_LPIPS

  a_weight = np.maximum(0, a_const - b_const * np.maximum(grad_norm, a_const / (2 * b_const)))

  start = time.time()
  denoised_v6v = chambolle_pock_modified_v(noisy, a_weight, tau=0.25, lam=lambda_dpROF_LPIPS, max_iter=max_iter_dpROF, tol=given_precision_dpROF)
  t6 = time.time() - start

  dpROF_LPIPS_time_sum += t6



  # Step 7: dpROF-noisy

  mollified_v1 = mollify_function(noisy, radius=2)
  grad_v1 = gradient(mollified_v1)
  grad_norm = norm2(grad_v1)

  a_const = a_dpROF
  b_const = b_dpROF

  a_weight = np.maximum(0, a_const - b_const * np.maximum(grad_norm, a_const / (2 * b_const)))

  start = time.time()
  denoised_v7v = chambolle_pock_modified_v(noisy, a_weight, tau=0.25, lam=lambda_dpROF, max_iter=max_iter_dpROF, tol=given_precision_dpROF)
  t7 = time.time() - start

  dpROF_noisy_time_sum += t7




  # Step 8: dpROF-edge

  start = time.time()
  noisy_canny = cv2.normalize(noisy, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

  # Option A: automatic choosing of parameters

  median_noisy = np.median(noisy_canny)
  sigma_canny = 0.3
  lowerbound_canny = int(max(0,1.0 - sigma_canny) * median_noisy)
  upperbound_canny = int(max(0,1.0 - sigma_canny) * median_noisy)

  # Option B: manual choosing of parameters

  #lowerbound_canny = 5
  #upperbound_canny = 50

  edges_canny = cv2.Canny(noisy_canny, threshold1=lowerbound_canny, threshold2=upperbound_canny)

  edges = edges_canny.astype(np.float32) / 255.0

  mollified_v1 = mollify_function(edges, radius=1)
  grad_v1 = gradient(mollified_v1)
  grad_norm = norm2(grad_v1)

  a_const = a_dpROF
  b_const = b_dpROF

  a_weight = np.maximum(0, a_const - b_const * np.maximum(grad_norm, a_const / (2 * b_const)))

  denoised_v8v = chambolle_pock_modified_v(noisy, a_weight, tau=0.25, lam=lambda_dpROF, max_iter=max_iter_dpROF)
  t8 = time.time() - start

  dpROF_edge_time_sum += t8


  # Step 9: dpROF-SSIM

  start = time.time()
  denoised_v9va = chambolle_pock_v(noisy, tau=0.25, lam=lambda_ROF_SSIM, max_iter=max_iter_initial_ROF, tol=given_precision_initial_ROF)
  t9a = time.time() - start

  ROF_SSIM_time_sum += t9a
  
  mollified_v9va = mollify_function(denoised_v9va, radius=2)
  grad_v1 = gradient(mollified_v9va)
  grad_norm = norm2(grad_v1)

  a_const = a_dpROF_SSIM
  b_const = b_dpROF_SSIM

  a_weight = np.maximum(0, a_const - b_const * np.maximum(grad_norm, a_const / (2 * b_const)))

  start = time.time()
  denoised_v9v = chambolle_pock_modified_v(noisy, a_weight, tau=0.25, lam=lambda_dpROF_SSIM, max_iter=max_iter_dpROF, tol=given_precision_dpROF)
  t9 = time.time() - start

  dpROF_SSIM_time_sum += t9



  # Step 10: NL-means

  start = time.time()
  denoised_v10v = denoised_v10v = denoise_nl_means(noisy, patch_size = size_1, patch_distance = size_2, h = nl_means_h * noise_level_estimate, sigma = noise_level_estimate, channel_axis = None)
  t10 = time.time() - start

  nl_means_time_sum += t10


  # Version for color images
  #denoised_v10v = denoise_nl_means(noisy, h = 1.15 * noise_level_estimate, channel_axis=-1)



  # Computing the metrics for all relevant pairs of images
  

  # Computing LPIPS: normalisation and computation

  # Normalisation for LPIPS (different image dimensions, rescaled to [-1,1])

  image_lpips = conversion_for_lpips(image)

  ROF_lpips = conversion_for_lpips(denoised_v1v)

  dpROF_lpips = conversion_for_lpips(denoised_v2v)

  TGV_lpips = conversion_for_lpips(denoised_v3v)

  CLR_lpips = conversion_for_lpips(denoised_v4v)

  Huber_lpips = conversion_for_lpips(denoised_v5v)

  dpROF_LPIPS_lpips = conversion_for_lpips(denoised_v6v)

  dpROF_noisy_lpips = conversion_for_lpips(denoised_v7v)

  dpROF_edge_lpips = conversion_for_lpips(denoised_v8v)

  dpROF_SSIM_lpips = conversion_for_lpips(denoised_v9v)

  nl_means_lpips = conversion_for_lpips(denoised_v10v)


  # Computing LPIPS

  with torch.no_grad():
     distance = loss_fn(image_lpips, ROF_lpips)

  score = distance.item()
  lpips_ROF_image[i] = score


  # dpROF vs image

  with torch.no_grad():
     distance = loss_fn(image_lpips, dpROF_lpips)

  score = distance.item()
  lpips_dpROF_image[i] = score


  # TGV vs image

  with torch.no_grad():
     distance = loss_fn(image_lpips, TGV_lpips)

  score = distance.item()
  lpips_TGV_image[i] = score


  # CLR vs image

  with torch.no_grad():
     distance = loss_fn(image_lpips, CLR_lpips)

  score = distance.item()
  lpips_CLR_image[i] = score


  # Huber-ROF vs image

  with torch.no_grad():
     distance = loss_fn(image_lpips, Huber_lpips)

  score = distance.item()
  lpips_Huber_image[i] = score



  # dpROF-LPIPS vs image

  with torch.no_grad():
     distance = loss_fn(image_lpips, dpROF_LPIPS_lpips)

  score = distance.item()
  lpips_dpROF_LPIPS_image[i] = score


  # dpROF-noisy vs image

  with torch.no_grad():
     distance = loss_fn(image_lpips, dpROF_noisy_lpips)

  score = distance.item()
  lpips_dpROF_noisy_image[i] = score


  # dpROF-edge vs image

  with torch.no_grad():
     distance = loss_fn(image_lpips, dpROF_edge_lpips)

  score = distance.item()
  lpips_dpROF_edge_image[i] = score


  # dpROF-SSIM vs image

  with torch.no_grad():
     distance = loss_fn(image_lpips, dpROF_SSIM_lpips)

  score = distance.item()
  lpips_dpROF_SSIM_image[i] = score


  # NL-means vs image

  with torch.no_grad():
     distance = loss_fn(image_lpips, nl_means_lpips)

  score = distance.item()
  lpips_nl_means_image[i] = score


  # Now SSIM and PSNR

  # ROF

  ssim_val = ssim(image, denoised_v1v, data_range = 1)
  psnr_val = psnr(image, denoised_v1v, data_range = 1)

  ssim_ROF_image[i] = float(ssim_val)
  psnr_ROF_image[i] = float(psnr_val)

  # dpROF

  ssim_val = ssim(image, denoised_v2v, data_range = 1)
  psnr_val = psnr(image, denoised_v2v, data_range = 1)

  ssim_dpROF_image[i] = float(ssim_val)
  psnr_dpROF_image[i] = float(psnr_val)

  # TGV

  ssim_val = ssim(image, denoised_v3v, data_range = 1)
  psnr_val = psnr(image, denoised_v3v, data_range = 1)

  ssim_TGV_image[i] = float(ssim_val)
  psnr_TGV_image[i] = float(psnr_val)

  # Chen-Levine-Rao

  ssim_val = ssim(image, denoised_v4v, data_range = 1)
  psnr_val = psnr(image, denoised_v4v, data_range = 1)

  ssim_CLR_image[i] = float(ssim_val)
  psnr_CLR_image[i] = float(psnr_val)

  # Huber-ROF

  ssim_val = ssim(image, denoised_v5v, data_range = 1)
  psnr_val = psnr(image, denoised_v5v, data_range = 1)

  ssim_Huber_image[i] = float(ssim_val)
  psnr_Huber_image[i] = float(psnr_val)

  # dpROF-LPIPS

  ssim_val = ssim(image, denoised_v6v, data_range = 1)
  psnr_val = psnr(image, denoised_v6v, data_range = 1)

  ssim_dpROF_LPIPS_image[i] = float(ssim_val)
  psnr_dpROF_LPIPS_image[i] = float(psnr_val)

  # dpROF-LPIPS

  ssim_val = ssim(image, denoised_v7v, data_range = 1)
  psnr_val = psnr(image, denoised_v7v, data_range = 1)

  ssim_dpROF_noisy_image[i] = float(ssim_val)
  psnr_dpROF_noisy_image[i] = float(psnr_val)


  # dpROF-edge

  ssim_val = ssim(image, denoised_v8v, data_range = 1)
  psnr_val = psnr(image, denoised_v8v, data_range = 1)

  ssim_dpROF_edge_image[i] = float(ssim_val)
  psnr_dpROF_edge_image[i] = float(psnr_val)


  # dpROF-SSIM

  ssim_val = ssim(image, denoised_v9v, data_range = 1)
  psnr_val = psnr(image, denoised_v9v, data_range = 1)

  ssim_dpROF_SSIM_image[i] = float(ssim_val)
  psnr_dpROF_SSIM_image[i] = float(psnr_val)


  # NL-means

  ssim_val = ssim(image, denoised_v10v, data_range = 1)
  psnr_val = psnr(image, denoised_v10v, data_range = 1)

  ssim_nl_means_image[i] = float(ssim_val)
  psnr_nl_means_image[i] = float(psnr_val)





  iterations_number += 1

  # Adding outputs to a log file

  # Add proper file path

  log_file = '/comparison_extendedlog.log'

  with open(log_file, 'a') as f:
      timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

      f.write(f"TIME: [{timestamp}]\n\n")
      f.write(f"Prescribed noise variance: {given_variance}\n\n")

      f.write(f"Image no.: {iterations_number} \n\n")

      f.write(f"Time, ROF: {t1:.2f}s \n")
      f.write(f"Time, dpROF: {t1a:.2f}s + {t2:.2f}s \n")
      f.write(f"Time, TGV: {t3:.2f}s \n")
      f.write(f"Time, NL-means: {t10:.2f}s \n")
      f.write(f"Time, Chen-Levine-Rao: {t4:.2f}s \n")
      f.write(f"Time, Huber-ROF: {t5:.2f}s \n")
      f.write(f"Time, dpROF-LPIPS: {t6a:.2f}s + {t6:.2f}s \n")
      f.write(f"Time, dpROF-noisy: {t7:.2f}s \n")
      f.write(f"Time, dpROF-edge: {t8:.2f}s \n")
      f.write(f"Time, dpROF-SSIM: {t9a:.2f}s + {t9:.2f}s \n\n")

      f.write(f"LPIPS, ROF: {lpips_ROF_image[i]} \n")
      f.write(f"LPIPS, dpROF: {lpips_dpROF_image[i]} \n")
      f.write(f"LPIPS, TGV: {lpips_TGV_image[i]} \n")
      f.write(f"LPIPS, NL-means: {lpips_nl_means_image[i]} \n")
      f.write(f"LPIPS, Chen-Levine-Rao: {lpips_CLR_image[i]} \n")
      f.write(f"LPIPS, Huber-ROF: {lpips_Huber_image[i]} \n")
      f.write(f"LPIPS, dpROF-LPIPS: {lpips_dpROF_LPIPS_image[i]} \n")
      f.write(f"LPIPS, dpROF-noisy: {lpips_dpROF_noisy_image[i]} \n")
      f.write(f"LPIPS, dpROF-edge: {lpips_dpROF_edge_image[i]} \n")
      f.write(f"LPIPS, dpROF-SSIM: {lpips_dpROF_SSIM_image[i]} \n\n")

      f.write(f"SSIM, ROF: {ssim_ROF_image[i]} \n")
      f.write(f"SSIM, dpROF: {ssim_dpROF_image[i]} \n")
      f.write(f"SSIM, TGV: {ssim_TGV_image[i]} \n")
      f.write(f"SSIM, NL-means: {ssim_nl_means_image[i]} \n")
      f.write(f"SSIM, Chen-Levine-Rao: {ssim_CLR_image[i]} \n")
      f.write(f"SSIM, Huber-ROF: {ssim_Huber_image[i]} \n")
      f.write(f"SSIM, dpROF-LPIPS: {ssim_dpROF_LPIPS_image[i]} \n")
      f.write(f"SSIM, dpROF-noisy: {ssim_dpROF_noisy_image[i]} \n")
      f.write(f"SSIM, dpROF-edge: {ssim_dpROF_edge_image[i]} \n")
      f.write(f"SSIM, dpROF-SSIM: {ssim_dpROF_SSIM_image[i]} \n\n")

      f.write(f"PSNR, ROF: {psnr_ROF_image[i]} \n")
      f.write(f"PSNR, dpROF: {psnr_dpROF_image[i]} \n")
      f.write(f"PSNR, TGV: {psnr_TGV_image[i]} \n")
      f.write(f"PSNR, NL-means: {psnr_nl_means_image[i]} \n")
      f.write(f"PSNR, Chen-Levine-Rao: {psnr_CLR_image[i]} \n")
      f.write(f"PSNR, Huber-ROF: {psnr_Huber_image[i]} \n")
      f.write(f"PSNR, dpROF-LPIPS: {psnr_dpROF_LPIPS_image[i]} \n")
      f.write(f"PSNR, dpROF-noisy: {psnr_dpROF_noisy_image[i]} \n")
      f.write(f"PSNR, dpROF-edge: {psnr_dpROF_edge_image[i]} \n")
      f.write(f"PSNR, dpROF-SSIM: {psnr_dpROF_SSIM_image[i]} \n\n")

      f.write("======================================\n\n")

  if iterations_number % 2 == 0:
            print(f"{iterations_number}/{number_of_images} iterations.")

print()

global_time_total = time.time() - global_time_start

ROF_time_average = ROF_time_sum / number_of_images
ROF_initial_time_average = ROF_initial_time_sum / number_of_images
dpROF_time_average = dpROF_time_sum / number_of_images
TGV_time_average = TGV_time_sum / number_of_images
CLR_time_average = CLR_time_sum / number_of_images
Huber_time_average = Huber_time_sum / number_of_images
ROF_LPIPS_time_average = ROF_LPIPS_time_sum / number_of_images
dpROF_LPIPS_time_average = dpROF_LPIPS_time_sum / number_of_images
dpROF_noisy_time_average = dpROF_noisy_time_sum / number_of_images
dpROF_edge_time_average = dpROF_edge_time_sum / number_of_images
ROF_SSIM_time_average = ROF_SSIM_time_sum / number_of_images
dpROF_SSIM_time_average = dpROF_SSIM_time_sum / number_of_images
nl_means_time_average = nl_means_time_sum / number_of_images

average_lpips_ROF_image = sum(lpips_ROF_image) / number_of_images
average_lpips_dpROF_image = sum(lpips_dpROF_image) / number_of_images
average_lpips_TGV_image = sum(lpips_TGV_image) / number_of_images
average_lpips_CLR_image = sum(lpips_CLR_image) / number_of_images
average_lpips_Huber_image = sum(lpips_Huber_image) / number_of_images
average_lpips_dpROF_LPIPS_image = sum(lpips_dpROF_LPIPS_image) / number_of_images
average_lpips_dpROF_noisy_image = sum(lpips_dpROF_noisy_image) / number_of_images
average_lpips_dpROF_edge_image = sum(lpips_dpROF_edge_image) / number_of_images
average_lpips_dpROF_SSIM_image = sum(lpips_dpROF_SSIM_image) / number_of_images
average_lpips_nl_means_image = sum(lpips_nl_means_image) / number_of_images

average_ssim_ROF_image = sum(ssim_ROF_image) / number_of_images
average_ssim_dpROF_image = sum(ssim_dpROF_image) / number_of_images
average_ssim_TGV_image = sum(ssim_TGV_image) / number_of_images
average_ssim_CLR_image = sum(ssim_CLR_image) / number_of_images
average_ssim_Huber_image = sum(ssim_Huber_image) / number_of_images
average_ssim_dpROF_LPIPS_image = sum(ssim_dpROF_LPIPS_image) / number_of_images
average_ssim_dpROF_noisy_image = sum(ssim_dpROF_noisy_image) / number_of_images
average_ssim_dpROF_edge_image = sum(ssim_dpROF_edge_image) / number_of_images
average_ssim_dpROF_SSIM_image = sum(ssim_dpROF_SSIM_image) / number_of_images
average_ssim_nl_means_image = sum(ssim_nl_means_image) / number_of_images

average_psnr_ROF_image = sum(psnr_ROF_image) / number_of_images
average_psnr_dpROF_image = sum(psnr_dpROF_image) / number_of_images
average_psnr_TGV_image = sum(psnr_TGV_image) / number_of_images
average_psnr_CLR_image = sum(psnr_CLR_image) / number_of_images
average_psnr_Huber_image = sum(psnr_Huber_image) / number_of_images
average_psnr_dpROF_LPIPS_image = sum(psnr_dpROF_LPIPS_image) / number_of_images
average_psnr_dpROF_noisy_image = sum(psnr_dpROF_noisy_image) / number_of_images
average_psnr_dpROF_edge_image = sum(psnr_dpROF_edge_image) / number_of_images
average_psnr_dpROF_SSIM_image = sum(psnr_dpROF_SSIM_image) / number_of_images
average_psnr_nl_means_image = sum(psnr_nl_means_image) / number_of_images




print(f"Total elapsed time: {global_time_total:.2f}s \n\n")

print(f"Average time, ROF: {ROF_time_average:.2f}s \n")
print(f"Average time, dpROF: {ROF_initial_time_average:.2f}s + {dpROF_time_average:.2f}s \n")
print(f"Average time, TGV: {TGV_time_average:.2f}s \n")
print(f"Average time, NL-means: {nl_means_time_average:.2f}s \n")
print(f"Average time, Chen-Levine-Rao: {CLR_time_average:.2f}s \n")
print(f"Average time, Huber-ROF: {Huber_time_average:.2f}s \n")
print(f"Average time, dpROF-LPIPS: {ROF_LPIPS_time_average:.2f}s + {dpROF_LPIPS_time_average:.2f}s \n")
print(f"Average time, dpROF-noisy: {dpROF_noisy_time_average:.2f}s \n")
print(f"Average time, dpROF-edge: {dpROF_edge_time_average:.2f}s \n")
print(f"Average time, dpROF-SSIM: {ROF_SSIM_time_average:.2f}s + {dpROF_SSIM_time_average:.2f}s \n\n")



print(f"Average LPIPS, ROF: {average_lpips_ROF_image} \n")
print(f"Average LPIPS, dpROF: {average_lpips_dpROF_image} \n")
print(f"Average LPIPS, TGV: {average_lpips_TGV_image} \n")
print(f"Average LPIPS, NL-means: {average_lpips_nl_means_image} \n")
print(f"Average LPIPS, Chen-Levine-Rao: {average_lpips_CLR_image} \n")
print(f"Average LPIPS, Huber-ROF: {average_lpips_Huber_image} \n")
print(f"Average LPIPS, dpROF-LPIPS: {average_lpips_dpROF_LPIPS_image} \n")
print(f"Average LPIPS, dpROF-noisy: {average_lpips_dpROF_noisy_image} \n")
print(f"Average LPIPS, dpROF-edge: {average_lpips_dpROF_edge_image} \n")
print(f"Average LPIPS, dpROF-SSIM: {average_lpips_dpROF_SSIM_image} \n\n")


print(f"Average SSIM, ROF: {average_ssim_ROF_image} \n")
print(f"Average SSIM, dpROF: {average_ssim_dpROF_image} \n")
print(f"Average SSIM, TGV: {average_ssim_TGV_image} \n")
print(f"Average SSIM, NL-means: {average_ssim_nl_means_image} \n")
print(f"Average SSIM, Chen-Levine-Rao: {average_ssim_CLR_image} \n")
print(f"Average SSIM, Huber-ROF: {average_ssim_Huber_image} \n")
print(f"Average SSIM, dpROF-LPIPS: {average_ssim_dpROF_LPIPS_image} \n")
print(f"Average SSIM, dpROF-noisy: {average_ssim_dpROF_noisy_image} \n")
print(f"Average SSIM, dpROF-edge: {average_ssim_dpROF_edge_image} \n")
print(f"Average SSIM, dpROF-SSIM: {average_ssim_dpROF_SSIM_image} \n\n")



print(f"Average PSNR, ROF: {average_psnr_ROF_image} \n")
print(f"Average PSNR, dpROF: {average_psnr_dpROF_image} \n")
print(f"Average PSNR, TGV: {average_psnr_TGV_image} \n")
print(f"Average PSNR, NL-means: {average_psnr_nl_means_image} \n")
print(f"Average PSNR, Chen-Levine-Rao: {average_psnr_CLR_image} \n")
print(f"Average PSNR, Huber-ROF: {average_psnr_Huber_image} \n")
print(f"Average PSNR, dpROF-LPIPS: {average_psnr_dpROF_LPIPS_image} \n")
print(f"Average PSNR, dpROF-noisy: {average_psnr_dpROF_noisy_image} \n")
print(f"Average PSNR, dpROF-edge: {average_psnr_dpROF_edge_image} \n")
print(f"Average PSNR, dpROF-SSIM: {average_psnr_dpROF_SSIM_image} \n")




# Adding outputs to a log file

# Add proper directory path

log_file = '/comparison.log'

with open(log_file, 'a') as f:
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    f.write(f"TIME: [{timestamp}]\n\n")

    f.write(f"Total elapsed time: {global_time_total:.2f}s \n\n")

    f.write(f"Prescribed noise variance: {given_variance}\n")
    f.write(f"Estimated average noise level: {noise_technical_sum}\n\n")

    f.write(f"Prescribed ROF accuracy: {given_precision_ROF}\n")
    f.write(f"Prescribed dpROF accuracy: {given_precision_dpROF} \n\n")

    f.write(f"Average time, ROF: {ROF_time_average:.2f}s \n")
    f.write(f"Average time, dpROF: {ROF_initial_time_average:.2f}s + {dpROF_time_average:.2f}s \n")
    f.write(f"Average time, TGV: {TGV_time_average:.2f}s \n")
    f.write(f"Average time, NL-means: {nl_means_time_average:.2f}s \n")
    f.write(f"Average time, Chen-Levine-Rao: {CLR_time_average:.2f}s \n")
    f.write(f"Average time, Huber-ROF: {Huber_time_average:.2f}s \n")
    f.write(f"Average time, dpROF-LPIPS: {ROF_LPIPS_time_average:.2f}s + {dpROF_LPIPS_time_average:.2f}s \n")
    f.write(f"Average time, dpROF-noisy: {dpROF_noisy_time_average:.2f}s \n")
    f.write(f"Average time, dpROF-edge: {dpROF_edge_time_average:.2f}s \n")
    f.write(f"Average time, dpROF-SSIM: {ROF_SSIM_time_average:.2f}s + {dpROF_SSIM_time_average:.2f}s \n\n")

    f.write(f"Average LPIPS, ROF: {average_lpips_ROF_image} \n")
    f.write(f"Average LPIPS, dpROF: {average_lpips_dpROF_image} \n")
    f.write(f"Average LPIPS, TGV: {average_lpips_TGV_image} \n")
    f.write(f"Average LPIPS, NL-means: {average_lpips_nl_means_image} \n")
    f.write(f"Average LPIPS, Chen-Levine-Rao: {average_lpips_CLR_image} \n")
    f.write(f"Average LPIPS, Huber-ROF: {average_lpips_Huber_image} \n")
    f.write(f"Average LPIPS, dpROF-LPIPS: {average_lpips_dpROF_LPIPS_image} \n")
    f.write(f"Average LPIPS, dpROF-noisy: {average_lpips_dpROF_noisy_image} \n")
    f.write(f"Average LPIPS, dpROF-edge: {average_lpips_dpROF_edge_image} \n")
    f.write(f"Average LPIPS, dpROF-SSIM: {average_lpips_dpROF_SSIM_image} \n\n")

    f.write(f"Average SSIM, ROF: {average_ssim_ROF_image} \n")
    f.write(f"Average SSIM, dpROF: {average_ssim_dpROF_image} \n")
    f.write(f"Average SSIM, TGV: {average_ssim_TGV_image} \n")
    f.write(f"Average SSIM, NL-means: {average_ssim_nl_means_image} \n")
    f.write(f"Average SSIM, Chen-Levine-Rao: {average_ssim_CLR_image} \n")
    f.write(f"Average SSIM, Huber-ROF: {average_ssim_Huber_image} \n")
    f.write(f"Average SSIM, dpROF-LPIPS: {average_ssim_dpROF_LPIPS_image} \n")
    f.write(f"Average SSIM, dpROF-noisy: {average_ssim_dpROF_noisy_image} \n")
    f.write(f"Average SSIM, dpROF-edge: {average_ssim_dpROF_edge_image} \n")
    f.write(f"Average SSIM, dpROF-SSIM: {average_ssim_dpROF_SSIM_image} \n\n")

    f.write(f"Average PSNR, ROF: {average_psnr_ROF_image} \n")
    f.write(f"Average PSNR, dpROF: {average_psnr_dpROF_image} \n")
    f.write(f"Average PSNR, TGV: {average_psnr_TGV_image} \n")
    f.write(f"Average PSNR, NL-means: {average_psnr_nl_means_image} \n")
    f.write(f"Average PSNR, Chen-Levine-Rao: {average_psnr_CLR_image} \n")
    f.write(f"Average PSNR, Huber-ROF: {average_psnr_Huber_image} \n")
    f.write(f"Average PSNR, dpROF-LPIPS: {average_psnr_dpROF_LPIPS_image} \n")
    f.write(f"Average PSNR, dpROF-noisy: {average_psnr_dpROF_noisy_image} \n")
    f.write(f"Average PSNR, dpROF-edge: {average_psnr_dpROF_edge_image} \n")
    f.write(f"Average PSNR, dpROF-SSIM: {average_psnr_dpROF_SSIM_image} \n\n")

    f.write("======================================\n\n")

print(f"Successfully added to {log_file}")





