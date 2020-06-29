import pycuda.autoinit
from pycuda import driver, compiler, gpuarray
import numpy as np
import cv2
import timeit

def bil_pixel(image, i, j, sigma_d, sigma_r):    
    c = 0
    s = 0
    for k in range(i-1, i+2):
        for l in range(j-1, j+2):
            g = np.exp(-((k - i) ** 2 + (l - j) ** 2) / sigma_d ** 2)
            i1 = image[k, l]/255
            i2 = image[i, j]/255
            r = np.exp(-((i1 - i2)*255) ** 2 / sigma_r ** 2)
            c += g*r
            s += g*r*image[k, l]
    result = s / c
    return result

def bilateral(image, sigma_d, sigma_r):
    n_image = np.zeros(image.shape)
    w = image.shape[0]
    h = image.shape[1]
    for i in range(1, w-1):
        for j in range(1, h-1):
            n_image[i, j] = bil_pixel(image, i, j, sigma_d, sigma_r)
    return n_image

IMG = 'rose.bmp'

image = cv2.imread(IMG, cv2.IMREAD_GRAYSCALE)

M, N = image.shape

sigma_d = 200
sigma_r = 20
gpu_result = np.zeros((M, N), dtype=np.uint32)
block = (16, 16, 1)
grid = (int(np.ceil(M/block[0])),int(np.ceil(N/block[1])))

mod = compiler.SourceModule(open("kernel.cu", "r").read())
bilateral_kernel = mod.get_function("interpolate")

start = driver.Event()
stop = driver.Event()

print("Reading GPU..")
start.record()

tex = mod.get_texref("tex")
tex.set_filter_mode(driver.filter_mode.LINEAR)
tex.set_address_mode(0, driver.address_mode.MIRROR)
tex.set_address_mode(1, driver.address_mode.MIRROR)
driver.matrix_to_texref(image.astype(np.uint32), tex, order="C")

bilateral_kernel(driver.Out(gpu_result), np.int32(M), np.int32(N), np.float32(sigma_d), np.float32(sigma_r), block=block, grid=grid, texrefs=[tex])
stop.record()
stop.synchronize()
gpu_time = stop.time_since(start)
print("GPU calculation time %.3f ms" % (gpu_time))

print("Reading CPU...")
start = timeit.default_timer()
cpu_result = bilateral(image, sigma_d, sigma_r)
cpu_time = timeit.default_timer() - start
print("CPU calculation time %.3f ms" % (cpu_time * 1e3))