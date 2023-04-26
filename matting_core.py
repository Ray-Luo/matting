import numpy as np
import cv2
import time
from pymatting import *


def trimap_from_mask(file_name:str, mask: np.array, guidance: np.array, epsilon: float=1e-4, radius: int =6, scale_factor: float=1.0, is_mask_rgb: bool=False, is_guidance_rgb: bool=True) -> np.array:

    # apply guided filter
    start_time = time.time()
    gf = cv2.ximgproc.guidedFilter(guidance, mask, radius=radius, eps=epsilon)
    end_time = time.time()
    latency = end_time - start_time
    print("GF Latency: {:.6f} seconds".format(latency))
    cv2.imwrite("./res/{}_gf.png".format(file_name), gf)

    # binarize mask
    mask = np.where(mask >= 127, 255.0, 0.0)

    """
    make trimap
    """
    start_time = time.time()

    diff = np.abs(gf - mask)
    diff_binarized = np.where(diff > 50, 255, 0).astype(np.uint8)

    # dilation
    d_radius = max(int(radius / 8), 3)
    kernel = np.ones((d_radius,d_radius), np.uint8)
    dilated_image = cv2.dilate(diff_binarized, kernel, iterations=1)

    unsure = np.where(dilated_image > 50)
    mask[unsure] = 128


    end_time = time.time()
    latency = end_time - start_time
    print("Trimap Latency: {:.6f} seconds".format(latency))


    cv2.imwrite("./res/{}_diff.png".format(file_name), diff)
    cv2.imwrite("./res/{}_diff_binarized.png".format(file_name), diff_binarized)
    cv2.imwrite("./res/{}_diff_dilation.png".format(file_name), dilated_image)
    cv2.imwrite("./res/{}_trimap.png".format(file_name), mask)

    return mask


def calculate_matte(file_name:str, trimap: np.array, guidance: np.array, radius: int=6):
    start_time = time.time()

    trimap /= 255.0
    guidance /= 255.0

    alpha = estimate_alpha_knn(
    guidance,
    trimap[:,:,0],
    laplacian_kwargs={"n_neighbors": [radius, radius+5],},
    cg_kwargs={"maxiter":2000})



    end_time = time.time()
    latency = end_time - start_time
    print("Matting Latency: {:.6f} seconds".format(latency))

    return alpha*255.0


mask_path = "./imgs/person1_mask.png"
guidance_path = "./imgs/person1.png"

mask = cv2.imread(mask_path, -1).astype(np.float32)#[:,:,0]
guidance = cv2.imread(guidance_path, -1).astype(np.float32)[:,:,:-1]

trimap = trimap_from_mask("person1", mask, guidance, )
print(trimap.shape, np.mean(trimap), guidance.shape, np.mean(guidance))
alpha_matte = calculate_matte("person1", trimap, guidance, radius=6)

cv2.imwrite("./res/{}_alpha_knn.png".format("person1"), alpha_matte)
