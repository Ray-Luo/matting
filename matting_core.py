import numpy as np
import cv2
import time

mask_path = "./imgs/person1_mask.png"
guidance_path = "./imgs/person1.png"

mask = cv2.imread(mask_path, -1).astype(np.float32)[:,:,0]
guidance = cv2.imread(guidance_path, -1).astype(np.float32)[:,:,:-1]


def core_matting(file_name:str, mask: np.array, guidance: np.array, epsilon: float=1e-4, radius: int =6, scale_factor: float=1.0, is_mask_rgb: bool=False, is_guidance_rgb: bool=True) -> np.array:

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
    dilated_image = cv2.dilate(diff, kernel, iterations=1)
    # cv2.imwrite("./res_dilation.png", dilated_image)

    unsure = np.where(dilated_image > 50)
    mask[unsure] = 128
    # cv2.imwrite("./trimap_res.png", mask)

    mask /= 255.0
    guidance /= 255.0

    end_time = time.time()
    latency = end_time - start_time
    print("GF Latency: {:.6f} seconds".format(latency))

    cv2.imwrite("./res/{}_diff.png".format(file_name), diff)
    cv2.imwrite("./res/{}_diff_binarized.png".format(file_name), diff_binarized)


def calculate_matte(file_name:str, trimap: np.array, guidance: np.array, epsilon: float=1e-4, radius: int=6):

    """
    perform matting
    """
    from pymatting import *

    mask_path = "/home/luoleyouluole/matting/imgs/GT04_trimap.png"
    img_path = "/home/luoleyouluole/matting/imgs/GT04.png"

    start_time = time.time()

    mask = cv2.imread(mask_path, -1).astype(np.float64)[:,:,0]

    guidance = cv2.imread(img_path, -1).astype(np.float64)#[:,:,:-1]

    print(mask.shape, guidance.shape)

    mask /= 255.0
    guidance /= 255.0

    # alpha = estimate_alpha_cf(
    # guidance,
    # mask,
    # laplacian_kwargs={"epsilon": 1e-8},
    # cg_kwargs={"maxiter":1000})

    alpha = estimate_alpha_knn(
    guidance,
    mask,
    laplacian_kwargs={"n_neighbors": [5, 10],},
    cg_kwargs={"maxiter":2000})

    # end_time = time.time()
    # # Calculate the latency
    # latency = end_time - start_time
    # print("Latency: {:.6f} seconds".format(latency))

    # cv2.imwrite("./alpha_knn.png", alpha*255) # --> 3 seconds


    # alpha = estimate_alpha_lbdm(
    # guidance,
    # mask.astype(np.float32),
    # laplacian_kwargs={"epsilon": 1e-6},
    # cg_kwargs={"maxiter":2000})

    # end_time = time.time()
    # # Calculate the latency
    # latency = end_time - start_time
    # print("Latency: {:.6f} seconds".format(latency))

    # cv2.imwrite("./alpha_lbdm.png", alpha*255)

    # alpha = estimate_alpha_lkm(
    # guidance,
    # mask.astype(np.float32),
    # laplacian_kwargs={"epsilon": 1e-6, "radius": radius},
    # cg_kwargs={"maxiter":500})

    # end_time = time.time()
    # # Calculate the latency
    # latency = end_time - start_time
    # print("Latency: {:.6f} seconds".format(latency))

    # cv2.imwrite("./alpha_lkm.png", alpha*255) # --> 28 seconds

    # alpha = estimate_alpha_rw(
    # guidance,
    # mask.astype(np.float32),
    # laplacian_kwargs={"sigma": 0.03},
    # cg_kwargs={"maxiter":2000})

    # end_time = time.time()
    # # Calculate the latency
    # latency = end_time - start_time
    # print("Latency: {:.6f} seconds".format(latency))

    # cv2.imwrite("./alpha_rw.png", alpha*255)

    end_time = time.time()
    # Calculate the latency
    latency = end_time - start_time
    print("Latency: {:.6f} seconds".format(latency))

    cv2.imwrite("./alpha_knn.png", alpha*255) # --> 3 seconds
