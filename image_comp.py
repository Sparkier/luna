import numpy as np

path_luna = r"C:\Repositories\luna\raw_image_test"
path_lucid = r"C:\Users\lucaz\Documents\Fuzhi\GitHub\featurevis_experimentation\output_tests\DanielNet_raw_images_lucid"

luna_image = np.load(f"{path_luna}/DanielNet_conv2d_1_0.npy")

print(f"luna {luna_image}")

lucid_image = np.load(f"{path_lucid}/Conv2D_1_0.npy")

print(f" lucid {lucid_image}")