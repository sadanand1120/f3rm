import math

# Hardcoded inputs
H = 540*2
W = 960*2
B = 1 << 13  # train_num_rays_per_batch
K = 32       # train_num_images_to_sample_from (-1 means all images)
R = 512     # train_num_times_to_repeat_images
I = 60000    # total training iterations
N = 1024      # total number of training images

# Expected # times a given pixel is sampled over all I steps
expected_pixel_samples = (I * B) / (N * H * W)

# Expected # times a given image is in the active pool
M = math.ceil(I / R)
pool_fraction = 1.0 if (K == -1 or K >= N) else (K / N)
expected_pool_inclusions = M * pool_fraction

print(f"Expected # times a given pixel is sampled over all I steps: {expected_pixel_samples}")
print(f"Expected # times a given image is in the active pool: {expected_pool_inclusions}")
