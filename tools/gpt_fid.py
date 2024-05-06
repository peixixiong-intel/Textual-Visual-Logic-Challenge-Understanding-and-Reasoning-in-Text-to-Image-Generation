import os
import numpy as np
import tensorflow as tf
from scipy import linalg
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# Initialize InceptionV3 model
model = InceptionV3(include_top=False, pooling='avg', input_shape=(299, 299, 3))


def load_and_preprocess_image(image_path):
    image = load_img(image_path, target_size=(299, 299))
    image = img_to_array(image)
    image = preprocess_input(image)
    return image


def calculate_fid(model, real_img_paths, fake_img_paths, batch_size=32):
    real_features = []
    fake_features = []

    for i in range(0, len(real_img_paths), batch_size):
        real_batch_paths = real_img_paths[i:i + batch_size]
        fake_batch_paths = fake_img_paths[i:i + batch_size]

        real_batch = np.array([load_and_preprocess_image(path) for path in real_batch_paths])
        fake_batch = np.array([load_and_preprocess_image(path) for path in fake_batch_paths])

        real_batch_features = model.predict(real_batch)
        fake_batch_features = model.predict(fake_batch)

        real_features.append(real_batch_features)
        fake_features.append(fake_batch_features)

    real_features = np.concatenate(real_features, axis=0)
    fake_features = np.concatenate(fake_features, axis=0)

    mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = fake_features.mean(axis=0), np.cov(fake_features, rowvar=False)

    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    covmean = linalg.sqrtm(sigma1.dot(sigma2), disp=False)[0]
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


# Prepare the list of image paths
# base_path = "/home/peixixio/LatteGAN/results/experiments/exp183-geneva-LRT2I-new-eval/images_test/general_all"
# base_path = "/home/peixixio/LatteGAN/results/experiments/exp183-geneva-LRT2I-new-base1-eval/images_test/general_all"
# base_path = "/home/peixixio/LatteGAN/results/experiments/exp183-geneva-LRT2I-new-base2-eval/images_test/general_all"
# base_path = "/home/peixixio/LatteGAN/results/experiments/exp183-geneva-LRT2I-new-base3-eval/images_test/general_all"
# base_path = "/home/peixixio/LatteGAN/results/experiments/exp183-geneva-LRT2I-new-latte-eval/images_test/general_all"
# base_path = "/home/peixixio/LatteGAN/results/experiments/geneva_lrt2i/general_all"
# base_path = "/net/csr-dgx1-04/data2/peixixio/exp-results/experiments/Comp_images/general_all"
# base_path = "/home/peixixio/LatteGAN/results/experiments/dalle_lrt2i/general_all"
# base_path = "/home/peixixio/LatteGAN/results/experiments/uvit_lrt2i/general_all"
# base_path = "/home/peixixio/LatteGAN/results/experiments/GPT_images/general_all"
# base_path = "/net/csr-dgx1-04/data2/peixixio/exp-results/experiments/GPT_images_small/general_all"
base_path = "/net/csr-dgx1-04/data2/peixixio/exp-results/experiments/GPT_images_small_rand/general_all"
image_range = range(0, 4753)

real_img_paths = []
fake_img_paths = []

for i in image_range:
    generated_path = os.path.join(base_path, f"{str(i).zfill(6)}/0.png")
    ground_truth_path = os.path.join(base_path, f"{str(i).zfill(6)}_gt/0.png")
    if os.path.exists(generated_path) and os.path.exists(ground_truth_path):
        real_img_paths.append(ground_truth_path)
        fake_img_paths.append(generated_path)
print(len(real_img_paths), len(fake_img_paths))
# Calculate FID in batches
fid_score = calculate_fid(model, real_img_paths, fake_img_paths)
print("FID score:", fid_score)

import json
with open('gpt4_rand_small_FID_result.json', 'w', encoding='utf-8') as f:
    json.dump(fid_score, f, ensure_ascii=False, indent=4)
