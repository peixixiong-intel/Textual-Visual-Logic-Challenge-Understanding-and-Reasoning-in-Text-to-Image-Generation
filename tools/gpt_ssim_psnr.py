import os
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.io import imread
from skimage.transform import resize


def load_image_in_grayscale(image_path):
    image = imread(image_path, as_gray=True)
    image = resize(image, (256, 256))  # Resize to ensure consistency
    return image


def calculate_metrics(base_path, generated_paths, ground_truth_paths):
    ssim_scores = []
    psnr_scores = []

    for gen_path, gt_path in zip(generated_paths, ground_truth_paths):
        gen_image_path = os.path.join(base_path, gen_path)
        gt_image_path = os.path.join(base_path, gt_path)

        # Skip if either image does not exist
        if not os.path.exists(gen_image_path) or not os.path.exists(gt_image_path):
            continue

        gen_image = load_image_in_grayscale(gen_image_path)
        gt_image = load_image_in_grayscale(gt_image_path)

        ssim_score = ssim(gen_image, gt_image)
        psnr_score = psnr(gen_image, gt_image)

        ssim_scores.append(ssim_score)
        psnr_scores.append(psnr_score)

    return ssim_scores, psnr_scores


# base_path = "/home/peixixio/LatteGAN/results/experiments/exp183-geneva-LRT2I-new-eval/images_test/general_all"
# base_path = "/home/peixixio/LatteGAN/results/experiments/exp183-geneva-LRT2I-new-base1-eval/images_test/general_all"
# base_path = "/home/peixixio/LatteGAN/results/experiments/exp183-geneva-LRT2I-new-base2-eval/images_test/general_all"
# base_path = "/home/peixixio/LatteGAN/results/experiments/exp183-geneva-LRT2I-new-base3-eval/images_test/general_all"
# base_path = "/home/peixixio/LatteGAN/results/experiments/exp183-geneva-LRT2I-new-latte-eval/images_test/general_all"
# base_path = "/net/csr-dgx1-04/data2/peixixio/exp-results/experiments/Comp_images/general_all"
# base_path = "/home/peixixio/LatteGAN/results/experiments/geneva_lrt2i/general_all"
# base_path = "/home/peixixio/LatteGAN/results/experiments/dalle_lrt2i/general_all"
# base_path = "/home/peixixio/LatteGAN/results/experiments/uvit_lrt2i/general_all"
# base_path = "/home/peixixio/LatteGAN/results/experiments/GPT_images/general_all"
# base_path = "/net/csr-dgx1-04/data2/peixixio/exp-results/experiments/GPT_images_small/general_all"
base_path = "/net/csr-dgx1-04/data2/peixixio/exp-results/experiments/GPT_images_small_rand/general_all"
generated_paths = [f"{str(i).zfill(6)}/0.png" for i in range(4753)]
ground_truth_paths = [f"{str(i).zfill(6)}_gt/0.png" for i in range(4753)]

ssim_scores, psnr_scores = calculate_metrics(base_path, generated_paths, ground_truth_paths)

# Calculate average scores
average_ssim = np.mean(ssim_scores)
average_psnr = np.mean(psnr_scores)

print(f"Average SSIM: {average_ssim}")
print(f"Average PSNR: {average_psnr}")

import json
with open('gpt4_rand_small_SSIM_result.json', 'w', encoding='utf-8') as f:
    json.dump(average_ssim, f, ensure_ascii=False, indent=4)
with open('gpt4_rand_small_PSNR_result.json', 'w', encoding='utf-8') as f:
    json.dump(average_psnr, f, ensure_ascii=False, indent=4)