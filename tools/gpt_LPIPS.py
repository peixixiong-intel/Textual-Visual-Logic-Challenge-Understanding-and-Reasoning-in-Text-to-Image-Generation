import os
import lpips
import torch
from PIL import Image
from torchvision.transforms import functional as TF


def load_image_tensor(image_path):
    image = Image.open(image_path).convert('RGB').resize((128,128))
    image_tensor = TF.to_tensor(image).unsqueeze(0)  # Add batch dimension
    return image_tensor


def calculate_lpips(base_path, generated_paths, ground_truth_paths, model):
    lpips_scores = []

    for gen_path, gt_path in zip(generated_paths, ground_truth_paths):
        gen_image_path = os.path.join(base_path, gen_path)
        gt_image_path = os.path.join(base_path, gt_path)

        # Skip if either image does not exist
        if not os.path.exists(gen_image_path) or not os.path.exists(gt_image_path):
            continue

        gen_image = load_image_tensor(gen_image_path)
        gt_image = load_image_tensor(gt_image_path)

        # Move images to GPU if available
        if torch.cuda.is_available():
            gen_image = gen_image.cuda()
            gt_image = gt_image.cuda()
        lpips_score = model(gen_image, gt_image)
        lpips_scores.append(lpips_score.item())

    return lpips_scores


# Initialize LPIPS model
lpips_model = lpips.LPIPS(net='alex')  # Using AlexNet; you can choose other networks like VGG

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

lpips_scores = calculate_lpips(base_path, generated_paths, ground_truth_paths, lpips_model)

# Calculate average score
average_lpips = sum(lpips_scores) / len(lpips_scores)

print(f"Average LPIPS: {average_lpips}")
import json
with open('gpt4_rand_small_LPIPS_result.json', 'w', encoding='utf-8') as f:
    json.dump(average_lpips, f, ensure_ascii=False, indent=4)