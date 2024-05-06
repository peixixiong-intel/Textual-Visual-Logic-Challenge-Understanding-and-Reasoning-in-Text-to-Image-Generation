import numpy as np
import cv2
import glob
import os
from optparse import OptionParser
import torch
from pytorch_msssim import ssim
from torchvision import transforms
from PIL import Image

def compute_ssim(img1, img2):
    # Convert numpy arrays to PIL Images
    img1 = Image.fromarray(img1)
    img2 = Image.fromarray(img2)

    # Transform images to tensor
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    img1 = transform(img1).unsqueeze(0)  # Add batch dimension
    img2 = transform(img2).unsqueeze(0)  # Add batch dimension

    # Move tensors to GPU if available
    if torch.cuda.is_available():
        img1 = img1.cuda()
        img2 = img2.cuda()

    # Compute SSIM
    ssim_value = ssim(img1, img2, data_range=1.0)

    return ssim_value.item()


def ssim_between_folders(images1, images2):
    if not len(images1) or not len(images2):
        print("One or both folders do not contain any images")
        return

    ssim_values = []

    for img1 in images1:
        for img2 in images2:
            ssim = compute_ssim(img1, img2)
            ssim_values.append(ssim)

    mean_ssim = np.mean(ssim_values)
    print(f"\nMean SSIM between the two folders: {mean_ssim}")


def load_images(path, split):
    image_paths = []
    image_extensions = ["png", "jpg"]

    if split == 'gt':
        postfix = '_gt'
    else:
        postfix = ''

    for sub_path in glob.glob(os.path.join(path, '*' + postfix)):
        for ext in image_extensions:
            for impath in glob.glob(os.path.join(sub_path, "*.{}".format(ext))):
                image_paths.append(impath)

    images = [cv2.imread(impath)[:, :, ::-1] for impath in image_paths]  # Convert from BGR to RGB
    return images


if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("--p", "--path", dest="path")
    parser.add_option("--multiprocessing", dest="use_multiprocessing",
                      help="Toggle use of multiprocessing for image pre-processing. Defaults to use all cores",
                      default=False,
                      action="store_true")
    parser.add_option("-b", "--batch-size", dest="batch_size",
                      help="Set batch size to use for InceptionV3 network",
                      type=int)

    options, _ = parser.parse_args()

    if options.path is None:
        print("Path is a required option. Use --path to specify it.")
    else:
        images1 = load_images(options.path, 'gt')
        images2 = load_images(options.path, 'fake')
        ssim_between_folders(images1, images2)
