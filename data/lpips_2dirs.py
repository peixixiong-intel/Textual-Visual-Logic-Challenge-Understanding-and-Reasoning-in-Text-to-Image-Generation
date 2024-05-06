import argparse
import lpips
import torch

im0 = r'C:\Users\peixixio\PycharmProjects\LatteGAN-master/data/GT/GT.png'
im1 = r'C:\Users\peixixio\PycharmProjects\LatteGAN-master/data/Result1/Result1.png'
version = '0.1'
use_gpu = False

imga = lpips.im2tensor(lpips.load_image(im0)) # RGB image from [-1,1]
imgb = lpips.im2tensor(lpips.load_image(im1))
imgb = torch.nn.functional.interpolate(imgb,size=(imga.size(2),imga.size(3)), mode='bilinear')



import lpips
loss_fn_alex = lpips.LPIPS(net='alex') # best forward scores
loss_fn_vgg = lpips.LPIPS(net='vgg') # closer to "traditional" perceptual loss, when used for optimization

d1 = loss_fn_alex(imga, imgb)
print(d1)
d2 = loss_fn_vgg(imga, imgb)
print(d2)