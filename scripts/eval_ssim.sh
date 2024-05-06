#!/bin/sh

python tools/ssim.py --path /home/peixixio/LatteGAN/results/experiments/exp183-geneva-LRT2I-eval/images_test/general_all --batch-size 32
python tools/ssim.py --path /home/peixixio/LatteGAN/results/experiments/exp183-geneva-LRT2I-eval/images_test/modify --batch-size 32
python tools/ssim.py --path /home/peixixio/LatteGAN/results/experiments/exp183-geneva-LRT2I-eval/images_test/ambre --batch-size 32
python tools/ssim.py --path /home/peixixio/LatteGAN/results/experiments/exp183-geneva-LRT2I-eval/images_test/infer --batch-size 32
python tools/ssim.py --path /home/peixixio/LatteGAN/results/experiments/exp183-geneva-LRT2I-eval/images_test/general --batch-size 32
python tools/ssim.py --path /home/peixixio/LatteGAN/results/experiments/exp183-geneva-LRT2I-eval/images_test/ambrr --batch-size 32
python tools/ssim.py --path /home/peixixio/LatteGAN/results/experiments/exp183-geneva-LRT2I-eval/images_test/detail --batch-size 32

