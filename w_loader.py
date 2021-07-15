"""Inversion in W+ space."""

from tqdm import tqdm
from networks import domain_generator
import data
import pdb
import argparse
import os
import pickle
from PIL import Image
from utils import util, netinit, pbar, pidfile, renormalize
import torch

def load_data():
    if opt.corruption:
        corruptions = [opt.corruption]
    else:
        corruptions = ['gaussian_noise','shot_noise','impulse_noise','defocus_blur','glass_blur','motion_blur','zoom_blur','snow','frost','fog','brightness','contrast','elastic_transform','pixelate','jpeg_compression']
        
    dest_path_root = opt.w_path
    source_path_root = opt.images_path

    generator = domain_generator.define_generator(
            'stylegan2', 'celebahq', load_encoder=True)
    util.set_requires_grad(False, generator.generator)
    util.set_requires_grad(False, generator.encoder)
    if opt.severity:
        severity = [opt.severity]
    else:
        severity = [1, 2, 3, 4, 5]
    
    for corruption in corruptions:
        print("corruption")
        os.makedirs(opt.w_path + "/" + corruption, exist_ok=True)
        for sev in severity:
            dest_path = dest_path_root + "/" + corruption + "/" + str(sev)
            os.makedirs(dest_path, exist_ok = True)
            for label in pbar(test_labels):
                img = source_path_root + "/" + corruption + "/" + str(sev) + "/" + label
                ## perform W+ inversion on img
                image = Image.open(img)
                transform = data.get_transform('celebahq', 'imval')
                # intermediate results are saved within "optimize" function
                suffix = label.replace(".jpg", ".pth")
                if os.path.isfile(dest_path + "/" + suffix):
                    print(dest_path + "/" + suffix + "already loaded... skipping")
                    continue
                ckpt, _ = generator.optimize(transform(image).unsqueeze(0).cuda(), mask = None)
                ws = ckpt['all_z']
                
                torch.save(ws, dest_path + "/" + suffix)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--images_path', type = str, default = '/storage/gan-ensembling/dataset/celebahq-C/corrupted256_test', help = 'location of images')
    parser.add_argument('--w_path', type=str, required=True,
                        help='directory to save the optimized latents')
    parser.add_argument('--corruption', type=str,
                        help='type of corruption to invert. default all')
    parser.add_argument('--severity', type=int, help='severity of corruption')
    opt = parser.parse_args()
    print(opt)

    os.makedirs(opt.w_path, exist_ok=True)
    
    # Loading in test labels
    with open("test_labels.txt", "rb") as f:
        test_labels = pickle.load(f)
    
    load_data()