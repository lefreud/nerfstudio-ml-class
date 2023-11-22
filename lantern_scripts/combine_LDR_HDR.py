import os
import numpy as np
from tqdm import tqdm
from envmap import EnvironmentMap, rotation_matrix
from hdrio import imwrite

# Outdoor Table
LDR_dir = '/mnt/data/arch_level1/panoramas_first_100/'
HDR_dir = '/mnt/data/arch_level1/panoramas_256x512_hdr/'
output_dir = '/mnt/data/arch_level1/panoramas_LDR_HDR_small/'

imgFiles = [os.path.join(LDR_dir, f) for f in os.listdir(LDR_dir) if os.path.isfile(os.path.join(LDR_dir, f)) and f.endswith('.png')]
imgFiles.sort()

for ldr_addr in tqdm(imgFiles):
    ldr_pano = EnvironmentMap(ldr_addr, 'latlong')
    
    hdr_addr = ldr_addr.replace('.png', '.exr')
    hdr_addr = hdr_addr.replace(LDR_dir, HDR_dir)
    assert os.path.exists(hdr_addr), "HDR file does not exists!!!"
    hdr_pano = EnvironmentMap(hdr_addr, 'latlong')
    hdr_pano.resize(1920)

    out_pano = ldr_pano.data ** 2.2
    list_hdr = hdr_pano.data > 1.0
    print("list_hdr.shape: ", list_hdr.shape, "list_hdr.shape: ",  ldr_pano.data.shape)
    out_pano[list_hdr] = hdr_pano.data[list_hdr]
    
    out_addr = hdr_addr.replace(HDR_dir, output_dir)
    imwrite(out_pano, out_addr)    

print("Done!")