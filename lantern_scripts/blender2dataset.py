import argparse
import json
import os
from pathlib import Path

import imageio
import imageio.v3 as iio
import numpy as np
from tqdm import tqdm


def exposure_tonemap(img, gamma=2.2, exposure=1.0):
    return (np.clip(np.power(img*exposure, 1/gamma), 0.0, 1.0) * 255).astype('uint8')


def main():
    os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
    imageio.plugins.freeimage.download()

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data/')
    parser.add_argument('--exposures', type=float, nargs='+', default=[0.125, 0.5, 2.0, 8.0, 32.0])
    parser.add_argument('--gamma', type=float, default=2.2)
    
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    exposures = args.exposures
    gamma = args.gamma

    # Get folders containing images
    for split in ["train", "test"]:
        # hdr images in split_hdr, ldr images in split
        split_dir = data_dir / split

        print(f"Processing {split}...")

        if not split_dir.exists():
            print(f"Skipping {split} because it does not exist.")
            continue

        split_dir_hdr = split_dir.rename(data_dir / f"{split}_hdr")
        split_dir.mkdir()

        # Open transforms_{split}.json
        with open(data_dir / f"transforms_{split}.json") as f:
            transforms = json.load(f)

        exposures_dict = {}

        for img_idx, img_file in tqdm(enumerate(split_dir_hdr.glob("*.exr"))):
            # Update transforms path
            rel_prefix = f"./{split}"
            img_name = f"t_{img_idx}"

            # Update transforms and read image
            transforms["frames"][img_idx]["file_path"] = f"{rel_prefix}/{img_name}"
            img = iio.imread(img_file)
            img = img[:, :, :3]

            for exp_idx, exp in enumerate(exposures):
                # Apply exposure and gamma correction
                tonemap = exposure_tonemap(img, gamma=gamma, exposure=exp)

                # Save image
                exp_name = f"{img_name}_{exp_idx}.png"
                exposures_dict[f"{rel_prefix}/{exp_name}"] = exp
                iio.imwrite(split_dir / exp_name, tonemap)
        
        # Save transforms and exposures
        with open(data_dir / f"transforms_{split}.json", "w") as f:
            json.dump(transforms, f, indent=4)
        
        with open(data_dir / f"exposures_{split}.json", "w") as f:
            json.dump(exposures_dict, f, indent=4)


if __name__ == "__main__":
    main()
