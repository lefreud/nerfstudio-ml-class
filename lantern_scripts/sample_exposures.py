import argparse
import json
import shutil
from pathlib import Path

import numpy as np


def process_split(in_dir, out_dir, split):
    in_split_dir = in_dir / split
    out_split_dir = out_dir / split
    out_split_dir.mkdir(parents=True, exist_ok=True)

    with open(in_dir / f"transforms_{split}.json") as f:
        transforms_old = json.load(f)
    
    with open(in_dir / f"exposure_{split}.json") as f:
        exposures_old = json.load(f)
    
    transforms = {'camera_angle_x': transforms_old['camera_angle_x'],
                  'frames': []}
    exposures = {} if split == "train" else exposures_old.copy()

    # Copy all files if test
    if split == "test":
        shutil.copytree(in_split_dir, out_split_dir, dirs_exist_ok=True)

    for frame in transforms_old['frames']:
        fpath_noext = frame['file_path']

        if split == "train":
            fpath_new = f"{fpath_noext}"
            frame['file_path'] = fpath_new
            transforms['frames'].append(frame)

            # Choose exposure
            idx = np.random.choice([0, 2, 4])
            fpath_old = f"{fpath_noext}_{idx}.png"
            exposures[fpath_new] = exposures_old[fpath_old]

            # Copy selected image
            shutil.copy2(in_dir / fpath_old, out_dir / fpath_new)

        elif split == "test":
            for i in range(5):
                fpath_new = f"{fpath_noext}_{i}"
                frame['file_path'] = fpath_new
                transforms['frames'].append(frame.copy())
    
    with open(out_dir / f"transforms_{split}.json", "w") as f:
        json.dump(transforms, f, indent=4)

    with open(out_dir / f"exposure_{split}.json", "w") as f:
        json.dump(exposures, f, indent=4)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_dir', type=str, default='./data/chair_old/')
    parser.add_argument('--out_dir', type=str, default='./data/chair_new/')
    parser.add_argument('--random_seed', type=int, default=2)
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    random_seed = args.random_seed

    out_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(random_seed)

    for split in ["train", "test"]:
        process_split(in_dir, out_dir, split)


if __name__ == "__main__":
    main()
