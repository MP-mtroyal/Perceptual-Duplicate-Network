import argparse
from FilterDuplicates import filterDuplicates
import os
import warnings

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perceptual Duplicate Generator")

    parser.add_argument("src", type=str, help="Source folder of the dataset to be filtered.")
    parser.add_argument("dst", type=str, help="Destination folder for the filtered dataset to be copied to.")
    parser.add_argument("--certainty", type=int, default=5, help="Level of certainty required to remove an image.")
    parser.add_argument("--modelPath", type=str, default="checkpoints/PDN.pth", help="Perceptual Duplicate Model checkpoint source path.")
    parser.add_argument("--clusterSize", type=int, default=64, help="Average size of the clusters to be use.")

    args = parser.parse_args()

    if not os.path.exists(args.src):
        print(f'Could not find dataset at {args.src}')
        exit()
    
    if not os.path.exists(args.modelPath):
        print(f'Could not find model at {args.modelPath}')
        exit()

    certainty = args.certainty
    certainty = max(1, certainty)
    certainty = min(10, certainty)

    clusterSize = args.clusterSize
    clusterSize = max(16, clusterSize)
    clusterSize = min(512, clusterSize)

    with warnings.catch_warnings(action="ignore"):
        filterDuplicates(args.src, args.dst, args.modelPath, certainty, clusterSize)

