# Perceptual-Duplicate-Network
Large image dataset duplicate filtering via a CNN duplicate detector and embedded vector clustering.

Given an image dataset, the Perceptual Duplicate Network finds images that are invariant under image augmentations such as flips, compression, color jitter, crops and rotation. After duplicates are found and filtered out of the dataset, the duplicate free dataset is copied to a specified folder.

## Usage

To use the PDN on your dataset, clone the repository and install the requirements from the `requirements.txt` file. The requirements specify specific versions, though these are likely quite flexible.
Navigate to the directory of the repository and run the following command

```python PDN.py "C:/SrcFolder/" "C:/DstFolder/" --certainty 7```

Manditory positional arguments are:
- src (string) : A string containing the absolute or relative path where the dataset is stored. The dataset is assumed to be entirely images in a single folder. This path must exist. Images cannot contain "\" in their name.
- dst (string) : A string containing the destination where the filtered dataset should be copied to. The dataset will be copied with the same image names as they did in the source folder. This folder does not need to exist.

Optional arguments are:
- --certainty (int) : How certain must the model be before flagging it as a duplicate. Ranged between 1-10 inclusive. Default is 5 if unspecified.
- --modelPath (string) : Where the checkpoint for the PDN model can be found. Default is a relative path of "checkpoints/PDN.pth". Checkpoints should be weights_only pytorch files.
- --clusterSize (int)  : How many images per cluster on average. Lower numbers are faster, but less accurate. Default value is 64. Valid range is 16-512.
