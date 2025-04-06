from sklearn.cluster import KMeans
import numpy as np
from libs.InfoPlotting import LoadingBar
from glob import glob
from PIL import Image
from typing import List

import torch
from torchvision import transforms



class VectorImage:
    def __init__(self, path:str):
        self.path = path
        self.pilImg    = None
        self.imgTensor = None
        self.imgVector = None
        self.cluster   = -1
    
    def setPilImg(self, img:Image):
        self.pilImg = img
    def setImgTensor(self, tensor:torch.Tensor):
        self.imgTensor = tensor
    def setImgVector(self, vec:torch.Tensor):
        self.imgVector = vec
    def setCluster(self, cluster:int):
        self.cluster = cluster


def clusterDataset(path: str, clusterSize=-1, clusterCount=-1, maxImgs=-1) -> List[List[VectorImage]]:
    allPILImages = loadImages(path, maxImgs=maxImgs)
    allImgVectors = extractImageVectors(allPILImages)
    #clean up PIL images, they are no longer needed
    for i in range(len(allPILImages)):
        allPILImages[i].pilImg = None
    clusterLabels = getClusterIndices(allImgVectors, clusterSize=clusterSize, clusterCount=clusterCount)
    clusters = [[] for _ in range(max(clusterLabels)+1)]
    for i in range(len(allPILImages)):
        allPILImages[i].setCluster(clusterLabels[i])
        clusters[clusterLabels[i]].append(allPILImages[i])

    return clusters


#=================== extractImageVectors ====================================
#   Extracts vectors from a given array of VectorImages using DinoV2
#   Populates VectorImages with their imgTensor and imgVector
# Inputs:
#       [VectorImage] : List of VectorImages to be vectorized. Vector images are assumed
#                       To have their PIL Images loaded. 
# Returns:
#       np.array : Vectors of the input images. Of shape [num images, vector size]

def extractImageVectors(imgs: List[VectorImage]) -> np.array:
    gradWasEnabled = torch.is_grad_enabled()
    vectors   = None
    batchSize = 128
    dinoImgSize = 14*7
    device    = 'cuda' if torch.cuda.is_available() else 'cpu'
    transformImg = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize([dinoImgSize, dinoImgSize])
    ])
    model     = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(device)

    loadingBar = LoadingBar(len(imgs), title="Vectorizing Images", interval=batchSize)
    torch.set_grad_enabled(False)
    imgBatch = []
    for i in range(len(imgs)):
        loadingBar.update()
        tensorImg = transformImg(imgs[i].pilImg)
        imgs[i].setImgTensor(tensorImg)
        imgBatch.append(tensorImg)
        if len(imgBatch) >= batchSize:
            inputTensor = torch.stack(imgBatch).to(device)
            outputVector = model(inputTensor).detach().cpu().numpy()
            vectors = outputVector if vectors is None else np.concatenate([vectors, outputVector])
            imgBatch = []

    if len(imgBatch) > 0:
        inputTensor = torch.stack(imgBatch).to(device)
        outputVector = model(inputTensor).detach().cpu().numpy()
        vectors = outputVector if vectors is None else np.concatenate([vectors, outputVector])

    loadingBar.complete()
    torch.set_grad_enabled(gradWasEnabled)

    for i in range(vectors.shape[0]):
        imgs[i].setImgVector(vectors[i])

    return vectors

#=================== loadImages =============================================
#   Loads all images from a given path as an array of RGB PIL images
# Inputs:
#       path (str): A string for the path of the folder containing images. Must end in /
# Returns:
#       [VectorImage] : Returns an array of VectorImages with their path and PIL images populated

def loadImages(path: str, maxImgs=-1) -> List[VectorImage]:
    imgPaths = glob(path + "*.*")
    if maxImgs > 0:
        imgPaths = imgPaths[:maxImgs]
    loadingBar = LoadingBar(len(imgPaths), title="Loading Images", interval=25)
    imgs = []
    for imgPath in imgPaths:
        loadingBar.update()
        try:
            img = Image.open(imgPath).convert("RGB")
            vecImg = VectorImage(imgPath)
            vecImg.setPilImg(img)
            imgs.append(vecImg)
        except:
            pass #Unable to open image or path is not an image file
    else:
        loadingBar.complete()
    return imgs

#===================== getClusterIndices ====================================
#   Clusters Vectors and returns their cluster indices
#       Must specify clusterSize OR clusterCount. If both are specified, clusterSize is used.
# Inputs:
#       vecs (np.array)   : shape of [batch size, vector dim], all the vectors to be clustered.
#       clusterSize (int) : average cluster size desired.
#       clusterCount (int): number of clusters desired.
# Returns:
#       [int] : List of ints indicating the cluster each vector is in. This list of the same length
#               as the number of vectors, and is parallel to the input vector array.

def getClusterIndices(vecs: np.array, clusterSize=-1, clusterCount=-1) -> List[int]:
    if clusterSize > 0:
        clusterCount = vecs.shape[0] // clusterSize
    elif clusterCount < 0:
        print("ERROR: You must specify either clusterSize or clusterCount to getClusterIndices")
        return
    
    clusters = KMeans(clusterCount, random_state=42).fit(vecs)

    return clusters.labels_