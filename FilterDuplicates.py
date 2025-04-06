# Wrapper for deploying the PDN with clustering
import torch
from torchvision import transforms
from torchvision.io import encode_jpeg, decode_jpeg
import os

from libs.InfoPlotting import LoadingBar
from libs.ClusterFuncs import clusterDataset
from libs.DupDetectorModels import DetectorDeep
import shutil
import numpy as np

def applyJpeg(x, quality):
    inputType = x.dtype
    x = x * 255
    x = x.type(torch.uint8)
    x = decode_jpeg(encode_jpeg(x, quality))
    return x.type(inputType) / 255

def identityTransform(x):
        return x

def applyJpegTransform(x):
    inputDevice = x.device
    newXs = []
    for i in range(x.shape[0]):
        newXs.append(applyJpeg(x[i].to('cpu'), 75))
    out = torch.stack(newXs).to(inputDevice)
    return out

def filterDuplicates(src, dst, modelPath, certainty):
    
    # ========================= Setup ==================================
    # Img size to use 
    detImgSize = 96

    flipImgTransform   = transforms.RandomHorizontalFlip(1.0)
    jitterTransform    = transforms.ColorJitter(brightness=0.5, hue=0.05)
    resizeTransform    = transforms.Resize([detImgSize, detImgSize])

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Filtering Duplicates using {device}.')
    torch.set_grad_enabled(False)

    print(f'Loading Model from {modelPath}')
    model = DetectorDeep(size=detImgSize)
    try:
        model.load_state_dict(torch.load(modelPath))
    except:
        print(f'Could not load model from {modelPath}')
        exit()
    model.to(device)
    model.eval()
    
    # ===================== Clustering ===================================
    imgClusters = clusterDataset(src, clusterSize=128)
    
    # ===================== Duplicate Detection ==========================

    totalImgs = sum([len(imgCluster) for imgCluster in imgClusters])
    loadingBar = LoadingBar(totalImgs, title="Finding Duplicates", interval=5)

    for cluster in imgClusters:
        for img in cluster:
            img.imgTensor = resizeTransform(img.imgTensor)

    uniqueImgPathes = []
    transformStack = [identityTransform, applyJpegTransform, flipImgTransform, jitterTransform]


    for cluster in imgClusters:
        # Memoize Imgs in cluster to avoid double counting
        clusterMemo = set()
        duplicateNumber = 1
        for n in range(len(cluster)):
            img = cluster[n]
            loadingBar.update()
            
            if img.path not in clusterMemo:
                uniqueImgPathes.append(img.path)
                # Create stacked tensors for inference
                srcVec = torch.stack([img.imgTensor for _ in range(len(cluster) - 1)]).to(device)
                cmpImgs = []
                for i in range(len(cluster)):
                    if i != n:
                        cmpImgs.append(cluster[i].imgTensor)
                cmpVec = torch.stack(cmpImgs).to(device)
                # Create prediction arrays for compound augment detection
                preds  = np.ones([srcVec.shape[0], 1]) # discrete duplicate prediction, 0 or 1
                compoundPreds = np.ones([srcVec.shape[0], 1]) # duplicate probability range [0,1]
                
                # Predict duplicate likelihood under each transformation
                for currTransform in transformStack:
                    transformedVec = currTransform(cmpVec).to(device)
                    # Concat img stacks on color dimension
                    inputVec = torch.concat([srcVec, transformedVec], dim=1).to(device)
                    # Predict, convert predictions to numpy array
                    currPreds = model(inputVec).detach().to('cpu').numpy()

                    # Continious predictions
                    compoundPreds *= currPreds
                    # Discrete predictions
                    currPreds = np.round(np.power(currPreds, certainty))
                    preds = preds * currPreds
                
                boolPreds = (preds[:, 0] > 0.5).tolist()
                
                # Memoize Duplicates
                for i in range(len(boolPreds)):
                    if boolPreds[i]:
                        clusterMemo.add(cluster[i].path)

    loadingBar.complete()
    
    print(f'Out of {totalImgs} total Images {totalImgs - len(uniqueImgPathes)} duplicates were found.')
    print(f'A total of {len(uniqueImgPathes) / totalImgs * 100 :0.3f}% of images were kept.')

    if not os.path.exists(dst):
        print(f'Destination folder {dst} does not exist, creating new folder')
        os.makedirs(dst)

    # Copy all Unique images
    loadingBar = LoadingBar(len(uniqueImgPathes), title="Copying Unique Images", interval=500)

    for path in uniqueImgPathes:
        loadingBar.update()
        # Glob seperates by \, use this to grab image name
        imgName = path.split('\\')[-1]
        shutil.copy(path, dst + imgName)

    loadingBar.complete()
