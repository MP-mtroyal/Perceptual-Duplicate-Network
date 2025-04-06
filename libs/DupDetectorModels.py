import torch.nn as nn
import torch

class DupDetectorHead(nn.Module):
    def __init__(self, inSize):
        super().__init__()

        self.decision = nn.Sequential(
            nn.Linear(inSize*2, inSize // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(inSize // 2, 16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.flatten(start_dim=1)
        out = self.decision(x)
        return out

class ConvResBlock(nn.Module):
    def __init__(self, filters):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(filters),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(filters),
            nn.LeakyReLU(0.2, inplace=True),
        )
    def forward(self, x):
        out = x + self.conv(x)
        return out


class DinoDetector(nn.Module):
    def __init__(self):
        super().__init__()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.vectorizer = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(device)
        self.head       = DupDetectorHead(384)

    def forward(self, x):
        img1Vec = self.vectorizer(x[:, 0:3])
        img2Vec = self.vectorizer(x[:, 3:6])
        imgVec = torch.stack([img1Vec, img2Vec], dim=1)
        out = self.head(imgVec)
        return out

    def loadHead(self, path):
        self.head.load_state_dict(torch.load(path))

class Detector(nn.Module):
    def __init__(self, size=96):
        super().__init__()

        if size % 16 != 0:
            raise ValueError("Invalid Detector Size, must be divisible by 16")

        compressedSize = size//16

        #Decision layer input size
        self.vecSize = 128*(compressedSize)*(compressedSize)

        def ConvDownBlock(inFilters):
            layers = nn.Sequential(
                nn.Conv2d(inFilters, inFilters*2, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(inFilters*2),
                nn.LeakyReLU(0.2, inplace=True),
                ConvResBlock(inFilters*2),
            )
            return layers

        self.inConv = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=9, stride=2, padding=4),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.convBlock = nn.Sequential(
            ConvDownBlock(16),
            ConvDownBlock(32),
            ConvDownBlock(64),
        )
        self.decision = nn.Sequential(
            nn.Linear(128*(compressedSize)*(compressedSize), 256),
            nn.Sigmoid(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.inConv(x)
        vec = self.convBlock(x)
        vec = vec.view((-1, self.vecSize))
        out = self.decision(vec)
        return out
    
    def predict(self, x):
        preds = self.forward(x)
        #forward pass result is tensor of size [batch, 1], convert to size [batch]
        preds = preds[:, 0]
        #Convert to boolean array, decision point is 0.5. Duplicates are True, non-duplicates are false
        preds = preds > 0.5
        return preds

class DetectorDeep(nn.Module):
    def __init__(self, size=96):
        super().__init__()

        if size % 16 != 0:
            raise ValueError("Invalid Detector Size, must be divisible by 16")

        compressedSize = size//16

        #Decision layer input size
        self.vecSize = 128*(compressedSize)*(compressedSize)

        def ConvDownBlock(inFilters):
            layers = nn.Sequential(
                ConvResBlock(inFilters),
                nn.Conv2d(inFilters, inFilters*2, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(inFilters*2),
                nn.LeakyReLU(0.2, inplace=True),
                ConvResBlock(inFilters*2),
            )
            return layers

        self.inConv = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=9, stride=2, padding=4),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.convBlock = nn.Sequential(
            ConvDownBlock(16),
            ConvDownBlock(32),
            ConvDownBlock(64),
        )
        self.decision = nn.Sequential(
            nn.Linear(128*(compressedSize)*(compressedSize), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.inConv(x)
        vec = self.convBlock(x)
        vec = vec.view((-1, self.vecSize))
        out = self.decision(vec)
        return out
    
    def predict(self, x):
        preds = self.forward(x)
        #forward pass result is tensor of size [batch, 1], convert to size [batch]
        preds = preds[:, 0]
        #Convert to boolean array, decision point is 0.5. Duplicates are True, non-duplicates are false
        preds = preds > 0.5
        return preds




class WeaveConvResBlock(nn.Module):
    def __init__(self, filters):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
    def forward(self, x):
        out = x + self.conv(x)
        return out


class WeaveDetector(nn.Module):
    def __init__(self, size=96):
        super().__init__()

        if size % 16 != 0:
            raise ValueError("Invalid Detector Size, must be divisible by 16")

        compressedSize = size//16

        #Decision layer input size
        self.vecSize = 128*(compressedSize)*(compressedSize)

        def ConvDownBlock(inFilters):
            layers = nn.Sequential(
                nn.Conv2d(inFilters, inFilters*2, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                WeaveConvResBlock(inFilters*2),
            )
            return layers

        self.inConv = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=9, stride=2, padding=4),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.convBlock = nn.Sequential(
            ConvDownBlock(16),
            ConvDownBlock(32),
            ConvDownBlock(64),
        )
        self.decision = nn.Sequential(
            nn.Linear(128*(compressedSize)*(compressedSize), 256),
            nn.Sigmoid(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.inConv(x)
        vec = self.convBlock(x)
        vec = vec.view((-1, self.vecSize))
        out = self.decision(vec)
        return out
    
    def predict(self, x):
        preds = self.forward(x)
        #forward pass result is tensor of size [batch, 1], convert to size [batch]
        preds = preds[:, 0]
        #Convert to boolean array, decision point is 0.5. Duplicates are True, non-duplicates are false
        preds = preds > 0.5
        return preds