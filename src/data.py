"""Date pentru Attack Zoo. Toate imaginile ies ca tensori in [0,1], 224x224.
Normalizarea ImageNet o face modelul (vezi models.py), ca atacurile sa lucreze in spatiul [0,1]."""
import os, argparse, tarfile, urllib.request
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
IMAGENETTE_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"
# cele 10 clase Imagenette, in ordinea alfabetica a folderelor (wnid), mapate la indexul ImageNet
IMAGENETTE_TO_IMAGENET = [0, 217, 482, 491, 497, 566, 569, 571, 574, 701]

TF = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])
TF_CIFAR = transforms.Compose([transforms.Resize(224), transforms.ToTensor()])

def download_imagenette():
    os.makedirs(DATA, exist_ok=True)
    tgz = os.path.join(DATA, "imagenette2-320.tgz")
    if not os.path.exists(os.path.join(DATA, "imagenette2-320")):
        print("descarc Imagenette (~330MB)..."); urllib.request.urlretrieve(IMAGENETTE_URL, tgz)
        with tarfile.open(tgz) as t: t.extractall(DATA)
    print("ok:", os.path.join(DATA, "imagenette2-320"))

def get_loader(name: str, n: int = 1000, batch: int = 32, train: bool = False):
    if name == "imagenette":
        split = "train" if train else "val"
        ds = datasets.ImageFolder(os.path.join(DATA, "imagenette2-320", split), transform=TF)
    elif name == "cifar10":
        ds = datasets.CIFAR10(DATA, train=train, download=True, transform=TF_CIFAR)
    else:
        raise ValueError(name)
    if n and n < len(ds):
        g = torch.Generator().manual_seed(0)
        ds = Subset(ds, torch.randperm(len(ds), generator=g)[:n].tolist())
    return DataLoader(ds, batch_size=batch, shuffle=False, num_workers=2)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--download", choices=["imagenette","cifar10"])
    a = ap.parse_args()
    if a.download == "imagenette": download_imagenette()
    if a.download == "cifar10": datasets.CIFAR10(DATA, download=True); print("ok")
