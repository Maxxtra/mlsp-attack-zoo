"""Fine-tune rapid pe CIFAR-10 (doar capul + ultimele straturi), ~10 min pe GPU.
Salveaza checkpoints/<model>_cifar10.pt, pe care models.load(..., 'cifar10') il incarca.

    python src/finetune_cifar.py --model resnet50 --epochs 3
"""
import argparse, os, torch, torch.nn as nn, timm
from data import get_loader
from models import NAMES, MEAN, STD

CK = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "checkpoints")

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", default="resnet50"); ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args(); d = a.device
    m = timm.create_model(NAMES[a.model], pretrained=True, num_classes=10).to(d)
    mean, std = MEAN.to(d), STD.to(d)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4); lossf = nn.CrossEntropyLoss()
    train = get_loader("cifar10", n=0, batch=64, train=True)
    for ep in range(a.epochs):
        m.train(); tot = correct = 0
        for x, y in train:
            x, y = x.to(d), y.to(d); out = m((x-mean)/std); loss = lossf(out, y)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += len(y); correct += (out.argmax(1)==y).sum().item()
        print(f"epoch {ep}: train acc {correct/tot:.3f}")
    os.makedirs(CK, exist_ok=True); torch.save(m.state_dict(), os.path.join(CK, f"{a.model}_cifar10.pt")); print("salvat")

if __name__ == "__main__":
    main()
