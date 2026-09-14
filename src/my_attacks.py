"""FGSM and PGD written from scratch, to check against torchattacks."""

import torch
import torch.nn as nn

def fgsm(model, images, labels, eps):
    images = images.clone().detach()
    labels = labels.clone().detach()

    images.requires_grad = True
    loss = nn.CrossEntropyLoss()(model(images), labels)

    grad = torch.autograd.grad(loss, images)[0]

    # move each pixel by the full budget
    adv = images + eps * grad.sign()

    #a pixel cannot leave the valid color range
    return torch.clamp(adv, 0, 1).detach()


def pgd(model, images, labels, eps, alpha, steps, random_start=True):

    images = images.clone().detach()
    labels = labels.clone().detach()
    adv = images.clone().detach()

    if random_start:
        adv = adv + torch.empty_like(adv).uniform_(-eps, eps)
        adv = torch.clamp(adv, 0, 1).detach()

    for _ in range(steps):
        adv.requires_grad = True
        loss = nn.CrossEntropyLoss()(model(adv), labels)
        grad = torch.autograd.grad(loss, adv)[0]

        adv = adv.detach() + alpha * grad.sign()

        #projection: no pixel may end up further than eps from the original
        delta = torch.clamp(adv - images, -eps, eps)
        adv = torch.clamp(images + delta, 0, 1).detach()

    return adv