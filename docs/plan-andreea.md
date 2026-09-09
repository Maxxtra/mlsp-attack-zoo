# Andreea, planul tău până pe 27 septembrie

Attack Zoo. Lista de mai jos e a ta: o iei de sus în jos, fiecare pas are termenul lui, care e întâlnirea la
care vreau să-l văd făcut. Primul pas e gândit să-l termini singur, fără să aștepți după nimeni.

*Attack Zoo · atacurile pe gradient și tabelul principal · repo: [github.com/Maxxtra/mlsp-attack-zoo](https://github.com/Maxxtra/mlsp-attack-zoo)*

1. **până vineri 11** Mediu cu torch și [torchattacks](https://github.com/Harry24k/adversarial-attacks-pytorch).
   Modele preantrenate din [timm](https://github.com/huggingface/pytorch-image-models),
   CIFAR-10 din [torchvision](https://pytorch.org/vision/stable/generated/torchvision.datasets.CIFAR10.html).
   Rulezi PGD (din librărie, deocamdată) la un singur ε pe ResNet-50 și ViT-B/16 și măsori
   acuratețea pe imagini curate și pe imagini atacate. Diferența e prima ta cifră. Citit:
   [FGSM](https://arxiv.org/abs/1412.6572) și
   [PGD](https://arxiv.org/abs/1706.06083), sunt scurte.
2. **până luni 14** Implementezi FGSM și PGD de la zero și verifici că dau la fel ca
   torchattacks; dacă nu dau, ai un bug și îl cauți. Adaugi BIM și
   [DeepFool](https://arxiv.org/abs/1511.04599) din librărie. Rulezi
   toate 4 atacurile pe toate 4 modelele (ResNet-50, EfficientNet, ViT-B/16, ConvNeXt), pe CIFAR-10,
   pe 6 valori de ε, în norma L∞. Ăsta e tabelul principal, prima versiune. Din luni, David îți
   aduce coloanele de Carlini-Wagner și AutoAttack în același tabel.
3. **până joi 17** Adaugi norma L2 și al doilea set,
   [Imagenette](https://github.com/fastai/imagenette). Refaci tabelul
   cu ambele norme și ambele seturi.
4. **până duminică 20** Îngheți tabelul. Figurile: robust accuracy în funcție de ε, o curbă
   per model, un panou per atac. Salvate în repo cu scriptul care le regenerează.
5. **20 - 25 sep** Scrii metoda, setup-ul experimental și secțiunea cu tabelul principal.

## Întâlnirile

| Când | Ce vreau să văd |
|---|---|
| **Vineri 11 sep, 20:00** | Primul tău pas făcut și rulând. Trimitem abstractele. |
| **Luni 14 sep, seara** | Al doilea pas. De aici task-urile se leagă cu ale colegilor. |
| **Joi 17 sep, seara** | Grosul experimentelor. |
| **Duminică 20 sep** | Experimentele înghețate. Toate tabelele și figurile în repo. După ziua asta nu mai atingem experimentele. |
| **20 - 25 sep** | Scrii secțiunea ta în `paper/` din repo, ca markdown, direct din `results/`. Miercuri 24 ne vedem pe draft. |
| **25 - 27 sep** | Alex face polish și încarcă. |

## Când ai nevoie de mine

La întâlnirile de mai sus și atât. Dacă te blochezi între ele, scrii în canal unde te-ai oprit și treci la
următorul pas din listă; nu stai pe loc așteptând răspuns. Regula de „gata": un pas e gata când există un
script care îl rulează de la zero și un CSV sau o figură comisă în repo.

---
*Grup de cercetare MLSP · mentor: Alex Deonise · coordonator: Răzvan Rughiniș · RoEduNet 2026*
