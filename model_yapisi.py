"""
Model yapilari ve egitim/olcum yardimcilari (3. haftadan itibaren betikler bunu kullanir).

  MLP            : 784-128-10, 1. ve 2. haftadaki referans model
  LeNet5Benzeri  : 2. haftadan beri kullanilan CNN (61.706 parametre)
  DerinCNN       : 4. haftada eklenen, 3 evrisim bloklu, BatchNorm ve Dropout'lu CNN
"""
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

TOHUM = 42


# ----------------------------------------------------------------------------- Modeller
class MLP(nn.Module):
    """Tek gizli katmanli cok katmanli algilayici (784 -> 128 -> 10)."""
    def __init__(self):
        super().__init__()
        self.ag = nn.Sequential(nn.Flatten(), nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10))

    def forward(self, x):
        return self.ag(x)


class LeNet5Benzeri(nn.Module):
    """LeCun vd. (1998) LeNet-5 yapisi; tanh yerine ReLU, ortalama havuzlama yerine maks. havuzlama."""
    def __init__(self):
        super().__init__()
        self.oznitelik = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.ReLU(), nn.MaxPool2d(2),   # 28x28x6 -> 14x14x6
            nn.Conv2d(6, 16, kernel_size=5), nn.ReLU(), nn.MaxPool2d(2),             # 10x10x16 -> 5x5x16
        )
        self.siniflandirici = nn.Sequential(
            nn.Flatten(), nn.Linear(16 * 5 * 5, 120), nn.ReLU(), nn.Linear(120, 84), nn.ReLU(), nn.Linear(84, 10),
        )

    def forward(self, x):
        return self.siniflandirici(self.oznitelik(x))


class DerinCNN(nn.Module):
    """3 evrisim bloklu CNN. Her blok: 3x3 evrisim -> BatchNorm -> ReLU -> 2x2 maks. havuzlama.
    28x28x1 -> 14x14x32 -> 7x7x64 -> 3x3x128 -> 128 (Dropout 0,3) -> 10"""
    def __init__(self):
        super().__init__()

        def blok(g, c):
            return [nn.Conv2d(g, c, 3, padding=1), nn.BatchNorm2d(c), nn.ReLU(), nn.MaxPool2d(2)]

        self.oznitelik = nn.Sequential(*blok(1, 32), *blok(32, 64), *blok(64, 128))
        self.siniflandirici = nn.Sequential(nn.Flatten(), nn.Linear(128 * 3 * 3, 128), nn.ReLU(), nn.Dropout(0.3),
                                            nn.Linear(128, 10))

    def forward(self, x):
        return self.siniflandirici(self.oznitelik(x))


MODEL_SINIFLARI = {"MLP": MLP, "LeNet-5 benzeri CNN": LeNet5Benzeri, "Derin CNN": DerinCNN}


def parametre_sayisi(model):
    return sum(p.numel() for p in model.parameters())


def baslangic(sinif, tohum=TOHUM):
    """Belirli tohumla model olusturur. 2. haftada CNN'den once MLP olusturuldugu icin ayni rastgele sayi
    sirasini korumak adina MLP'nin katmanlari da (kullanilmadan) olusturulur. Boylece LeNet + tohum 42,
    2. haftadaki modelin aynisi olur."""
    torch.manual_seed(tohum); np.random.seed(tohum)
    nn.Linear(784, 128); nn.Linear(128, 10)
    return sinif()


def kayan_nokta(xb):
    """uint8 (N, 28, 28) -> float (N, 1, 28, 28), 0-1 arasi. Bellek icin veriyi uint8 tutup parti parti ceviriyoruz."""
    return xb.float().div_(255.0).unsqueeze(1)


def tahmin_et(model, x_u8, cihaz, parti=2048):
    """x_u8: uint8 tensor (N, 28, 28). Donus: tahmin edilen siniflar (numpy)."""
    model.eval()
    sonuc = []
    with torch.no_grad():
        for i in range(0, len(x_u8), parti):
            sonuc.append(model(kayan_nokta(x_u8[i:i + parti]).to(cihaz)).argmax(1).cpu())
    return torch.cat(sonuc).numpy()


# ----------------------------------------------------------------------------- Veri artirma
# Her egitim partisine her seferinde farkli rastgele degerlerle uygulanir (test verisine uygulanmaz).
ARTIRMA = {
    "dondurme_derece": 15,       # -15 ile +15 derece arasi
    "olcek": 0.15,               # %85 ile %115 arasi
    "kaydirma_piksel": 2,        # -2 ile +2 piksel arasi (yatay ve dikey)
    "kalinlik": (-0.7, 0.3),     # <0 inceltir, >0 kalinlastirir (3x3 gri tonlu asindirma / genisletme)
}


def artir(xb, gen, ayar=None):
    """xb: float (B, 1, 28, 28). Doner: artirilmis kopya. ayar=None ise ARTIRMA'daki tum yontemler uygulanir;
    bir yontemi kapatmak icin degerini 0 (kalinlik icin None) yapmak yeterlidir.

    1) Cizgi kalinligi: 3x3 maksimum (genisletme) ve minimum (asindirma) filtresi ile kalin ve ince
       surumler uretilir, goruntu rastgele bir oranla bunlardan birine dogru kaydirilir.
    2) Geometri: dondurme, olcekleme ve kaydirma tek bir afin donusumle uygulanir.
    """
    ayar = ARTIRMA if ayar is None else ayar
    B = xb.shape[0]
    r = lambda *s: torch.rand(*s, generator=gen)

    if ayar.get("kalinlik"):
        alt, ust = ayar["kalinlik"]
        a = (alt + (ust - alt) * r(B, 1, 1, 1)).to(xb.device)
        kalin = F.max_pool2d(xb, 3, 1, 1)
        ince = -F.max_pool2d(-xb, 3, 1, 1)
        xb = torch.where(a > 0, xb + a * (kalin - xb), xb + (-a) * (ince - xb))

    d, o, k = ayar.get("dondurme_derece", 0), ayar.get("olcek", 0), ayar.get("kaydirma_piksel", 0)
    if d or o or k:
        aci = (r(B) * 2 - 1) * math.radians(d)
        olcek = 1 + (r(B) * 2 - 1) * o
        kay = (r(B, 2) * 2 - 1) * k / 14.0     # 28 piksel = 2 birim -> 1 piksel = 1/14
        cos, sin = torch.cos(aci) / olcek, torch.sin(aci) / olcek
        theta = torch.stack([torch.stack([cos, -sin, kay[:, 0]], 1), torch.stack([sin, cos, kay[:, 1]], 1)], 1)
        izgara = F.affine_grid(theta.to(xb.device), list(xb.shape), align_corners=False)
        xb = F.grid_sample(xb, izgara, mode="bilinear", padding_mode="zeros", align_corners=False)
    return xb.clamp_(0, 1)


# ----------------------------------------------------------------------------- Egitim
def egit(model, x_tr, y_tr, x_val, y_val, cihaz, epoch, ad, artirma=False, parti=128, lr=1e-3, tohum=TOHUM,
         sessiz=False):
    """x_tr, x_val: uint8 tensor (N, 28, 28); y: int64 tensor. Dogrulamada en iyi epoch'un agirliklarini birakir.
    artirma: False (yok), True (ARTIRMA'daki tum yontemler) veya bir ayar sozlugu."""
    ayar = None if artirma is True else (artirma or None)
    model.to(cihaz)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    kayip_fn = nn.CrossEntropyLoss()
    gen = torch.Generator().manual_seed(tohum)
    gecmis, en_iyi, en_iyi_agirlik = [], -1.0, None
    for e in range(1, epoch + 1):
        model.train()
        sira = torch.randperm(len(x_tr), generator=gen)
        toplam, t0 = 0.0, time.time()
        for i in range(0, len(sira), parti):
            j = sira[i:i + parti]
            xb, yb = kayan_nokta(x_tr[j]).to(cihaz), y_tr[j].to(cihaz)
            if artirma:
                xb = artir(xb, gen, ayar)
            opt.zero_grad()
            kayip = kayip_fn(model(xb), yb)
            kayip.backward()
            opt.step()
            toplam += kayip.item() * len(j)
        val = float((tahmin_et(model, x_val, cihaz) == y_val.numpy()).mean())
        gecmis.append({"model": ad, "epoch": e, "egitim_kaybi": toplam / len(sira), "dogrulama_dogrulugu": val})
        if not sessiz:
            print(f"   {ad:18s} epoch {e:2d}/{epoch}  kayip={toplam / len(sira):.4f}  "
                  f"dogrulama={val * 100:.2f}%  ({time.time() - t0:.0f} sn)", flush=True)
        if val > en_iyi:
            en_iyi, en_iyi_agirlik = val, {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(en_iyi_agirlik)
    return gecmis


def dogrulama_ayir(n, oran=0.1):
    """Egitim kumesinin %10'unu dogrulama icin ayirir (2. haftadakiyle ayni yontem)."""
    sira = np.random.default_rng(TOHUM).permutation(n)
    k = int(round(n * oran))
    return sira[k:], sira[:k]
