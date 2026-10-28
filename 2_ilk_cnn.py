"""
HAFTA 2 - ADIM 2: Ilk CNN modelini egit ve MLP ile karsilastir

Ne yapar?
  1) QMNIST egitim kumesiyle (MNIST egitim kumesinin aynisi) iki model egitir:
       - MLP (784-128-10): 1. haftadaki referans modelin PyTorch hali
       - LeNet-5 benzeri CNN: projenin ilk evrisimli sinir agi
  2) Iki modeli iki farkli test kumesinde olcer:
       - MNIST test (QMNIST test10k): literaturle karsilastirmak icin
       - Dokunmatik ekran verisi: kagit ile ekran arasindaki farki gormek icin
  3) Sonuclari, egitim egrisini ve karmasiklik matrislerini 'cikti' klasorune kaydeder

Bu hafta veri artirma YOK; once en sade haliyle temel sonucu goruyoruz.
Calistirma:  python 2_ilk_cnn.py          (varsayilan 10 epoch)
             python 2_ilk_cnn.py --epoch 5 (daha hizli deneme icin)
"""
import argparse
import os
import platform
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from sklearn.metrics import f1_score, confusion_matrix

from ortak import CIKTI, MODELLER, npz_yukle

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
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


def parametre_sayisi(model):
    return sum(p.numel() for p in model.parameters())


# ----------------------------------------------------------------------------- Yardimcilar
def tensore(x):
    return torch.from_numpy(x.astype(np.float32) / 255.0).unsqueeze(1)   # (N, 1, 28, 28), 0-1 arasi


def tahmin_et(model, x, cihaz, parti=2048):
    model.eval()
    sonuc = []
    with torch.no_grad():
        for i in range(0, len(x), parti):
            sonuc.append(model(x[i:i + parti].to(cihaz)).argmax(1).cpu())
    return torch.cat(sonuc).numpy()


def egit(model, x_tr, y_tr, x_val, y_val, cihaz, epoch, ad):
    model.to(cihaz)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    kayip_fn = nn.CrossEntropyLoss()
    gen = torch.Generator().manual_seed(TOHUM)
    gecmis, en_iyi, en_iyi_agirlik = [], -1.0, None
    for e in range(1, epoch + 1):
        model.train()
        sira = torch.randperm(len(x_tr), generator=gen)
        toplam, t0 = 0.0, time.time()
        for i in range(0, len(sira), 128):
            j = sira[i:i + 128]
            xb, yb = x_tr[j].to(cihaz), y_tr[j].to(cihaz)
            opt.zero_grad()
            kayip = kayip_fn(model(xb), yb)
            kayip.backward()
            opt.step()
            toplam += kayip.item() * len(j)
        val_dogruluk = (tahmin_et(model, x_val, cihaz) == y_val.numpy()).mean()
        gecmis.append({"model": ad, "epoch": e, "egitim_kaybi": toplam / len(sira), "dogrulama_dogrulugu": val_dogruluk})
        print(f"   {ad:4s} epoch {e}/{epoch}  kayip={toplam / len(sira):.4f}  "
              f"dogrulama={val_dogruluk * 100:.2f}%  ({time.time() - t0:.0f} sn)")
        if val_dogruluk > en_iyi:
            en_iyi, en_iyi_agirlik = val_dogruluk, {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(en_iyi_agirlik)          # dogrulamada en iyi olan epoch'u kullan
    return gecmis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epoch", type=int, default=10)
    args = ap.parse_args()

    torch.manual_seed(TOHUM); np.random.seed(TOHUM)
    cihaz = "cuda" if torch.cuda.is_available() else "cpu"
    cihaz_adi = torch.cuda.get_device_name(0) if cihaz == "cuda" else (platform.processor() or platform.machine())
    print(f"Cihaz: {cihaz} ({cihaz_adi}), PyTorch {torch.__version__}")

    q = npz_yukle("qmnist.npz")
    d = npz_yukle("dokunmatik.npz")

    # egitim kumesinin %10'unu dogrulama icin ayir (test kumelerine egitim sirasinda hic bakilmaz)
    rng = np.random.default_rng(TOHUM)
    sira = rng.permutation(len(q["x_train"]))
    val_idx, tr_idx = sira[:6000], sira[6000:]
    x_tr, y_tr = tensore(q["x_train"][tr_idx]), torch.from_numpy(q["y_train"][tr_idx])
    x_val, y_val = tensore(q["x_train"][val_idx]), torch.from_numpy(q["y_train"][val_idx])
    testler = {
        "MNIST test (QMNIST test10k)": (tensore(q["x_test10k"]), q["y_test10k"]),
        "Dokunmatik ekran": (tensore(d["x"]), d["y"]),
    }
    print(f"Egitim: {len(x_tr)}, dogrulama: {len(x_val)}, "
          + ", ".join(f"{k}: {len(v[1])}" for k, v in testler.items()))

    modeller = {"MLP": MLP(), "CNN": LeNet5Benzeri()}
    gecmis, sonuclar, tahminler = [], [], {}
    for ad, model in modeller.items():
        torch.manual_seed(TOHUM)
        print(f"\n{ad} egitiliyor ({parametre_sayisi(model):,} parametre)...")
        t0 = time.time()
        gecmis += egit(model, x_tr, y_tr, x_val, y_val, cihaz, args.epoch, ad)
        sure = time.time() - t0
        for test_ad, (x, y) in testler.items():
            p = tahmin_et(model, x, cihaz)
            tahminler[(ad, test_ad)] = p
            sonuclar.append({
                "model": ad, "parametre": parametre_sayisi(model), "test_kumesi": test_ad, "ornek": len(y),
                "dogruluk_%": round((p == y).mean() * 100, 2), "hata_sayisi": int((p != y).sum()),
                "makro_F1": round(f1_score(y, p, average="macro"), 4), "egitim_suresi_sn": round(sure),
            })
        torch.save(model.state_dict(), os.path.join(MODELLER, f"{ad.lower()}_hafta2.pt"))

    tablo = pd.DataFrame(sonuclar)
    tablo.to_csv(os.path.join(CIKTI, "sonuclar_hafta2.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(gecmis).to_csv(os.path.join(CIKTI, "egitim_gecmisi_hafta2.csv"), index=False, encoding="utf-8-sig")

    # --- egitim egrisi
    g = pd.DataFrame(gecmis)
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.5))
    for ad, renk in (("MLP", "#4C72B0"), ("CNN", "#DD8452")):
        s = g[g.model == ad]
        axs[0].plot(s.epoch, s.egitim_kaybi, "o-", color=renk, label=ad, ms=3)
        axs[1].plot(s.epoch, s.dogrulama_dogrulugu * 100, "o-", color=renk, label=ad, ms=3)
    axs[0].set_title("Eğitim kaybı", fontsize=9); axs[1].set_title("Doğrulama doğruluğu (%)", fontsize=9)
    for a in axs:
        a.set_xlabel("Epoch"); a.legend(fontsize=7, frameon=False); a.spines[["top", "right"]].set_visible(False)
        a.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    fig.tight_layout(); fig.savefig(os.path.join(CIKTI, "sekil_egitim_egrisi.png"), dpi=200); plt.close(fig)

    # --- CNN karmasiklik matrisleri (satir: gercek, sutun: tahmin; satir bazinda yuzde)
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.3))
    for a, test_ad in zip(axs, testler):
        y = testler[test_ad][1]
        cm = confusion_matrix(y, tahminler[("CNN", test_ad)], labels=range(10))
        oran = cm / cm.sum(axis=1, keepdims=True) * 100
        a.imshow(oran, cmap="Blues", vmin=0, vmax=100)
        for i in range(10):
            for j in range(10):
                if cm[i, j]:
                    a.text(j, i, cm[i, j], ha="center", va="center", fontsize=5.5,
                           color="white" if oran[i, j] > 60 else "#333333")
        a.set_xticks(range(10)); a.set_yticks(range(10))
        a.set_xlabel("Tahmin"); a.set_ylabel("Gerçek"); a.set_title(test_ad, fontsize=8.5)
    fig.tight_layout(); fig.savefig(os.path.join(CIKTI, "sekil_karmasiklik_cnn.png"), dpi=200); plt.close(fig)

    # --- dokunmatik veride CNN'in yanlis bildigi ornekler
    y = testler["Dokunmatik ekran"][1]
    p = tahminler[("CNN", "Dokunmatik ekran")]
    yanlis = np.where(p != y)[0]
    if len(yanlis):
        secim = np.random.default_rng(1).choice(yanlis, min(16, len(yanlis)), replace=False)
        fig, axs = plt.subplots(2, 8, figsize=(7.0, 2.2))
        for a in axs.ravel():
            a.axis("off")
        for a, i in zip(axs.ravel(), secim):
            a.imshow(d["x"][i], cmap="gray_r"); a.set_title(f"G:{y[i]} T:{p[i]}", fontsize=7)
        fig.tight_layout(); fig.savefig(os.path.join(CIKTI, "sekil_dokunmatik_hatalar.png"), dpi=200); plt.close(fig)

    # --- ozet (bu dosyayi rapor icin gonderin)
    pd.set_option("display.width", 200); pd.set_option("display.max_columns", 20)
    ozet = (f"HAFTA 2 SONUCLARI\nTarih: {time.strftime('%Y-%m-%d %H:%M')}\n"
            f"Cihaz: {cihaz} ({cihaz_adi}), PyTorch {torch.__version__}, epoch: {args.epoch}, tohum: {TOHUM}\n\n"
            + tablo.to_string(index=False) + "\n")
    open(os.path.join(CIKTI, "ozet_hafta2.txt"), "w", encoding="utf-8").write(ozet)
    print("\n" + ozet)
    print(f"Sekiller ve tablolar '{CIKTI}' klasorune, modeller '{MODELLER}' klasorune kaydedildi.")


if __name__ == "__main__":
    main()
