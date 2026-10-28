"""
HAFTA 3 - ADIM 2: Veri toplama sayfasindan gelen dosyalari isle ve modelleri bu veride dene

Ne yapar?
  1) 'toplanan_veri' klasorundeki tum .json dosyalarini (veri_toplama.html'in indirdigi dosyalar) okur
  2) Her cizimin vurus koordinatlarini, dokunmatik ekran verisinde yaptigimiz gibi goruntuye cizer
     ve MNIST bicimine (28x28) getirir
  3) Katilimci, rakam ve giris turu (fare / dokunma / kalem) bazinda ozet cikarir
  4) 'modeller' klasorundeki 2. ve 3. hafta CNN modellerini bu veride olcer

Calistirma:  python 4_toplanan_veri.py
Not: Ekip ici ilk veriler oldugu icin ornek sayisi azdir; sonuclar on sonuc olarak yorumlanmalidir.
"""
import glob
import json
import os

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ortak import CIKTI, MODELLER, VERI, mnist_formatina_getir, vuruslari_ciz
from model_yapisi import MODEL_SINIFLARI, tahmin_et

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
KLASOR = "toplanan_veri"
os.makedirs(KLASOR, exist_ok=True)

MODEL_LISTESI = [
    ("2. hafta CNN (QMNIST)", "cnn_hafta2.pt", "LeNet-5 benzeri CNN"),
    ("3. hafta D1: QMNIST", "cnn_qmnist_hafta3.pt", "LeNet-5 benzeri CNN"),
    ("3. hafta D2: QMNIST + artirma", "cnn_qmnist_artirmali_hafta3.pt", "LeNet-5 benzeri CNN"),
    ("3. hafta D3: EMNIST", "cnn_emnist_hafta3.pt", "LeNet-5 benzeri CNN"),
    ("3. hafta D4: EMNIST + artirma", "cnn_emnist_artirmali_hafta3.pt", "LeNet-5 benzeri CNN"),
]
_secilen = os.path.join(MODELLER, "secilen_model_hafta4.txt")
if os.path.exists(_secilen):
    _ad = open(_secilen, encoding="utf-8").readline().strip()
    MODEL_LISTESI.append((f"4. hafta secilen model ({_ad})", "secilen_model_hafta4.pt", _ad))


def oku():
    dosyalar = sorted(glob.glob(os.path.join(KLASOR, "*.json")))
    if not dosyalar:
        raise SystemExit(f"'{KLASOR}' klasorunde .json dosyasi yok. veri_toplama.html ile toplanan dosyalari "
                         f"bu klasore koyup tekrar calistirin.")
    satirlar, goruntuler, gorulen = [], [], set()
    for yol in dosyalar:
        with open(yol, encoding="utf-8") as f:
            o = json.load(f)
        for orn in o.get("ornekler", []):
            anahtar = (o["katilimci_no"], orn["sira"])
            if anahtar in gorulen:                      # ayni dosya iki kez konduysa tekrar sayma
                continue
            gorulen.add(anahtar)
            vuruslar = [[(p[0], p[1]) for p in v] for v in orn["vuruslar"] if len(v)]
            if not vuruslar:
                continue
            goruntuler.append(mnist_formatina_getir(vuruslari_ciz(vuruslar)))
            satirlar.append({"dosya": os.path.basename(yol), "katilimci": o["katilimci_no"], "cihaz": o.get("cihaz", ""),
                             "giris_turu": orn.get("giris_turu", ""), "rakam": int(orn["rakam"]),
                             "vurus_sayisi": len(vuruslar), "sure_ms": orn.get("sure_ms", np.nan)})
    print(f"{len(dosyalar)} dosya, {len(satirlar)} cizim okundu.")
    return pd.DataFrame(satirlar), np.stack(goruntuler)


def main():
    tablo, x = oku()
    y = tablo.rakam.values.astype(np.int64)
    np.savez_compressed(os.path.join(VERI, "toplanan.npz"), x=x, y=y, katilimci=tablo.katilimci.values.astype(str),
                        giris_turu=tablo.giris_turu.values.astype(str))

    # --- ozet
    ozet = ["TOPLANAN VERI OZETI",
            f"Katilimci sayisi : {tablo.katilimci.nunique()}",
            f"Toplam cizim     : {len(tablo)}",
            f"Giris turu       : " + ", ".join(f"{k}: {v}" for k, v in tablo.giris_turu.value_counts().items()),
            f"Cihaz            : " + ", ".join(f"{k}: {v}" for k, v in tablo.cihaz.value_counts().items()),
            f"Rakam basina     : " + ", ".join(f"{r}: {(y == r).sum()}" for r in range(10)),
            f"Ortalama cizim suresi: {tablo.sure_ms.mean() / 1000:.2f} sn, ortalama vurus sayisi: "
            f"{tablo.vurus_sayisi.mean():.2f}",
            f"Murekkep orani (28x28, >50): {(x > 50).mean():.3f}", ""]
    kisi = tablo.groupby("katilimci").agg(cizim=("rakam", "size"), giris_turu=("giris_turu", lambda s: s.mode()[0]),
                                           cihaz=("cihaz", "first"))
    ozet.append("Katilimcilar:\n" + kisi.to_string())

    # --- modellerle olcum
    sonuc, tahminler = [], {}
    xt = torch.from_numpy(x)
    for ad, dosya, sinif in MODEL_LISTESI:
        yol = os.path.join(MODELLER, dosya)
        if not os.path.exists(yol):
            continue
        m = MODEL_SINIFLARI[sinif]()
        m.load_state_dict(torch.load(yol, map_location="cpu"))
        p = tahmin_et(m, xt, "cpu")
        tahminler[ad] = p
        sonuc.append({"model": ad, "ornek": len(y), "dogru": int((p == y).sum()),
                      "dogruluk_%": round(float((p == y).mean()) * 100, 2), "hata_sayisi": int((p != y).sum())})
    if sonuc:
        st = pd.DataFrame(sonuc)
        st.to_csv(os.path.join(CIKTI, "toplanan_sonuclar.csv"), index=False, encoding="utf-8-sig")
        ozet += ["", "MODELLERIN TOPLANAN VERIDEKI SONUCLARI (on sonuc, ornek sayisi az)", st.to_string(index=False)]
        for ad, p in tahminler.items():
            yanlis = [f"{g}->{t}" for g, t in zip(y[p != y], p[p != y])]
            ozet.append(f"  {ad} yanlislari: " + (", ".join(yanlis) if yanlis else "yok"))
    else:
        ozet += ["", "Model dosyasi bulunamadi; once 3_emnist_artirma.py calistirilmali."]

    tablo.to_csv(os.path.join(CIKTI, "toplanan_veri_listesi.csv"), index=False, encoding="utf-8-sig")
    metin = "\n".join(ozet) + "\n"
    open(os.path.join(CIKTI, "ozet_toplanan.txt"), "w", encoding="utf-8").write(metin)
    print("\n" + metin)

    # --- sekil: her katilimcidan her rakam icin bir ornek
    kisiler = list(kisi.index)[:6]
    fig, axs = plt.subplots(len(kisiler), 10, figsize=(6.4, 0.72 * len(kisiler) + 0.25), squeeze=False)
    for i, k in enumerate(kisiler):
        for r in range(10):
            a = axs[i, r]
            a.set_xticks([]); a.set_yticks([])
            for s in a.spines.values():
                s.set_edgecolor("#BBBBBB"); s.set_linewidth(0.6)
            idx = np.where((tablo.katilimci.values == k) & (y == r))[0]
            if len(idx):
                a.imshow(x[idx[0]], cmap="gray_r")
        axs[i, 0].set_ylabel(k, fontsize=7, rotation=0, ha="right", va="center", labelpad=4)
    fig.tight_layout(h_pad=0.2, w_pad=0.2)
    fig.savefig(os.path.join(CIKTI, "sekil_toplanan_ornekler.png"), dpi=200); plt.close(fig)
    print(f"Sekil ve tablolar '{CIKTI}' klasorune, hazir veri '{os.path.join(VERI, 'toplanan.npz')}' dosyasina kaydedildi.")


if __name__ == "__main__":
    main()
