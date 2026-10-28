"""
HAFTA 2 - ADIM 1: Veri setlerini indir, hazirla ve incele

Ne yapar?
  1) QMNIST'i indirir (MNIST'in kaynagindan yeniden olusturulmus hali, yazar numarasi var)
  2) EMNIST-Digits'i indirir ve yan yatik (transpoze) goruntuleri duzeltir
  3) Dokunmatik ekran rakam veri setini (Corr vd., 2017) indirir; parmak izlerini
     goruntuye cizip MNIST bicimine getirir (bizim tuvalin yapacagi isin aynisi)
  4) Her veri seti icin sayilari, sinif dagilimini ve ornek gorselleri 'cikti' klasorune kaydeder

Calistirma:  python 1_veri_hazirla.py
Not: EMNIST yaklasik 550 MB indirir, internet hizina gore birkac dakika surebilir.
"""
import os
import sys
import time
import urllib.request

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ortak import HAM, VERI, CIKTI, mnist_formatina_getir, vuruslari_ciz

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
RENK = {"QMNIST": "#4C72B0", "EMNIST-Digits": "#DD8452", "Dokunmatik": "#55A868"}


# ----------------------------------------------------------------------------- QMNIST
def qmnist_hazirla():
    from torchvision.datasets import QMNIST
    print("\n[1/3] QMNIST indiriliyor...")
    parcalar = {}
    for ad in ("train", "test10k", "test50k"):
        ds = QMNIST(root=HAM, what=ad, compat=False, download=True)
        x = ds.data.numpy().astype(np.uint8)
        hedef = ds.targets.numpy()
        parcalar[ad] = (x, hedef[:, 0].astype(np.int64), hedef[:, 2].astype(np.int64))  # sinif, yazar no
        print(f"   {ad:8s}: {len(x):6d} goruntu, {len(np.unique(hedef[:, 2])):4d} farkli yazar")
    np.savez_compressed(os.path.join(VERI, "qmnist.npz"),
                        x_train=parcalar["train"][0], y_train=parcalar["train"][1], yazar_train=parcalar["train"][2],
                        x_test10k=parcalar["test10k"][0], y_test10k=parcalar["test10k"][1],
                        yazar_test10k=parcalar["test10k"][2],
                        x_test50k=parcalar["test50k"][0], y_test50k=parcalar["test50k"][1],
                        yazar_test50k=parcalar["test50k"][2])
    return parcalar


# ----------------------------------------------------------------------------- EMNIST
def emnist_hazirla():
    from torchvision.datasets import EMNIST
    print("\n[2/3] EMNIST-Digits indiriliyor (yaklasik 550 MB)...")
    try:
        egitim = EMNIST(root=HAM, split="digits", train=True, download=True)
        test = EMNIST(root=HAM, split="digits", train=False, download=True)
    except Exception as hata:
        print("\n   EMNIST indirilemedi:", hata)
        print("   Cozum: https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip dosyasini tarayicidan indirip")
        print(f"   '{os.path.join(HAM, 'EMNIST', 'raw')}' klasorune koyun ve betigi tekrar calistirin.")
        return None
    ham_ornek = egitim.data[:6].numpy().copy()
    # EMNIST goruntuleri dosyada yan yatik (transpoze) saklanir; satir ve sutunlari yer degistiriyoruz
    x_tr = egitim.data.transpose(1, 2).numpy().astype(np.uint8)
    x_te = test.data.transpose(1, 2).numpy().astype(np.uint8)
    y_tr = egitim.targets.numpy().astype(np.int64)
    y_te = test.targets.numpy().astype(np.int64)
    print(f"   egitim : {len(x_tr):6d} goruntu")
    print(f"   test   : {len(x_te):6d} goruntu")
    np.savez_compressed(os.path.join(VERI, "emnist_digits.npz"), x_train=x_tr, y_train=y_tr, x_test=x_te, y_test=y_te)

    # duzeltme oncesi / sonrasi sekli
    fig, axs = plt.subplots(2, 6, figsize=(6.4, 2.4))
    for i in range(6):
        axs[0, i].imshow(ham_ornek[i], cmap="gray_r"); axs[1, i].imshow(x_tr[i], cmap="gray_r")
        axs[1, i].set_title(str(y_tr[i]), fontsize=8)
        for a in (axs[0, i], axs[1, i]):
            a.set_xticks([]); a.set_yticks([])
    axs[0, 0].set_ylabel("Ham", fontsize=8); axs[1, 0].set_ylabel("Düzeltilmiş", fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(CIKTI, "sekil_emnist_duzeltme.png"), dpi=200); plt.close(fig)
    return {"train": (x_tr, y_tr), "test": (x_te, y_te)}


# ----------------------------------------------------------------------------- Dokunmatik
DOKUNMATIK_URL = "https://raw.githubusercontent.com/PhilipCorr/numeral-gesture-dataset/master/"


def dokunmatik_hazirla():
    print("\n[3/3] Dokunmatik ekran veri seti indiriliyor (yaklasik 50 MB)...")
    klasor = os.path.join(HAM, "dokunmatik")
    os.makedirs(klasor, exist_ok=True)
    for f in ("glyph.csv", "stroke.csv", "touch.csv", "subject.csv"):
        yol = os.path.join(klasor, f)
        if not os.path.exists(yol):
            urllib.request.urlretrieve(DOKUNMATIK_URL + f, yol)
    glyph = pd.read_csv(os.path.join(klasor, "glyph.csv"))
    stroke = pd.read_csv(os.path.join(klasor, "stroke.csv"))
    touch = pd.read_csv(os.path.join(klasor, "touch.csv"))

    glyph = glyph[glyph.ZVALID == 1]                      # yazarlarin gecersiz isaretledigi ornekleri at
    # Veri iPhone'un Core Data veritabanindan aktarilmis: siralama Z_FOK_* sutunlarinda tutuluyor
    # (ZINDEX genel bir sayac, vurus icindeki sirayi vermiyor)
    touch = touch.sort_values(["ZSTROKE", "Z_FOK_STROKE"])
    nokta = {k: list(zip(g.ZX.values, g.ZY.values)) for k, g in touch.groupby("ZSTROKE")}
    stroke = stroke.sort_values(["ZGLYPH", "Z_FOK_GLYPH"])
    vurus = {k: [nokta[s] for s in g.Z_PK.values if s in nokta] for k, g in stroke.groupby("ZGLYPH")}

    xs, ys, kisi, parmak = [], [], [], []
    ornek_vurus = None
    t0 = time.time()
    for i, satir in enumerate(glyph.itertuples()):
        v = vurus.get(satir.Z_PK)
        if not v:
            continue
        cizim = vuruslari_ciz(v)
        xs.append(mnist_formatina_getir(cizim))
        ys.append(int(satir.ZCHARACTER)); kisi.append(int(satir.ZSUBJECT)); parmak.append(satir.ZFINGER)
        if ornek_vurus is None and satir.ZCHARACTER == 7:
            ornek_vurus = (v, cizim, xs[-1])
        if (i + 1) % 4000 == 0:
            print(f"   {i + 1} / {len(glyph)} cizim islendi ({time.time() - t0:.0f} sn)")
    x = np.stack(xs); y = np.array(ys, np.int64)
    kisi = np.array(kisi, np.int64); parmak = np.array(parmak)
    print(f"   toplam : {len(x):6d} cizim, {len(np.unique(kisi))} farkli kisi "
          f"(isaret parmagi: {(parmak == 'index').sum()}, basparmak: {(parmak == 'thumb').sum()})")
    np.savez_compressed(os.path.join(VERI, "dokunmatik.npz"), x=x, y=y, kisi=kisi, parmak=parmak)

    # dokunmatik iz -> goruntu donusumu sekli
    v, cizim, son = ornek_vurus
    fig, axs = plt.subplots(1, 3, figsize=(6.4, 2.3))
    for s in v:
        s = np.array(s); axs[0].plot(s[:, 0], s[:, 1], ".-", ms=3, lw=0.8)
    axs[0].invert_yaxis(); axs[0].set_aspect("equal"); axs[0].set_title("(a) Dokunma noktaları", fontsize=8)
    axs[1].imshow(cizim, cmap="gray_r"); axs[1].set_title("(b) Çizilmiş iz", fontsize=8)
    axs[2].imshow(son, cmap="gray_r"); axs[2].set_title("(c) MNIST biçimi 28×28", fontsize=8)
    for a in axs:
        a.set_xticks([]); a.set_yticks([])
    fig.tight_layout(); fig.savefig(os.path.join(CIKTI, "sekil_dokunmatik_donusum.png"), dpi=200); plt.close(fig)
    return {"tum": (x, y), "kisi": kisi}


# ----------------------------------------------------------------------------- Inceleme
def incele(q, e, d):
    kumeler = {"QMNIST": q["train"][:2], "Dokunmatik": d["tum"]}
    if e is not None:
        kumeler = {"QMNIST": q["train"][:2], "EMNIST-Digits": e["train"], "Dokunmatik": d["tum"]}

    # ozet tablo
    satirlar = []
    def ekle(ad, parca, x, y, yazar=None):
        sayilar = np.bincount(y, minlength=10)
        satirlar.append({
            "veri_seti": ad, "parca": parca, "goruntu_sayisi": len(x),
            "kisi_sayisi": len(np.unique(yazar)) if yazar is not None else "",
            "en_az_sinif": int(sayilar.argmin()), "en_az": int(sayilar.min()),
            "en_cok_sinif": int(sayilar.argmax()), "en_cok": int(sayilar.max()),
            "denge_orani": round(sayilar.min() / sayilar.max(), 3),
            "murekkep_orani": round(float((x > 50).mean()), 4),
            "ortalama_parlaklik": round(float(x.mean()), 2),
        })
    ekle("QMNIST", "egitim", q["train"][0], q["train"][1], q["train"][2])
    ekle("QMNIST", "test10k (=MNIST test)", q["test10k"][0], q["test10k"][1], q["test10k"][2])
    ekle("QMNIST", "test50k", q["test50k"][0], q["test50k"][1], q["test50k"][2])
    if e is not None:
        ekle("EMNIST-Digits", "egitim", *e["train"]); ekle("EMNIST-Digits", "test", *e["test"])
    ekle("Dokunmatik", "tum", d["tum"][0], d["tum"][1], d["kisi"])
    tablo = pd.DataFrame(satirlar)
    tablo.to_csv(os.path.join(CIKTI, "veri_ozeti.csv"), index=False, encoding="utf-8-sig")

    # ornek gorseller: her veri setinden her rakam icin 2 ornek
    rng = np.random.default_rng(0)
    fig, axs = plt.subplots(2 * len(kumeler), 10, figsize=(7.2, 1.55 * len(kumeler)))
    for k, (ad, (x, y)) in enumerate(kumeler.items()):
        for r in range(10):
            idx = rng.choice(np.where(y == r)[0], 2, replace=False)
            for j in range(2):
                a = axs[2 * k + j, r]
                a.imshow(x[idx[j]], cmap="gray_r"); a.set_xticks([]); a.set_yticks([])
                for s in a.spines.values():
                    s.set_edgecolor(RENK[ad]); s.set_linewidth(1.2)
        axs[2 * k, 0].set_ylabel(ad, fontsize=8, rotation=0, ha="right", va="center", labelpad=6)
    fig.subplots_adjust(left=0.16, right=0.99, top=0.99, bottom=0.01, wspace=0.08, hspace=0.08)
    fig.savefig(os.path.join(CIKTI, "sekil_veri_ornekleri.png"), dpi=200); plt.close(fig)

    # sinif dagilimi (yuzde)
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    gen = 0.8 / len(kumeler)
    for k, (ad, (x, y)) in enumerate(kumeler.items()):
        yuzde = np.bincount(y, minlength=10) / len(y) * 100
        ax.bar(np.arange(10) + (k - (len(kumeler) - 1) / 2) * gen, yuzde, gen, label=ad, color=RENK[ad])
    ax.axhline(10, color="#888888", lw=0.8, ls="--")
    ax.set_xticks(range(10)); ax.set_xlabel("Rakam"); ax.set_ylabel("Oran (%)")
    ax.set_ylim(0, 13.5); ax.legend(fontsize=7, ncol=3, frameon=False, loc="upper center")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(CIKTI, "sekil_sinif_dagilimi.png"), dpi=200); plt.close(fig)
    return tablo


def main():
    q = qmnist_hazirla()
    e = emnist_hazirla()
    d = dokunmatik_hazirla()
    tablo = incele(q, e, d)
    pd.set_option("display.width", 200); pd.set_option("display.max_columns", 20)
    metin = "VERI OZETI\n" + tablo.to_string(index=False)
    open(os.path.join(CIKTI, "veri_ozeti.txt"), "w", encoding="utf-8").write(metin + "\n")
    print("\n" + metin)
    print(f"\nTamam. Sekiller ve tablo '{CIKTI}' klasorunde, hazir veriler '{VERI}' klasorunde.")
    if e is None:
        print("UYARI: EMNIST hazirlanamadi, yukaridaki cozumu uygulayip tekrar calistirin.")
        sys.exit(1)


if __name__ == "__main__":
    main()
