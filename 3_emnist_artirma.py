"""
HAFTA 3 - ADIM 1: CNN'i EMNIST ile egit ve veri artirmanin etkisini olc

Ne yapar?
  2. haftadaki LeNet-5 benzeri CNN'i, diger her sey ayni kalacak sekilde 4 farkli durumda egitir:

      Egitim verisi   Veri artirma
      -------------   ------------
      QMNIST          yok     (2. haftadaki model, karsilastirma icin tekrar egitilir)
      QMNIST          var
      EMNIST-Digits   yok
      EMNIST-Digits   var

  Boylece iki sorunun cevabi ayri ayri gorulur:
      - Daha buyuk veri seti (EMNIST, 4 kat) ekran cizimlerinde basariyi artiriyor mu?
      - Veri artirma (dondurme, olcekleme, kaydirma, cizgi kalinligi) basariyi artiriyor mu?

  Her model iki yerde olculur:
      - Kendi veri setinin test kumesi (QMNIST modelleri: MNIST test, EMNIST modelleri: EMNIST test)
      - Dokunmatik ekran verisi (hic egitimde kullanilmadi, dort model icin ortak olcut)

  Ayrica 2. haftada soz verdigimiz gibi dokunmatik veride en cok karisan rakam ciftlerinin
  ornek goruntulerini kaydeder (sekil_hatali_ornekler.png).

Calistirma:  python 3_emnist_artirma.py            (varsayilan 10 epoch, yaklasik 10-15 dk)
             python 3_emnist_artirma.py --epoch 1  (her seyin calistigini hizlica gormek icin)
Once 1_veri_hazirla.py calistirilmis olmali (veri klasoru).
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
from scipy.stats import chi2
from sklearn.metrics import f1_score, confusion_matrix

from ortak import CIKTI, MODELLER, VERI, npz_yukle
from model_yapisi import (TOHUM, ARTIRMA, LeNet5Benzeri, artir, dogrulama_ayir, egit, kayan_nokta,
                          parametre_sayisi, tahmin_et)

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
MAVI, TURUNCU, GRI = "#4C72B0", "#DD8452", "#6B6B6B"

DENEYLER = [
    # (kisa ad, egitim verisi, veri artirma)
    ("QMNIST", "QMNIST", False),
    ("QMNIST + artirma", "QMNIST", True),
    ("EMNIST", "EMNIST-Digits", False),
    ("EMNIST + artirma", "EMNIST-Digits", True),
]
ETIKET = {"QMNIST": "D1: QMNIST, artırma yok", "QMNIST + artirma": "D2: QMNIST, artırma var",
          "EMNIST": "D3: EMNIST, artırma yok", "EMNIST + artirma": "D4: EMNIST, artırma var"}


def baslangic_modeli():
    """Dort deneyde de ayni baslangic agirliklari kullanilir; boylece fark sadece veriden ve artirmadan gelir.
    2. haftada CNN'den once MLP olusturuldugu icin ayni rastgele sayi sirasini korumak adina
    MLP'nin katmanlari da (kullanilmadan) olusturulur. Bu sayede 'QMNIST' deneyi 2. haftadaki modelin aynisi olur."""
    torch.manual_seed(TOHUM); np.random.seed(TOHUM)
    nn.Linear(784, 128); nn.Linear(128, 10)
    return LeNet5Benzeri()


def u8(x):
    return torch.from_numpy(np.ascontiguousarray(x, dtype=np.uint8))


def en_cok_karisanlar(y, p, n=5):
    cm = confusion_matrix(y, p, labels=range(10))
    np.fill_diagonal(cm, 0)
    sira = np.dstack(np.unravel_index(np.argsort(-cm.ravel()), cm.shape))[0][:n]
    return [(int(g), int(t), int(cm[g, t])) for g, t in sira]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epoch", type=int, default=10)
    args = ap.parse_args()

    cihaz = "cuda" if torch.cuda.is_available() else "cpu"
    cihaz_adi = torch.cuda.get_device_name(0) if cihaz == "cuda" else (platform.processor() or platform.machine())
    print(f"Cihaz: {cihaz} ({cihaz_adi}), PyTorch {torch.__version__}, epoch: {args.epoch}")

    q = npz_yukle("qmnist.npz")
    d = npz_yukle("dokunmatik.npz")
    kaynaklar = [("QMNIST", q["x_train"], q["y_train"], q["x_test10k"], q["y_test10k"], "MNIST test")]
    if os.path.exists(os.path.join(VERI, "emnist_digits.npz")):
        e = npz_yukle("emnist_digits.npz")
        kaynaklar.append(("EMNIST-Digits", e["x_train"], e["y_train"], e["x_test"], e["y_test"], "EMNIST test"))
    else:
        print("UYARI: veri/emnist_digits.npz yok, EMNIST deneyleri atlaniyor (sadece QMNIST deneyleri yapilacak).")
    deneyler = [dn for dn in DENEYLER if dn[1] in {k[0] for k in kaynaklar}]

    veri = {}
    for ad, x, y, x_te, y_te, test_ad in kaynaklar:
        tr, val = dogrulama_ayir(len(x))
        veri[ad] = {"x_tr": u8(x[tr]), "y_tr": torch.from_numpy(y[tr].astype(np.int64)),
                    "x_val": u8(x[val]), "y_val": torch.from_numpy(y[val].astype(np.int64)),
                    "test_ad": test_ad, "x_te": u8(x_te), "y_te": y_te.astype(np.int64)}
        print(f"{ad}: egitim {len(tr)}, dogrulama {len(val)}, {test_ad} {len(y_te)}")
    x_dok, y_dok = u8(d["x"]), d["y"].astype(np.int64)
    print(f"Dokunmatik ekran: {len(y_dok)} cizim (sadece test)")

    sonuclar, gecmis, tahmin_dok, karisanlar = [], [], {}, []
    for kisa, veri_ad, art in deneyler:
        v = veri[veri_ad]
        model = baslangic_modeli()
        print(f"\n{kisa} egitiliyor ({parametre_sayisi(model):,} parametre)...")
        t0 = time.time()
        gecmis += egit(model, v["x_tr"], v["y_tr"], v["x_val"], v["y_val"], cihaz, args.epoch, kisa, artirma=art)
        sure = time.time() - t0
        dosya = "cnn_" + kisa.lower().replace(" + ", "_").replace("artirma", "artirmali") + "_hafta3.pt"
        torch.save(model.state_dict(), os.path.join(MODELLER, dosya))

        for test_ad, x, y in ((v["test_ad"], v["x_te"], v["y_te"]), ("Dokunmatik ekran", x_dok, y_dok)):
            p = tahmin_et(model, x, cihaz)
            sonuclar.append({
                "deney": kisa, "egitim_verisi": veri_ad, "veri_artirma": "var" if art else "yok",
                "egitim_ornegi": len(v["y_tr"]), "test_kumesi": test_ad, "ornek": len(y),
                "dogruluk_%": round(float((p == y).mean()) * 100, 2), "hata_sayisi": int((p != y).sum()),
                "hata_orani_%": round(float((p != y).mean()) * 100, 2),
                "makro_F1": round(f1_score(y, p, average="macro"), 4), "egitim_suresi_sn": round(sure),
                "model_dosyasi": dosya,
            })
            if test_ad == "Dokunmatik ekran":
                tahmin_dok[kisa] = p
                for g, t, n in en_cok_karisanlar(y, p, 8):
                    karisanlar.append({"deney": kisa, "gercek": g, "tahmin": t, "sayi": n})

    tablo = pd.DataFrame(sonuclar)
    tablo.to_csv(os.path.join(CIKTI, "sonuclar_hafta3.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(gecmis).to_csv(os.path.join(CIKTI, "egitim_gecmisi_hafta3.csv"), index=False, encoding="utf-8-sig")
    kar = pd.DataFrame(karisanlar)
    kar.to_csv(os.path.join(CIKTI, "karisan_rakamlar_hafta3.csv"), index=False, encoding="utf-8-sig")

    # dokunmatik veride her rakamin hata orani (deney bazinda)
    rakam = []
    for kisa, p in tahmin_dok.items():
        for r in range(10):
            m = y_dok == r
            rakam.append({"deney": kisa, "rakam": r, "ornek": int(m.sum()),
                          "hata_orani_%": round(float((p[m] != r).mean()) * 100, 2)})
    pd.DataFrame(rakam).to_csv(os.path.join(CIKTI, "rakam_bazinda_hata_hafta3.csv"), index=False, encoding="utf-8-sig")

    sekil_artirma_ornekleri(q)
    if len(deneyler) == len(DENEYLER):
        sekil_karsilastirma(tablo)
    sekil_rakam_bazinda(pd.DataFrame(rakam))
    sekil_egitim(pd.DataFrame(gecmis))
    sekil_hatali_ornekler(d["x"], y_dok, tahmin_dok)

    # McNemar testi: ayni veri setiyle egitilen artirmasiz ve artirmali modeller dokunmatik veride farkli mi?
    mcnemar = []
    for kok in ("QMNIST", "EMNIST"):
        a, b = tahmin_dok.get(kok), tahmin_dok.get(kok + " + artirma")
        if a is None or b is None:
            continue
        sadece_a = int(((a == y_dok) & (b != y_dok)).sum())     # artirmasiz dogru, artirmali yanlis
        sadece_b = int(((a != y_dok) & (b == y_dok)).sum())     # artirmasiz yanlis, artirmali dogru
        ki2 = (abs(sadece_a - sadece_b) - 1) ** 2 / max(sadece_a + sadece_b, 1)
        p = float(chi2.sf(ki2, 1))
        mcnemar.append(f"  {kok} - {kok} + artirma: sadece artirmasiz dogru = {sadece_a}, sadece artirmali dogru = "
                       f"{sadece_b}, ki-kare = {ki2:.1f}, p = {p:.2e}")

    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 20)
    goster = tablo.drop(columns=["model_dosyasi", "egitim_verisi", "veri_artirma"])
    ozet = (f"HAFTA 3 SONUCLARI\nTarih: {time.strftime('%Y-%m-%d %H:%M')}\n"
            f"Cihaz: {cihaz} ({cihaz_adi}), PyTorch {torch.__version__}, epoch: {args.epoch}, tohum: {TOHUM}\n"
            f"Veri artirma ayarlari: {ARTIRMA}\n\n"
            + goster.to_string(index=False) + "\n\n"
            + "Dokunmatik veride en cok karisan rakamlar (gercek -> tahmin: sayi)\n"
            + "\n".join(f"  {k:18s}: " + ", ".join(f"{r.gercek}->{r.tahmin}: {r.sayi}" for r in g.itertuples())
                        for k, g in kar.groupby("deney", sort=False)) + "\n"
            + ("\nMcNemar testi (dokunmatik veri, sureklilik duzeltmeli)\n" + "\n".join(mcnemar) + "\n"
               if mcnemar else ""))
    open(os.path.join(CIKTI, "ozet_hafta3.txt"), "w", encoding="utf-8").write(ozet)
    print("\n" + ozet)
    print(f"Sekiller ve tablolar '{CIKTI}' klasorune, modeller '{MODELLER}' klasorune kaydedildi.")


# ----------------------------------------------------------------------------- Sekiller
def sekil_artirma_ornekleri(q):
    """Ayni 8 rakamin orijinali ve 3 farkli rastgele artirilmis hali."""
    secim = [np.where(q["y_train"] == r)[0][0] for r in (0, 1, 2, 3, 4, 5, 7, 9)]
    x = kayan_nokta(u8(q["x_train"][secim]))
    gen = torch.Generator().manual_seed(7)
    satirlar = [x] + [artir(x, gen) for _ in range(3)]
    fig, axs = plt.subplots(4, 8, figsize=(6.4, 3.4))
    for i, s in enumerate(satirlar):
        for j in range(8):
            a = axs[i, j]
            a.imshow(s[j, 0].numpy(), cmap="gray_r", vmin=0, vmax=1)
            a.set_xticks([]); a.set_yticks([])
            for k in a.spines.values():
                k.set_edgecolor("#BBBBBB"); k.set_linewidth(0.6)
        axs[i, 0].set_ylabel("Orijinal" if i == 0 else f"Artırılmış {i}", fontsize=7.5)
    fig.tight_layout(h_pad=0.3, w_pad=0.2)
    fig.savefig(os.path.join(CIKTI, "sekil_artirma_ornekleri.png"), dpi=200); plt.close(fig)


def sekil_karsilastirma(tablo):
    """Dokunmatik ekran verisinde hata orani: veri seti x veri artirma."""
    t = tablo[tablo.test_kumesi == "Dokunmatik ekran"].set_index("deney")
    gruplar = [("QMNIST\n(54.000 eğitim)" if t.loc["QMNIST", "egitim_ornegi"] == 54000 else "QMNIST", "QMNIST"),
               ("EMNIST-Digits\n(216.000 eğitim)" if t.loc["EMNIST", "egitim_ornegi"] == 216000 else "EMNIST-Digits",
                "EMNIST")]
    fig, ax = plt.subplots(figsize=(5.6, 2.9))
    gen = 0.36
    for k, (etiket, kok) in enumerate(gruplar):
        for m, (son, renk, ad) in enumerate(((kok, MAVI, "Veri artırma yok"),
                                              (kok + " + artirma", TURUNCU, "Veri artırma var"))):
            deger = t.loc[son, "hata_orani_%"]
            x = k + (m - 0.5) * (gen + 0.02)
            ax.bar(x, deger, gen, color=renk, label=ad if k == 0 else None, zorder=2)
            ax.text(x, deger + 0.05, f"%{deger:.2f}".replace(".", ","), ha="center", va="bottom", fontsize=8,
                    color="#222222")
    ax.set_xticks(range(len(gruplar))); ax.set_xticklabels([g[0] for g in gruplar], fontsize=8)
    ax.set_ylabel("Dokunmatik veride hata oranı (%)", fontsize=8)
    ax.set_ylim(0, max(t["hata_orani_%"]) * 1.22)
    ax.grid(axis="y", color="#E5E5E5", lw=0.6, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=7.5, frameon=False, loc="upper right")
    fig.tight_layout(); fig.savefig(os.path.join(CIKTI, "sekil_hafta3_karsilastirma.png"), dpi=200); plt.close(fig)


def sekil_rakam_bazinda(r):
    """Dokunmatik veride her rakamin hata orani, deney bazinda yan yana cubuklar."""
    renkler = {"QMNIST": MAVI, "QMNIST + artirma": TURUNCU, "EMNIST": "#55A868", "EMNIST + artirma": "#8172B3"}
    adlar = list(dict.fromkeys(r.deney))
    fig, ax = plt.subplots(figsize=(6.4, 2.7))
    gen = 0.8 / len(adlar)
    for k, ad in enumerate(adlar):
        s = r[r.deney == ad].sort_values("rakam")
        x = s.rakam.values + (k - (len(adlar) - 1) / 2) * gen
        ax.bar(x, s["hata_orani_%"], gen * 0.92, color=renkler.get(ad, GRI), zorder=2, label=ETIKET.get(ad, ad))
        if len(adlar) <= 2:
            for xi, v in zip(x, s["hata_orani_%"]):
                if v >= 1.5:
                    ax.text(xi, v + 0.25, f"{v:.1f}".replace(".", ","), ha="center", va="bottom", fontsize=6.5,
                            color="#222222")
    ax.set_xticks(range(10)); ax.set_xlabel("Gerçek rakam", fontsize=8)
    ax.set_ylabel("Hata oranı (%)", fontsize=8)
    ax.set_ylim(0, r["hata_orani_%"].max() * 1.18)
    ax.grid(axis="y", color="#E5E5E5", lw=0.6, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    fig.tight_layout(); fig.savefig(os.path.join(CIKTI, "sekil_rakam_bazinda_hata.png"), dpi=200); plt.close(fig)


def sekil_egitim(g):
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.5))
    stiller = {"QMNIST": (MAVI, "-"), "QMNIST + artirma": (TURUNCU, "--"),
               "EMNIST": ("#55A868", "-"), "EMNIST + artirma": ("#8172B3", "--")}
    for ad, (renk, cizgi) in stiller.items():
        s = g[g.model == ad]
        if s.empty:
            continue
        etiket = ETIKET.get(ad, ad)
        axs[0].plot(s.epoch, s.egitim_kaybi, cizgi, color=renk, marker="o", ms=3, lw=1.4, label=etiket)
        axs[1].plot(s.epoch, s.dogrulama_dogrulugu * 100, cizgi, color=renk, marker="o", ms=3, lw=1.4, label=etiket)
    axs[0].set_title("Eğitim kaybı", fontsize=9); axs[1].set_title("Doğrulama doğruluğu (%)", fontsize=9)
    for a in axs:
        a.set_xlabel("Epoch"); a.spines[["top", "right"]].set_visible(False)
        a.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    axs[1].legend(fontsize=6.5, frameon=False, loc="lower right")
    fig.tight_layout(); fig.savefig(os.path.join(CIKTI, "sekil_egitim_egrisi_hafta3.png"), dpi=200); plt.close(fig)


def sekil_hatali_ornekler(x, y, tahminler, n=8):
    """2. haftanin modelinde (QMNIST, artirma yok) en cok karisan 3 cift icin ornek yanlis tahminler."""
    p = tahminler["QMNIST"]
    ciftler = en_cok_karisanlar(y, p, 3)
    fig, axs = plt.subplots(len(ciftler), n, figsize=(6.4, 0.95 * len(ciftler) + 0.2))
    axs = np.atleast_2d(axs)
    rng = np.random.default_rng(3)
    for i, (g, t, sayi) in enumerate(ciftler):
        idx = np.where((y == g) & (p == t))[0]
        secim = rng.choice(idx, min(n, len(idx)), replace=False) if len(idx) else []
        for j in range(n):
            a = axs[i, j]
            a.set_xticks([]); a.set_yticks([])
            for k in a.spines.values():
                k.set_edgecolor("#BBBBBB"); k.set_linewidth(0.6)
            if j < len(secim):
                a.imshow(x[secim[j]], cmap="gray_r")
        axs[i, 0].set_ylabel(f"{g} → {t}\n({sayi} kez)", fontsize=7.5, rotation=0, ha="right", va="center",
                             labelpad=4)
    fig.tight_layout(h_pad=0.3, w_pad=0.2)
    fig.savefig(os.path.join(CIKTI, "sekil_hatali_ornekler.png"), dpi=200); plt.close(fig)


if __name__ == "__main__":
    main()
