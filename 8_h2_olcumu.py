"""
HAFTA 5 - ADIM 3: Ekibin topladigi veride ilk H2 olcumu

H2 hedefi: En az 20 kisiden toplanan ve egitimde kullanilmayan kisilere ait tuval cizimlerinde en az %95 dogruluk.

Ne yapar?
  1) 4_toplanan_veri.py'nin hazirladigi veri/toplanan.npz dosyasini okur
  2) Uygulamadaki ONNX modelleriyle (onnx/ klasoru) tahmin yapar
  3) Dogruluk ve %95 guven araligi (Clopper-Pearson), katilimci bazinda ve giris turune gore (dokunma / fare)
     dogruluk, rakam bazinda hatalar, iki model arasinda McNemar testi
  4) Sekiller: katilimci bazinda dogruluk, yanlis tahmin edilen cizimler

Calistirma: python 4_toplanan_veri.py  (once)
            python 8_h2_olcumu.py
"""
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

from ortak import CIKTI, VERI

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})
HEDEF = 95.0


def guven_araligi(k, n, a=0.05):
    alt = stats.beta.ppf(a / 2, k, n - k + 1) if k > 0 else 0.0
    ust = stats.beta.ppf(1 - a / 2, k + 1, n - k) if k < n else 1.0
    return alt * 100, ust * 100


def mcnemar(d1, d2):
    b, c = int((d1 & ~d2).sum()), int((~d1 & d2).sum())
    if b + c == 0:
        return b, c, 1.0
    return b, c, float(stats.chi2.sf((abs(b - c) - 1) ** 2 / (b + c), 1))


def main():
    import onnxruntime as ort
    t = np.load(os.path.join(VERI, "toplanan.npz"), allow_pickle=True)
    x, y = t["x"], t["y"]
    kisi, tur = t["katilimci"].astype(str), t["giris_turu"].astype(str)
    bilgi = json.load(open(os.path.join("onnx", "bilgi.json"), encoding="utf-8"))
    xf = x.astype(np.float32)[:, None] / 255

    satirlar, dogrular, ozet = [], {}, []
    for dosya in ("secilen_int8.onnx", "derin_int8.onnx"):
        if dosya not in bilgi:
            continue
        o = ort.InferenceSession(os.path.join("onnx", dosya), providers=["CPUExecutionProvider"])
        p = o.run(None, {"giris": xf})[0].argmax(1)
        d = p == y
        ad = bilgi[dosya]["model"] + " (8 bit)"
        dogrular[ad] = (d, p)
        alt, ust = guven_araligi(int(d.sum()), len(d))
        kb = pd.Series(d).groupby(kisi).mean() * 100
        satir = {"model": ad, "katilimci": len(kb), "cizim": len(d), "dogru": int(d.sum()),
                 "dogruluk_%": round(d.mean() * 100, 2), "guven_araligi_%95": f"{alt:.2f}-{ust:.2f}",
                 "hedef_%95_saglandi_mi": "evet" if d.mean() * 100 >= HEDEF else "hayir",
                 "en_dusuk_katilimci_%": round(kb.min(), 1), "medyan_katilimci_%": round(kb.median(), 1),
                 "%95_alti_katilimci": int((kb < HEDEF).sum())}
        for g in sorted(set(tur)):
            m = tur == g
            satir[f"{g}_cizim"] = int(m.sum())
            satir[f"{g}_dogruluk_%"] = round(d[m].mean() * 100, 2)
        satirlar.append(satir)
        hata = pd.Series([f"{a}->{b}" for a, b in zip(y[~d], p[~d])]).value_counts()
        ozet.append(f"{ad} en sik hatalar: " + ", ".join(f"{k} ({v})" for k, v in hata.head(8).items()))
        ozet.append(f"{ad} rakam bazinda hata sayisi: " +
                    ", ".join(f"{r}: {int((~d & (y == r)).sum())}" for r in range(10)))
        ozet.append(f"{ad} katilimci bazinda dogruluk: " +
                    ", ".join(f"{k}: {v:.0f}" for k, v in kb.sort_values().items()))

    tablo = pd.DataFrame(satirlar)
    tablo.to_csv(os.path.join(CIKTI, "hafta5_h2.csv"), index=False, encoding="utf-8-sig")
    if len(dogrular) == 2:
        (a1, (d1, _)), (a2, (d2, _)) = list(dogrular.items())
        b, c, pv = mcnemar(d1, d2)
        ozet.append(f"McNemar ({a1} - {a2}): yalniz birincinin dogru bildigi {b}, yalniz ikincinin {c}, p = {pv:.4f}")
    # ayni kisinin hatalari: 9 hatalarinin kac kisiden geldigi
    d_app, p_app = list(dogrular.values())[0]
    for r in range(10):
        m = (~d_app) & (y == r)
        if m.sum() >= 5:
            ozet.append(f"Uygulama modelinde {r} rakamindaki {int(m.sum())} hata {len(set(kisi[m]))} katilimcidan geliyor: "
                        + ", ".join(f"{k} ({v})" for k, v in pd.Series(kisi[m]).value_counts().items()))
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    metin = "HAFTA 5 - TOPLANAN VERIDE H2 OLCUMU\n\n" + tablo.T.to_string(header=False) + "\n\n" + "\n".join(ozet) + "\n"
    open(os.path.join(CIKTI, "ozet_hafta5_h2.txt"), "w", encoding="utf-8").write(metin)
    print(metin)

    # sekil 1: katilimci bazinda dogruluk
    adlar = list(dogrular)
    kb = pd.DataFrame({ad: pd.Series(d).groupby(kisi).mean() * 100 for ad, (d, _) in dogrular.items()})
    kb["tur"] = pd.Series(tur).groupby(kisi).agg(lambda s: s.mode()[0])
    kb = kb.sort_values(adlar[0])
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    xs = np.arange(len(kb)); w = 0.4 if len(adlar) == 2 else 0.7
    renk = ["#2F4B7C", "#B8BEC7"]
    for i, ad in enumerate(adlar):
        ax.bar(xs + (i - (len(adlar) - 1) / 2) * w, kb[ad], w, color=renk[i], label=ad)
    ax.axhline(HEDEF, color="#C0392B", lw=0.9, ls="--"); ax.text(len(kb) - 0.4, HEDEF + 0.3, "H2: %95", color="#C0392B",
                                                                ha="right", va="bottom", fontsize=7)
    ax.set_xticks(xs); ax.set_xticklabels([f"{k} ({'D' if t_ == 'touch' else 'F'})" for k, t_ in
                                           zip(kb.index, kb.tur)], fontsize=6, rotation=60, ha="right")
    ax.set_ylim(75, 101); ax.set_ylabel("Doğruluk (%)")
    ax.legend(fontsize=7, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(CIKTI, "sekil_hafta5_h2_katilimci.png"), dpi=200); plt.close(fig)

    # sekil 2: uygulama modelinin yanlis tahmin ettigi cizimler
    idx = np.where(~d_app)[0]
    n = len(idx); sut = 10; sat = int(np.ceil(n / sut))
    if n:
        fig, axs = plt.subplots(sat, sut, figsize=(6.4, 0.78 * sat + 0.1), squeeze=False)
        for j, a in enumerate(axs.flat):
            a.set_xticks([]); a.set_yticks([])
            for s in a.spines.values():
                s.set_visible(False)
            if j < n:
                i = idx[j]
                a.imshow(x[i], cmap="gray_r")
                a.set_title(f"{y[i]} → {p_app[i]}", fontsize=6.5, pad=1.5)
        fig.tight_layout(h_pad=0.3, w_pad=0.2)
        fig.savefig(os.path.join(CIKTI, "sekil_hafta5_h2_hatalar.png"), dpi=200); plt.close(fig)
    print(f"Sonuclar '{CIKTI}' klasorune kaydedildi.")


if __name__ == "__main__":
    main()
