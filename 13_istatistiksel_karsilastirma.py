"""
HAFTA 8: Modellerin istatistiksel karsilastirmasi

4. haftada uc model (MLP, LeNet-5 benzeri CNN, derin CNN) 5 tohumla egitilip tohumlar uzerinden esli t testi ve
her tohumda ayri McNemar testiyle karsilastirilmisti. 5. haftada da uygulamadaki iki model toplanan veride McNemar
ile karsilastirilmisti. Bu yontemlerin bir eksigi var: ayni kisinin cizimleri birbirine benzer, yani cizimler
birbirinden bagimsiz degildir. McNemar her cizimi bagimsiz saydigi icin birkac kisinin cizimlerinden gelen bir
farki oldugundan daha kesin gosterebilir. 5 tohumlu t testinin ise gucu dusuktur.

Bu betik karsilastirmayi kisi duzeyinde yapar:
  1) Modeller 4. haftadaki ayarlarla ve ayni tohumlarla yeniden egitilir, sonuclarin 4. haftadakilerle ayni
     oldugu kontrol edilir. Her model ve tohumun tahminleri modeller/hafta8/ klasorune yazilir (betik yeniden
     calistirilirsa egitim yapilmaz).
  2) Her veri kumesinde (MNIST test, dokunmatik test, toplanan veri) her model cifti icin:
     - fark: 5 tohumun ortalamasiyla iki modelin dogrulugu arasindaki fark (puan)
     - %95 guven araligi: kisiler yeniden orneklenerek (kume bootstrap, 10.000 tekrar). MNIST'te yazarlar
       bilinmedigi icin her goruntu ayri sayilir.
     - p degeri: kisi duzeyinde isaret degistirme (permutasyon) testi, 20.000 tekrar. Sifir hipotezi: bir
       kisinin cizimlerinde iki modelin sonuclari yer degistirebilir.
     - 9 karsilastirma Holm yontemiyle duzeltilir.
     Karsilastirma icin cizim duzeyindeki McNemar sonucu (tohum 42, kesin binom testi) da verilir.
  3) Uygulamadaki 8 bit modeller 32 bit modellerle esdegerlik testiyle (TOST) karsilastirilir: farkin %90 guven
     araligi +-0,5 puanin icinde kaliyorsa nicemleme dogrulugu degistirmemis kabul edilir.
  4) 5. haftadaki "uygulama modeli - derin CNN" karsilastirmasi (toplanan veri, 8 bit) kisi duzeyinde tekrarlanir.
Calistirma: python 13_istatistiksel_karsilastirma.py   (ilk calistirmada yaklasik 15-20 dk, cogu derin CNN egitimi)
"""
import json
import os
import time
from itertools import combinations

import numpy as np
import pandas as pd
import torch
from scipy.stats import binomtest

from ortak import CIKTI, MODELLER, VERI, npz_yukle
from model_yapisi import MODEL_SINIFLARI, baslangic, dogrulama_ayir, egit, tahmin_et

TOHUMLAR = [42, 43, 44, 45, 46]
KLASOR = os.path.join(MODELLER, "hafta8")
BOOT, PERM, TOST_SINIR = 10000, 20000, 0.5
KISA = {"MLP": "MLP", "LeNet-5 benzeri CNN": "LeNet", "Derin CNN": "Derin"}


def u8(x):
    return torch.from_numpy(np.ascontiguousarray(x, dtype=np.uint8))


def veriler():
    q = npz_yukle("qmnist.npz")
    d = npz_yukle("dokunmatik.npz")
    t = dict(np.load(os.path.join(VERI, "toplanan.npz"), allow_pickle=True))
    ayrim = pd.read_csv(os.path.join(CIKTI, "hafta4_dokunmatik_kisi_ayrimi.csv"), encoding="utf-8-sig")
    test = np.isin(d["kisi"], ayrim[ayrim.kume == "test"].kisi)
    kumeler = {
        "MNIST test": (q["x_test10k"], q["y_test10k"].astype(np.int64), np.arange(len(q["y_test10k"]))),
        "Dokunmatik test": (d["x"][test], d["y"][test].astype(np.int64), d["kisi"][test]),
        "Toplanan veri": (t["x"], t["y"].astype(np.int64), t["katilimci"].astype(str)),
    }
    return q, kumeler


def egit_ve_tahmin(q, kumeler):
    """4. haftadaki B bolumu: QMNIST, veri artirma yok, 10 epoch, tohumlar 42-46."""
    os.makedirs(KLASOR, exist_ok=True)
    tr, val = dogrulama_ayir(len(q["x_train"]))
    x_tr, y_tr = u8(q["x_train"][tr]), torch.from_numpy(q["y_train"][tr].astype(np.int64))
    x_val, y_val = u8(q["x_train"][val]), torch.from_numpy(q["y_train"][val].astype(np.int64))
    tahmin = {}
    for ad, sinif in MODEL_SINIFLARI.items():
        for tohum in TOHUMLAR:
            yol = os.path.join(KLASOR, f"{KISA[ad]}_{tohum}.npz")
            if not os.path.exists(yol):
                t0 = time.time()
                model = baslangic(sinif, tohum)
                egit(model, x_tr, y_tr, x_val, y_val, "cpu", 10, ad, artirma=False, tohum=tohum, sessiz=True)
                np.savez_compressed(yol, **{k: tahmin_et(model, u8(v[0]), "cpu") for k, v in kumeler.items()})
                print(f"  {ad:20s} tohum {tohum}: {time.time() - t0:5.0f} sn", flush=True)
            p = np.load(yol)
            tahmin[(ad, tohum)] = {k: p[k] for k in kumeler}
    return tahmin


def onnx_tahmin(kumeler):
    import onnxruntime as ort
    sonuc = {}
    for dosya in ("secilen.onnx", "secilen_int8.onnx", "derin.onnx", "derin_int8.onnx"):
        o = ort.InferenceSession(os.path.join("onnx", dosya), providers=["CPUExecutionProvider"])
        sonuc[dosya] = {}
        for k, (x, _, _) in kumeler.items():
            xf = x.astype(np.float32)[:, None] / 255.0
            sonuc[dosya][k] = np.concatenate([o.run(None, {"giris": xf[i:i + 1024]})[0] for i in range(0, len(xf), 1024)]).argmax(1)
    return sonuc


# ------------------------------------------------------------------------------------------------ istatistik
def kume_toplamlari(dogru, kume):
    _, ters = np.unique(kume, return_inverse=True)
    return np.bincount(ters, weights=dogru), np.bincount(ters).astype(float)


def bootstrap(da, db, n, rng, tekrar=BOOT, parca=200):
    """Kume bootstrap: kumeler yeniden orneklenir, havuzlanmis dogruluk farki (puan)."""
    k = len(n); sonuc = []
    for bas in range(0, tekrar, parca):
        i = rng.integers(0, k, size=(min(parca, tekrar - bas), k))
        sonuc.append((da[i].sum(1) - (db[i].sum(1) if db is not None else 0)) / n[i].sum(1) * 100)
    return np.concatenate(sonuc)


def isaret_testi(d, rng, tekrar=PERM, parca=500):
    """Kume duzeyinde isaret degistirme testi (iki yonlu). d: her kumedeki dogru sayisi farki."""
    gozlenen, buyuk = abs(d.sum()), 0
    for bas in range(0, tekrar, parca):
        s = rng.choice([-1.0, 1.0], size=(min(parca, tekrar - bas), len(d)))
        buyuk += int((np.abs(s @ d) >= gozlenen - 1e-9).sum())
    return (buyuk + 1) / (tekrar + 1)


def holm(p):
    p = np.asarray(p, float); sira = np.argsort(p); m = len(p); duz = np.empty(m); en = 0.0
    for r, i in enumerate(sira):
        en = max(en, min(1.0, (m - r) * p[i])); duz[i] = en
    return duz


def mcnemar_kesin(dogru_a, dogru_b):
    b = int((dogru_a & ~dogru_b).sum()); c = int((~dogru_a & dogru_b).sum())
    return b, c, (binomtest(b, b + c, 0.5).pvalue if b + c else 1.0)


def karsilastir(dogru_a, dogru_b, kume, rng, tek_a=None, tek_b=None):
    """dogru_a, dogru_b: her cizim icin dogru tahmin orani (tohumlarin ortalamasi, 0-1)."""
    da, n = kume_toplamlari(dogru_a, kume); db, _ = kume_toplamlari(dogru_b, kume)
    fark = (da.sum() - db.sum()) / n.sum() * 100
    bs = bootstrap(da, db, n, rng)
    s = {"dogruluk_a_%": round(da.sum() / n.sum() * 100, 2), "dogruluk_b_%": round(db.sum() / n.sum() * 100, 2),
         "fark_puan": round(fark, 2), "ga95_alt": round(np.percentile(bs, 2.5), 2), "ga95_ust": round(np.percentile(bs, 97.5), 2),
         "ga90_alt": round(np.percentile(bs, 5), 2), "ga90_ust": round(np.percentile(bs, 95), 2),
         "kume_sayisi": len(n), "cizim_sayisi": int(n.sum()), "p_kisi_duzeyi": isaret_testi(da - db, rng),
         "farkli_kisi_sayisi": int((np.abs(da - db) > 1e-9).sum())}
    if tek_a is not None:
        b, c, p = mcnemar_kesin(tek_a, tek_b)
        s.update({"mcnemar_yalniz_a": b, "mcnemar_yalniz_b": c, "p_mcnemar": p})
    return s


def dogruluk_ga(dogru, kume, rng):
    da, n = kume_toplamlari(dogru, kume)
    bs = bootstrap(da, None, n, rng)
    return round(da.sum() / n.sum() * 100, 2), round(np.percentile(bs, 2.5), 2), round(np.percentile(bs, 97.5), 2)


def sekil(tablo, yol):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5})
    kumeler = list(dict.fromkeys(tablo.veri_kumesi))
    fig, eksen = plt.subplots(1, len(kumeler), figsize=(9.2, 2.6), sharey=True)
    for ax, k in zip(eksen, kumeler):
        t = tablo[tablo.veri_kumesi == k].reset_index(drop=True)
        for i, r in t.iterrows():
            renk = "#2F4B7C" if r.p_holm < 0.05 else "#8C8C8C"
            ax.errorbar(r.fark_puan, i, xerr=[[r.fark_puan - r.ga95_alt], [r.ga95_ust - r.fark_puan]], fmt="o", color=renk,
                        capsize=3, ms=4, lw=1.2)
        ax.axvline(0, color="#C44E52", lw=0.8, ls="--")
        ax.set_yticks(range(len(t))); ax.set_yticklabels(t.karsilastirma)
        ax.set_title(f"{k} ({t.kume_sayisi[0]:,} {'görüntü' if k == 'MNIST test' else 'kişi'})".replace(",", "."), fontsize=9)
        ax.set_xlabel("Doğruluk farkı (puan), %95 GA")
        ax.invert_yaxis(); ax.grid(axis="x", alpha=0.3)
    fig.tight_layout(); fig.savefig(yol, dpi=200); plt.close(fig)


def main():
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    q, kumeler = veriler()
    tahmin = egit_ve_tahmin(q, kumeler)
    rng = np.random.default_rng(2026)

    # 1) 4. hafta ile ayni mi?
    eski = pd.read_csv(os.path.join(CIKTI, "hafta4_kosular.csv"), encoding="utf-8-sig")
    eski = eski[eski.bolum == "B"]
    tekrar = []
    for (ad, tohum), p in tahmin.items():
        e = eski[(eski.ad == ad) & (eski.tohum == tohum)].iloc[0]
        yeni = {"mnist_test_%": round(float((p["MNIST test"] == kumeler["MNIST test"][1]).mean()) * 100, 2),
                "dok_test_%": round(float((p["Dokunmatik test"] == kumeler["Dokunmatik test"][1]).mean()) * 100, 2)}
        tekrar.append({"model": ad, "tohum": tohum, **{f"{k}_4.hafta": e[k] for k in yeni}, **{f"{k}_simdi": v for k, v in yeni.items()}})
    tekrar = pd.DataFrame(tekrar)
    ayni = bool(np.allclose(tekrar["mnist_test_%_4.hafta"], tekrar["mnist_test_%_simdi"]) and
                np.allclose(tekrar["dok_test_%_4.hafta"], tekrar["dok_test_%_simdi"]))
    print("4. haftadaki sonuclarla ayni:", ayni)

    # 2) model ciftleri, kisi duzeyinde
    dogruluk, satirlar = [], []
    for k, (x, y, kume) in kumeler.items():
        ort_dogru = {ad: np.mean([tahmin[(ad, t)][k] == y for t in TOHUMLAR], 0) for ad in MODEL_SINIFLARI}
        for ad in MODEL_SINIFLARI:
            d_, a_, u_ = dogruluk_ga(ort_dogru[ad], kume, rng)
            tohum_d = [round(float((tahmin[(ad, t)][k] == y).mean()) * 100, 2) for t in TOHUMLAR]
            dogruluk.append({"veri_kumesi": k, "model": ad, "dogruluk_%": d_, "ga95_alt": a_, "ga95_ust": u_,
                             "tohum_en_dusuk": min(tohum_d), "tohum_en_yuksek": max(tohum_d)})
        for a, b in [("LeNet-5 benzeri CNN", "MLP"), ("Derin CNN", "MLP"), ("Derin CNN", "LeNet-5 benzeri CNN")]:
            s = karsilastir(ort_dogru[a], ort_dogru[b], kume, rng, tahmin[(a, 42)][k] == y, tahmin[(b, 42)][k] == y)
            satirlar.append({"veri_kumesi": k, "karsilastirma": f"{KISA[a]} - {KISA[b]}", **s})
    tablo = pd.DataFrame(satirlar)
    tablo["p_holm"] = holm(tablo.p_kisi_duzeyi)
    tablo["p_mcnemar_holm"] = holm(tablo.p_mcnemar)
    tablo["anlamli"] = tablo.p_holm < 0.05
    print(tablo[["veri_kumesi", "karsilastirma", "fark_puan", "ga95_alt", "ga95_ust", "p_kisi_duzeyi", "p_holm", "p_mcnemar"]].to_string())

    # 3) 8 bit - 32 bit (TOST) ve 4) 5. haftadaki karsilastirma
    o = onnx_tahmin(kumeler)
    tost, h5 = [], []
    for k, (x, y, kume) in kumeler.items():
        for m in ("secilen", "derin"):
            s = karsilastir((o[f"{m}_int8.onnx"][k] == y).astype(float), (o[f"{m}.onnx"][k] == y).astype(float), kume, rng,
                            o[f"{m}_int8.onnx"][k] == y, o[f"{m}.onnx"][k] == y)
            s["esdeger"] = bool(s["ga90_alt"] > -TOST_SINIR and s["ga90_ust"] < TOST_SINIR)
            s["farkli_tahmin"] = int((o[f"{m}_int8.onnx"][k] != o[f"{m}.onnx"][k]).sum())
            tost.append({"veri_kumesi": k, "model": "LeNet-5 benzeri CNN" if m == "secilen" else "Derin CNN", **s})
        s = karsilastir((o["derin_int8.onnx"][k] == y).astype(float), (o["secilen_int8.onnx"][k] == y).astype(float), kume, rng,
                        o["derin_int8.onnx"][k] == y, o["secilen_int8.onnx"][k] == y)
        h5.append({"veri_kumesi": k, "karsilastirma": "Derin (8 bit) - LeNet (8 bit)", **s})
    tost, h5 = pd.DataFrame(tost), pd.DataFrame(h5)
    print(tost[["veri_kumesi", "model", "fark_puan", "ga90_alt", "ga90_ust", "esdeger", "farkli_tahmin"]].to_string())
    print(h5[["veri_kumesi", "fark_puan", "ga95_alt", "ga95_ust", "p_kisi_duzeyi", "p_mcnemar", "mcnemar_yalniz_a", "mcnemar_yalniz_b", "farkli_kisi_sayisi"]].to_string())

    # toplanan veride farkin kisilere dagilimi (5. haftadaki 27 / 11 cizim kac kisiden geliyor)
    x, y, kume = kumeler["Toplanan veri"]
    da, n = kume_toplamlari((o["derin_int8.onnx"]["Toplanan veri"] == y).astype(float), kume)
    db, _ = kume_toplamlari((o["secilen_int8.onnx"]["Toplanan veri"] == y).astype(float), kume)
    kisi_farki = sorted((da - db).astype(int).tolist(), reverse=True)

    pd.DataFrame(dogruluk).to_csv(os.path.join(CIKTI, "hafta8_model_dogruluk.csv"), index=False, encoding="utf-8-sig")
    tablo.to_csv(os.path.join(CIKTI, "hafta8_model_karsilastirma.csv"), index=False, encoding="utf-8-sig")
    tost.to_csv(os.path.join(CIKTI, "hafta8_nicemleme_esdegerlik.csv"), index=False, encoding="utf-8-sig")
    tekrar.to_csv(os.path.join(CIKTI, "hafta8_4hafta_tekrari.csv"), index=False, encoding="utf-8-sig")
    ozet = {"4_hafta_ile_ayni": ayni, "bootstrap_tekrar": BOOT, "permutasyon_tekrar": PERM, "tost_sinir_puan": TOST_SINIR,
            "dogruluk": dogruluk, "karsilastirma": tablo.to_dict("records"), "nicemleme": tost.to_dict("records"),
            "hafta5_tekrari": h5.to_dict("records"), "toplanan_kisi_basina_fark_derin_eksi_lenet": kisi_farki}
    with open(os.path.join(CIKTI, "hafta8_istatistik.json"), "w", encoding="utf-8") as f:
        json.dump(ozet, f, ensure_ascii=False, indent=2, default=lambda v: v.item() if hasattr(v, "item") else str(v))
    sekil(tablo, os.path.join(CIKTI, "sekil_hafta8_model_farklari.png"))
    print("Toplanan veride kisi basina fark (derin - LeNet, dogru sayisi):", kisi_farki)


if __name__ == "__main__":
    main()
