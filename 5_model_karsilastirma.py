"""
HAFTA 4: Model karsilastirmasi, veri artirma yontemlerinin tek tek denenmesi ve veri kaynagi deneyi

Ne yapar?
  0) Dokunmatik ekran verisini KISI BAZINDA ikiye ayirir:
       - dokunmatik DOGRULAMA (kisilerin %20'si): ayar ve model SECMEK icin kullanilir
       - dokunmatik TEST (kalan %80): sadece en sondaki olcum icin kullanilir, hicbir secimde kullanilmaz
     Ayni kisinin cizimleri iki tarafa birden dusmez.
  A) Veri artirma yontemleri: LeNet-5 benzeri CNN, QMNIST ile, her artirma yontemi tek basina, 5 tohumla.
     Dokunmatik dogrulamada ise yarayan yontemler birlestirilip bir kez daha denenir.
  B) Model karsilastirmasi: MLP, LeNet-5 benzeri CNN ve Derin CNN, A'da secilen artirma ayariyla, 5 tohumla.
  C) Veri kaynagi: LeNet-5 benzeri CNN; QMNIST, EMNIST-Digits ve QMNIST + EMNIST birlikte, 5 tohumla.

  Her kosunun sonucu hemen 'cikti/hafta4_kosular.csv' dosyasina yazilir. Betik yarida kesilirse tekrar
  calistirildiginda biten kosulari atlar, kaldigi yerden devam eder.

Calistirma:  python 5_model_karsilastirma.py                (tamami, yaklasik 35-45 dk)
             python 5_model_karsilastirma.py --hizli        (her seyin calistigini 2-3 dakikada gormek icin)
             python 5_model_karsilastirma.py --bolum B      (sadece bir bolum: A, B veya C)
"""
import argparse
import os
import platform
import time
from itertools import combinations

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import chi2, ttest_rel
from sklearn.metrics import confusion_matrix

from ortak import CIKTI, MODELLER, VERI, npz_yukle
from model_yapisi import ARTIRMA, MODEL_SINIFLARI, baslangic, dogrulama_ayir, egit, parametre_sayisi, tahmin_et

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
MAVI, TURUNCU, YESIL, MOR, GRI = "#4C72B0", "#DD8452", "#55A868", "#8172B3", "#8C8C8C"
TOHUMLAR = [42, 43, 44, 45, 46]
KLASOR = os.path.join(MODELLER, "hafta4")
CIK = CIKTI
KOSU_CSV = os.path.join(CIK, "hafta4_kosular.csv")

ARTIRMA_TEKLI = [
    ("Yok", {}),
    ("Döndürme", {"dondurme_derece": ARTIRMA["dondurme_derece"]}),
    ("Ölçekleme", {"olcek": ARTIRMA["olcek"]}),
    ("Kaydırma", {"kaydirma_piksel": ARTIRMA["kaydirma_piksel"]}),
    ("Çizgi kalınlığı", {"kalinlik": ARTIRMA["kalinlik"]}),
    ("Hepsi (3. hafta)", dict(ARTIRMA)),
]


def u8(x):
    return torch.from_numpy(np.ascontiguousarray(x, dtype=np.uint8))


def guvenli(metin):
    tablo = str.maketrans("çğıöşüÇĞİÖŞÜ +()", "cgiosuCGIOSU_p__")
    return metin.translate(tablo).replace(".", "").replace(":", "").replace("__", "_").strip("_")


def mcnemar(dogru_a, dogru_b):
    b = int((dogru_a & ~dogru_b).sum()); c = int((~dogru_a & dogru_b).sum())
    ki2 = (abs(b - c) - 1) ** 2 / max(b + c, 1)
    return b, c, float(chi2.sf(ki2, 1))


def eslestirilmis_t(a, b):
    """Ayni tohumlarla elde edilen iki sonuc dizisi icin eslestirilmis t testi (p degeri)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or np.allclose(a - b, (a - b)[0]):
        return float("nan")
    return float(ttest_rel(a, b).pvalue)


def ort_ss(s):
    s = np.asarray(s, float)
    return f"{s.mean():.2f} ± {s.std(ddof=1) if len(s) > 1 else 0:.2f}"


class Deney:
    def __init__(self, args):
        self.args = args
        self.cihaz = "cuda" if torch.cuda.is_available() else "cpu"
        self.yapilan = pd.read_csv(KOSU_CSV, encoding="utf-8-sig") if os.path.exists(KOSU_CSV) else pd.DataFrame()
        self.tahmin = {}
        q = npz_yukle("qmnist.npz")
        d = npz_yukle("dokunmatik.npz")
        tr, val = dogrulama_ayir(len(q["x_train"]))
        if args.hizli:
            tr = tr[:8000]
        self.kaynak = {"QMNIST": (u8(q["x_train"][tr]), q["y_train"][tr], u8(q["x_train"][val]), q["y_train"][val])}
        self.x_mnist, self.y_mnist = u8(q["x_test10k"]), q["y_test10k"].astype(np.int64)

        # --- dokunmatik veriyi kisi bazinda ayir
        kisiler = np.unique(d["kisi"])
        self.n_kisi = len(kisiler)
        karisik = np.random.default_rng(42).permutation(kisiler)
        n_val = int(round(len(kisiler) * 0.2))
        self.val_kisi = set(karisik[:n_val].tolist())
        self.dok_val = np.isin(d["kisi"], list(self.val_kisi))
        self.x_dok, self.y_dok = u8(d["x"]), d["y"].astype(np.int64)
        pd.DataFrame({"kisi": kisiler, "kume": ["dogrulama" if k in self.val_kisi else "test" for k in kisiler]}) \
            .to_csv(os.path.join(CIK, "hafta4_dokunmatik_kisi_ayrimi.csv"), index=False, encoding="utf-8-sig")
        print(f"Dokunmatik veri: {len(kisiler)} kisi -> dogrulama {n_val} kisi ({self.dok_val.sum()} cizim), "
              f"test {len(kisiler) - n_val} kisi ({(~self.dok_val).sum()} cizim)")

        self.toplanan = None
        if os.path.exists(os.path.join(VERI, "toplanan.npz")):
            # 3. haftadaki betik katilimci numaralarini metin dizisi olarak kaydettigi icin allow_pickle gerekiyor
            # (dosyayi kendi betigimiz uretiyor)
            t = dict(np.load(os.path.join(VERI, "toplanan.npz"), allow_pickle=True))
            self.toplanan = (u8(t["x"]), t["y"].astype(np.int64), len(np.unique(t["katilimci"])))
            print(f"Ekibin topladigi veri: {self.toplanan[2]} katilimci, {len(t['y'])} cizim")

        if os.path.exists(os.path.join(VERI, "emnist_digits.npz")):
            e = npz_yukle("emnist_digits.npz")
            tr_e, val_e = dogrulama_ayir(len(e["x_train"]))
            if args.hizli:
                tr_e = tr_e[:8000]
            self.kaynak["EMNIST-Digits"] = (u8(e["x_train"][tr_e]), e["y_train"][tr_e],
                                            u8(e["x_train"][val_e]), e["y_train"][val_e])
            a, b = self.kaynak["QMNIST"], self.kaynak["EMNIST-Digits"]
            self.kaynak["QMNIST + EMNIST"] = (torch.cat([a[0], b[0]]), np.concatenate([a[1], b[1]]),
                                              torch.cat([a[2], b[2]]), np.concatenate([a[3], b[3]]))

    # ------------------------------------------------------------------ tek kosu
    def kos(self, bolum, ad, model_adi, veri_adi, ayar, tohum):
        anahtar = guvenli(f"{bolum}_{ad}_{tohum}")
        dosya = os.path.join(KLASOR, anahtar + ".npz")
        if len(self.yapilan) and ((self.yapilan.bolum == bolum) & (self.yapilan.ad == ad) &
                                  (self.yapilan.tohum == tohum)).any() and os.path.exists(dosya):
            self.tahmin[(bolum, ad, tohum)] = dict(np.load(dosya))
            return
        x_tr, y_tr, x_val, y_val = self.kaynak[veri_adi]
        model = baslangic(MODEL_SINIFLARI[model_adi], tohum)
        t0 = time.time()
        gecmis = egit(model, x_tr, torch.from_numpy(y_tr.astype(np.int64)), x_val,
                      torch.from_numpy(y_val.astype(np.int64)), self.cihaz, self.args.epoch, ad,
                      artirma=ayar or False, tohum=tohum, sessiz=True)
        sure = time.time() - t0
        p = {"dok": tahmin_et(model, self.x_dok, self.cihaz), "mnist": tahmin_et(model, self.x_mnist, self.cihaz)}
        if self.toplanan is not None:
            p["toplanan"] = tahmin_et(model, self.toplanan[0], self.cihaz)
        np.savez_compressed(dosya, **p)
        if tohum == TOHUMLAR[0]:
            torch.save(model.state_dict(), os.path.join(KLASOR, anahtar + ".pt"))
        self.tahmin[(bolum, ad, tohum)] = p
        en_iyi = max(gecmis, key=lambda g: g["dogrulama_dogrulugu"])
        dogru = p["dok"] == self.y_dok
        satir = {
            "bolum": bolum, "ad": ad, "model": model_adi, "egitim_verisi": veri_adi, "tohum": tohum,
            "egitim_ornegi": len(y_tr), "parametre": parametre_sayisi(model),
            "en_iyi_epoch": en_iyi["epoch"], "egitim_dogrulama_%": round(en_iyi["dogrulama_dogrulugu"] * 100, 2),
            # EMNIST, MNIST ile ayni NIST kaynagindan geldigi icin EMNIST iceren egitimlerde MNIST sonucu yazilmaz
            "mnist_test_%": round(float((p["mnist"] == self.y_mnist).mean()) * 100, 2) if veri_adi == "QMNIST" else np.nan,
            "dok_dogrulama_%": round(float(dogru[self.dok_val].mean()) * 100, 2),
            "dok_test_%": round(float(dogru[~self.dok_val].mean()) * 100, 2),
            "dok_tum_%": round(float(dogru.mean()) * 100, 2),
            "toplanan_%": (round(float((p["toplanan"] == self.toplanan[1]).mean()) * 100, 2)
                           if self.toplanan is not None else np.nan),
            "egitim_suresi_sn": round(sure, 1),
        }
        self.yapilan = pd.concat([self.yapilan, pd.DataFrame([satir])], ignore_index=True)
        self.yapilan.to_csv(KOSU_CSV, index=False, encoding="utf-8-sig")
        print(f"   {bolum} | {ad:24s} | tohum {tohum} | MNIST {satir['mnist_test_%']:>6} | dok. dogrulama "
              f"{satir['dok_dogrulama_%']:6.2f} | dok. test {satir['dok_test_%']:6.2f} | {sure:5.0f} sn", flush=True)

    def sonuclar(self, bolum):
        return self.yapilan[self.yapilan.bolum == bolum]

    def dizi(self, bolum, ad, sutun):
        s = self.sonuclar(bolum)
        return s[s.ad == ad].sort_values("tohum")[sutun].values

    # ------------------------------------------------------------------ A) veri artirma
    def bolum_a(self, tohumlar):
        print("\n[A] Veri artirma yontemleri (LeNet-5 benzeri CNN, QMNIST)")
        for ad, ayar in ARTIRMA_TEKLI:
            for t in tohumlar:
                self.kos("A", ad, "LeNet-5 benzeri CNN", "QMNIST", ayar, t)
        yok = self.dizi("A", "Yok", "dok_dogrulama_%").mean()
        faydali = [ad for ad, _ in ARTIRMA_TEKLI[1:5] if self.dizi("A", ad, "dok_dogrulama_%").mean() > yok]
        self.birlesik = None
        if 2 <= len(faydali) < 4:          # dordu de faydaliysa birlesim zaten "Hepsi" denemesidir
            ayar = {}
            for ad, a in ARTIRMA_TEKLI[1:5]:
                if ad in faydali:
                    ayar.update(a)
            self.birlesik = ("Faydalı olanlar: " + " + ".join(faydali), ayar)
            for t in tohumlar:
                self.kos("A", self.birlesik[0], "LeNet-5 benzeri CNN", "QMNIST", ayar, t)
        adlar = [a for a, _ in ARTIRMA_TEKLI] + ([self.birlesik[0]] if self.birlesik else [])
        ayarlar = dict(ARTIRMA_TEKLI + ([self.birlesik] if self.birlesik else []))
        # secim: dokunmatik dogrulamada en yuksek ortalama; "Yok"a gore anlamli iyilesme yoksa "Yok"
        ort = {a: self.dizi("A", a, "dok_dogrulama_%").mean() for a in adlar}
        en_iyi = max(ort, key=ort.get)
        p = eslestirilmis_t(self.dizi("A", en_iyi, "dok_dogrulama_%"), self.dizi("A", "Yok", "dok_dogrulama_%"))
        self.secilen_artirma = en_iyi if (en_iyi != "Yok" and p < 0.05) else "Yok"
        self.secilen_ayar = ayarlar[self.secilen_artirma]
        self.a_rapor = (adlar, ort, en_iyi, p)

    # ------------------------------------------------------------------ B) modeller
    def bolum_b(self, tohumlar):
        print(f"\n[B] Model karsilastirmasi (QMNIST, veri artirma: {self.secilen_artirma})")
        for model_adi in MODEL_SINIFLARI:
            for t in tohumlar:
                if model_adi == "LeNet-5 benzeri CNN" and ("A", self.secilen_artirma, t) in self.tahmin:
                    self.kopyala("A", self.secilen_artirma, "B", model_adi, t)
                else:
                    self.kos("B", model_adi, model_adi, "QMNIST", self.secilen_ayar, t)

    def kopyala(self, b1, a1, b2, a2, t):
        """A bolumunde ayni ayarla egitilmis LeNet kosusunu B'de tekrar egitmeden kullanir (sonuc aynidir)."""
        s = self.yapilan
        if ((s.bolum == b2) & (s.ad == a2) & (s.tohum == t)).any():
            self.tahmin[(b2, a2, t)] = self.tahmin.get((b1, a1, t)) or dict(
                np.load(os.path.join(KLASOR, guvenli(f"{b1}_{a1}_{t}") + ".npz")))
            return
        satir = s[(s.bolum == b1) & (s.ad == a1) & (s.tohum == t)].iloc[0].copy()
        satir["bolum"], satir["ad"] = b2, a2
        self.yapilan = pd.concat([self.yapilan, pd.DataFrame([satir])], ignore_index=True)
        self.yapilan.to_csv(KOSU_CSV, index=False, encoding="utf-8-sig")
        self.tahmin[(b2, a2, t)] = self.tahmin[(b1, a1, t)]
        np.savez_compressed(os.path.join(KLASOR, guvenli(f"{b2}_{a2}_{t}") + ".npz"), **self.tahmin[(b2, a2, t)])
        kaynak_pt = os.path.join(KLASOR, guvenli(f"{b1}_{a1}_{t}") + ".pt")
        if t == TOHUMLAR[0] and os.path.exists(kaynak_pt):
            import shutil
            shutil.copy(kaynak_pt, os.path.join(KLASOR, guvenli(f"{b2}_{a2}_{t}") + ".pt"))

    # ------------------------------------------------------------------ C) veri kaynagi
    def bolum_c(self, tohumlar):
        if "EMNIST-Digits" not in self.kaynak:
            print("\n[C] EMNIST bulunamadi, veri kaynagi deneyi atlaniyor.")
            return
        print(f"\n[C] Veri kaynagi (LeNet-5 benzeri CNN, veri artirma: {self.secilen_artirma})")
        for veri_adi in ("QMNIST", "EMNIST-Digits", "QMNIST + EMNIST"):
            for t in tohumlar:
                if veri_adi == "QMNIST" and ("A", self.secilen_artirma, t) in self.tahmin:
                    self.kopyala("A", self.secilen_artirma, "C", veri_adi, t)
                else:
                    self.kos("C", veri_adi, "LeNet-5 benzeri CNN", veri_adi, self.secilen_ayar, t)


# ---------------------------------------------------------------------------- rapor ve sekiller
def hiz_ve_boyut(model_adi, pt_yolu):
    m = MODEL_SINIFLARI[model_adi]()
    if os.path.exists(pt_yolu):
        m.load_state_dict(torch.load(pt_yolu, map_location="cpu"))
    m.eval()
    x = torch.rand(1, 1, 28, 28)
    with torch.no_grad():
        for _ in range(20):
            m(x)
        sureler = []
        for _ in range(200):
            t0 = time.perf_counter(); m(x); sureler.append((time.perf_counter() - t0) * 1000)
    boyut = os.path.getsize(pt_yolu) / 1024 if os.path.exists(pt_yolu) else float("nan")
    return parametre_sayisi(m), boyut, float(np.median(sureler))


def nokta_grafigi(ax, gruplar, renkler, cizgi=None, cizgi_etiket=None):
    """gruplar: [(etiket, degerler)]. Her tohum bir nokta, siyah cizgi ortalama."""
    for i, (et, v) in enumerate(gruplar):
        y = len(gruplar) - 1 - i
        ax.scatter(v, np.full(len(v), y) + np.linspace(-0.12, 0.12, len(v)), s=26, color=renkler[i], alpha=0.85,
                   zorder=3, edgecolors="white", linewidths=0.6)
        ax.plot([np.mean(v)] * 2, [y - 0.3, y + 0.3], color="#222222", lw=2, zorder=4)
        ax.text(np.mean(v), y + 0.32, "ort. " + f"{np.mean(v):.2f}".replace(".", ","), fontsize=7, ha="center",
                va="bottom", color="#333333")
    ax.set_yticks(range(len(gruplar))); ax.set_yticklabels([g[0] for g in gruplar][::-1], fontsize=8)
    ax.set_ylim(-0.75, len(gruplar) - 0.25)
    if cizgi is not None:
        ax.axvline(cizgi, color=GRI, lw=0.9, ls="--", zorder=1)
        if cizgi_etiket:
            ax.text(cizgi, -0.7, " " + cizgi_etiket, fontsize=6.5, color="#555555", ha="left", va="bottom")
    ax.grid(axis="x", color="#E8E8E8", lw=0.6, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)


def raporla(D):
    s = D.yapilan
    metin = [f"HAFTA 4 SONUCLARI\nTarih: {time.strftime('%Y-%m-%d %H:%M')}",
             f"Cihaz: {D.cihaz} ({platform.processor() or platform.machine()}), PyTorch {torch.__version__}, "
             f"epoch: {D.args.epoch}, tohumlar: {D.tohumlar}" + ("  [HIZLI DENEME - rapora konmaz]" if D.args.hizli else ""),
             f"Dokunmatik veri ayrimi: dogrulama {len(D.val_kisi)} kisi ({int(D.dok_val.sum())} cizim), test "
             f"{D.n_kisi - len(D.val_kisi)} kisi "
             f"({int((~D.dok_val).sum())} cizim)", ""]
    sut = ["mnist_test_%", "dok_dogrulama_%", "dok_test_%", "toplanan_%"]

    def ozet_tablo(bolum, adlar):
        satirlar = []
        for a in adlar:
            r = s[(s.bolum == bolum) & (s.ad == a)]
            if r.empty:
                continue
            satirlar.append({"ad": a, "n": len(r), **{c: (ort_ss(r[c]) if r[c].notna().all() else "-") for c in sut},
                             "sure_sn": round(r.egitim_suresi_sn.mean(), 1)})
        return pd.DataFrame(satirlar)

    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 20)
    if hasattr(D, "a_rapor"):
        adlar, ort, en_iyi, p = D.a_rapor
        t = ozet_tablo("A", adlar)
        t.to_csv(os.path.join(CIK, "hafta4_A_artirma.csv"), index=False, encoding="utf-8-sig")
        metin += ["[A] VERI ARTIRMA YONTEMLERI (ortalama ± standart sapma, %)", t.to_string(index=False), ""]
        yok = D.dizi("A", "Yok", "dok_dogrulama_%")
        for a in adlar[1:]:
            v = D.dizi("A", a, "dok_dogrulama_%")
            metin.append(f"  {a:40s} - Yok: fark {np.mean(v - yok):+.2f} puan, eslestirilmis t testi p = "
                         f"{eslestirilmis_t(v, yok):.3f}")
        metin += [f"  En yuksek dokunmatik dogrulama: {en_iyi} (Yok'a gore p = {p:.3f})",
                  f"  SECILEN ARTIRMA AYARI: {D.secilen_artirma}  "
                  "(kural: en iyisi Yok'tan anlamli olarak iyi degilse artirma kullanilmaz)", ""]
        fig, ax = plt.subplots(figsize=(6.4, 0.42 * len(adlar) + 0.9))
        gr = [("Birleşik (faydalı olanlar)" if a.startswith("Faydalı") else a, D.dizi("A", a, "dok_dogrulama_%"))
              for a in adlar]
        nokta_grafigi(ax, gr, [GRI] + [TURUNCU] * (len(gr) - 1), cizgi=np.mean(yok), cizgi_etiket="Yok ortalaması")
        ax.set_xlabel("Dokunmatik doğrulama kümesinde doğruluk (%)", fontsize=8)
        fig.tight_layout(); fig.savefig(os.path.join(CIK, "sekil_hafta4_artirma.png"), dpi=200); plt.close(fig)

    modeller = list(MODEL_SINIFLARI)
    if (s.bolum == "B").any():
        t = ozet_tablo("B", modeller)
        hiz = []
        for m in modeller:
            pt = os.path.join(KLASOR, guvenli(f"B_{m}_{TOHUMLAR[0]}") + ".pt")
            n, kb, ms = hiz_ve_boyut(m, pt)
            hiz.append({"ad": m, "parametre": n, "dosya_KB": round(kb, 1), "tahmin_ms (CPU, tek goruntu, medyan)": round(ms, 3)})
        t = t.merge(pd.DataFrame(hiz), on="ad")
        t.to_csv(os.path.join(CIK, "hafta4_B_modeller.csv"), index=False, encoding="utf-8-sig")
        metin += [f"[B] MODEL KARSILASTIRMASI (QMNIST, artirma: {D.secilen_artirma})", t.to_string(index=False), ""]
        metin.append("  Eslestirilmis t testi (5 tohum) ve McNemar testi (her tohumda ayri, dokunmatik test):")
        for a, b in combinations(modeller, 2):
            for c in ("mnist_test_%", "dok_test_%"):
                va, vb = D.dizi("B", a, c), D.dizi("B", b, c)
                metin.append(f"    {a} - {b} | {c}: fark {np.mean(va - vb):+.2f} puan, p = {eslestirilmis_t(va, vb):.4f}")
            anlamli, ayrinti = 0, []
            for tt in D.tohumlar:
                pa, pb = D.tahmin[("B", a, tt)]["dok"], D.tahmin[("B", b, tt)]["dok"]
                ya = (pa == D.y_dok)[~D.dok_val]; yb = (pb == D.y_dok)[~D.dok_val]
                b_, c_, p_ = mcnemar(ya, yb)
                anlamli += p_ < 0.05; ayrinti.append(f"{tt}: {b_}/{c_} p={p_:.1e}")
            metin.append(f"    {a} - {b} | McNemar: {anlamli}/{len(D.tohumlar)} tohumda p < 0,05  ({'; '.join(ayrinti)})")
        # secim: dokunmatik dogrulamada en yuksek ortalama; daha kucuk bir modelle farki anlamli degilse kucuk model
        ort = {m: D.dizi("B", m, "dok_dogrulama_%").mean() for m in modeller}
        en_iyi = max(ort, key=ort.get)
        secilen = en_iyi
        for m in sorted(modeller, key=lambda m: dict(zip(t.ad, t.parametre))[m]):
            if m == en_iyi:
                break
            if eslestirilmis_t(D.dizi("B", en_iyi, "dok_dogrulama_%"), D.dizi("B", m, "dok_dogrulama_%")) >= 0.05:
                secilen = m
                break
        D.secilen_model = secilen
        metin += [f"  Dokunmatik dogrulamada en yuksek ortalama: {en_iyi}",
                  f"  UYGULAMA ICIN SECILEN MODEL: {secilen}  (kural: daha kucuk bir modelle fark anlamli degilse "
                  "kucuk model secilir)", ""]
        src = os.path.join(KLASOR, guvenli(f"B_{secilen}_{TOHUMLAR[0]}") + ".pt")
        if os.path.exists(src) and not D.args.hizli:
            import shutil
            shutil.copy(src, os.path.join(MODELLER, "secilen_model_hafta4.pt"))
            open(os.path.join(MODELLER, "secilen_model_hafta4.txt"), "w", encoding="utf-8").write(
                f"{secilen}\nartirma: {D.secilen_artirma}\ntohum: {TOHUMLAR[0]}\n")

        renk = {"MLP": GRI, "LeNet-5 benzeri CNN": MAVI, "Derin CNN": TURUNCU}
        fig, axs = plt.subplots(1, 2, figsize=(6.8, 2.4))
        nokta_grafigi(axs[0], [(m, D.dizi("B", m, "mnist_test_%")) for m in modeller], [renk[m] for m in modeller],
                      cizgi=99.3, cizgi_etiket="H1 hedefi (%99,3)")
        axs[0].set_title("MNIST test", fontsize=8.5)
        nokta_grafigi(axs[1], [(m, D.dizi("B", m, "dok_test_%")) for m in modeller], [renk[m] for m in modeller])
        axs[1].set_title("Dokunmatik test kümesi", fontsize=8.5)
        axs[1].set_yticklabels([])
        for a in axs:
            a.set_xlabel("Doğruluk (%)", fontsize=8)
        fig.tight_layout(); fig.savefig(os.path.join(CIK, "sekil_hafta4_modeller.png"), dpi=200); plt.close(fig)

        # rakam bazinda hata (dokunmatik test, 5 tohum ortalamasi)
        rakam = []
        for m in modeller:
            for r in range(10):
                mask = (~D.dok_val) & (D.y_dok == r)
                h = [float((D.tahmin[("B", m, tt)]["dok"][mask] != r).mean()) * 100 for tt in D.tohumlar]
                rakam.append({"model": m, "rakam": r, "hata_%": round(np.mean(h), 2)})
        rk = pd.DataFrame(rakam)
        rk.to_csv(os.path.join(CIK, "hafta4_rakam_bazinda.csv"), index=False, encoding="utf-8-sig")
        fig, ax = plt.subplots(figsize=(6.4, 2.6))
        g = 0.8 / len(modeller)
        for k, m in enumerate(modeller):
            v = rk[rk.model == m].sort_values("rakam")["hata_%"].values
            ax.bar(np.arange(10) + (k - 1) * g, v, g * 0.92, color=renk[m], label=m, zorder=2)
        ax.set_xticks(range(10)); ax.set_xlabel("Gerçek rakam", fontsize=8); ax.set_ylabel("Hata oranı (%)", fontsize=8)
        ax.grid(axis="y", color="#E8E8E8", lw=0.6, zorder=0); ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=7, frameon=False)
        fig.tight_layout(); fig.savefig(os.path.join(CIK, "sekil_hafta4_rakam_bazinda.png"), dpi=200); plt.close(fig)

        # secilen modelin karmasiklik matrisi (dokunmatik test, ilk tohum)
        p = D.tahmin[("B", secilen, D.tohumlar[0])]["dok"][~D.dok_val]
        y = D.y_dok[~D.dok_val]
        cm = confusion_matrix(y, p, labels=range(10))
        oran = cm / cm.sum(axis=1, keepdims=True) * 100
        fig, ax = plt.subplots(figsize=(3.9, 3.5))
        ax.imshow(oran, cmap="Blues", vmin=0, vmax=100)
        for i in range(10):
            for j in range(10):
                if cm[i, j]:
                    ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=5.5,
                            color="white" if oran[i, j] > 60 else "#333333")
        ax.set_xticks(range(10)); ax.set_yticks(range(10)); ax.set_xlabel("Tahmin"); ax.set_ylabel("Gerçek")
        ax.set_title(f"{secilen}, dokunmatik test", fontsize=8)
        fig.tight_layout(); fig.savefig(os.path.join(CIK, "sekil_hafta4_karmasiklik.png"), dpi=200); plt.close(fig)
        np.fill_diagonal(cm, 0)
        sira = np.dstack(np.unravel_index(np.argsort(-cm.ravel()), cm.shape))[0][:6]
        metin += ["  Secilen modelin dokunmatik testte en cok karistirdiklari (tohum 42): " +
                  ", ".join(f"{g}->{t}: {cm[g, t]}" for g, t in sira), ""]

    if (s.bolum == "C").any():
        adlar = ["QMNIST", "EMNIST-Digits", "QMNIST + EMNIST"]
        t = ozet_tablo("C", adlar)
        t.to_csv(os.path.join(CIK, "hafta4_C_veri_kaynagi.csv"), index=False, encoding="utf-8-sig")
        metin += [f"[C] VERI KAYNAGI (LeNet-5 benzeri CNN, artirma: {D.secilen_artirma})", t.to_string(index=False)]
        for a in adlar[1:]:
            va, vb = D.dizi("C", a, "dok_test_%"), D.dizi("C", "QMNIST", "dok_test_%")
            metin.append(f"  {a} - QMNIST | dok. test: fark {np.mean(va - vb):+.2f} puan, p = {eslestirilmis_t(va, vb):.4f}")
        metin.append("")
        fig, ax = plt.subplots(figsize=(6.4, 1.9))
        nokta_grafigi(ax, [(a, D.dizi("C", a, "dok_test_%")) for a in adlar], [MAVI, YESIL, MOR])
        ax.set_xlabel("Dokunmatik testte doğruluk (%)", fontsize=8)
        fig.tight_layout(); fig.savefig(os.path.join(CIK, "sekil_hafta4_veri_kaynagi.png"), dpi=200); plt.close(fig)

    if D.toplanan is not None:
        metin.append(f"Ekibin topladigi veri: {D.toplanan[2]} katilimci, {len(D.toplanan[1])} cizim "
                     "(sonuclar tablolardaki toplanan_% sutununda)")
    metin = "\n".join(metin) + "\n"
    open(os.path.join(CIK, "ozet_hafta4.txt"), "w", encoding="utf-8").write(metin)
    print("\n" + metin)
    print(f"Tum kosular: {KOSU_CSV}\nSekiller ve tablolar '{CIK}' klasorunde."
          + ("" if D.args.hizli else f" Secilen model: {os.path.join(MODELLER, 'secilen_model_hafta4.pt')}"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epoch", type=int, default=10)
    ap.add_argument("--tohum-sayisi", type=int, default=5)
    ap.add_argument("--bolum", choices=["hepsi", "A", "B", "C"], default="hepsi")
    ap.add_argument("--hizli", action="store_true", help="kucuk veri, 2 epoch, 2 tohum: sadece deneme icin")
    args = ap.parse_args()
    global KOSU_CSV, KLASOR, CIK
    if args.hizli:                 # deneme ciktilari ayri klasorlere yazilir, asil sonuclara karismaz
        args.epoch, args.tohum_sayisi = 2, 2
        CIK, KLASOR = os.path.join(CIKTI, "hizli_deneme"), os.path.join(MODELLER, "hafta4_hizli")
        KOSU_CSV = os.path.join(CIK, "hafta4_kosular.csv")
        if os.path.exists(KOSU_CSV):
            os.remove(KOSU_CSV)
    os.makedirs(CIK, exist_ok=True); os.makedirs(KLASOR, exist_ok=True)
    D = Deney(args)
    D.tohumlar = TOHUMLAR[:args.tohum_sayisi]
    print(f"Cihaz: {D.cihaz}, epoch: {args.epoch}, tohumlar: {D.tohumlar}")
    t0 = time.time()
    if args.bolum in ("hepsi", "A"):
        D.bolum_a(D.tohumlar)
    else:
        D.secilen_artirma, D.secilen_ayar = "Yok", {}
        if (D.yapilan.bolum == "A").any():         # daha once A yapildiysa oradaki secimi kullan
            D.bolum_a(D.tohumlar)
    if args.bolum in ("hepsi", "B"):
        D.bolum_b(D.tohumlar)
    if args.bolum in ("hepsi", "C"):
        D.bolum_c(D.tohumlar)
    raporla(D)
    print(f"Toplam sure: {(time.time() - t0) / 60:.1f} dk")


if __name__ == "__main__":
    main()
