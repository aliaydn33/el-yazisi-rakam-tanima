"""
HAFTA 6 - ADIM 1: "Rakam degil" uyarisi (H5)

H5 hedefi: Rakam olmayan cizimlerin (harf, karalama vb.) en az %80'i "tanınamadı" olarak reddedilecek, bu sirada
gercek rakamlarin yanlislikla reddedilme orani %5'i gecmeyecek.

Karar kurali: Model, en olasi rakama verdigi olasilik (MSP) esigin altindaysa tahmin vermez. Esik, dokunmatik
DOGRULAMA kisilerinin (51 kisi) gercek rakamlarinin en fazla %5'i reddedilecek sekilde secilir.

Iki model karsilastirilir (5 tohum):
  A) Normal egitim: LeNet-5 benzeri CNN yalnizca rakamlarla egitilir (4. haftadaki ayarlar)
  B) Aykiri ornekle egitim (Hendrycks vd., 2019): her adimda rakamlarin yanina rakam olmayan cizimler de verilir ve
     modelden bu cizimlerde 10 rakama esit olasilik vermesi istenir. Boylece rakam olmayan cizimlerde guveni duser.

Rakam olmayan veriler:
  - EMNIST harfleri: egitim kismi B'nin egitiminde ve dogrulamada, test kismi yalnizca olcumde kullanilir.
    Rakama cok benzeyen harfler (O, I, L, Z, S, B, G, Q) egitimde kullanilmaz, olcumde ayrica raporlanir.
  - Quick, Draw!: 12 kategori egitimde ve dogrulamada, hic gorulmemis 8 kategori yalnizca olcumde kullanilir.
    Cizimler vurus olarak indirilir ve uygulamadaki on islemeden gecirilir.

Secim kurali (sonuclara bakmadan once belirlendi): B, dogrulamadaki rakam olmayan cizimlerde daha fazla ret sagliyorsa
ve dokunmatik dogrulama dogrulugunu 0,5 puandan fazla dusurmuyorsa uygulama modeli olur, aksi halde A kalir.

Calistirma:  python 9_rakam_degil.py
Sonra:       python 10_gradcam.py  ve  python 7_uygulama_olustur.py
Sonuclar cikti/ klasorune ve cikti/hafta6_sonuclar.zip dosyasina yazilir.
"""
import argparse
import importlib
import json
import os
import shutil
import sys
import time
import urllib.parse
import urllib.request
import zipfile

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ortak import CIKTI, HAM, MODELLER, VERI, mnist_formatina_getir, npz_yukle, vuruslari_ciz
from model_yapisi import LeNet5Benzeri, baslangic, dogrulama_ayir, kayan_nokta

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})
TOHUMLAR = [42, 43, 44, 45, 46]
YANLIS_RET = 0.05
RAKAMA_BENZEYEN = set("OILZSBGQ")
QD_EGITIM = ["house", "tree", "cat", "car", "sun", "flower", "fish", "cloud", "star", "triangle", "square", "zigzag"]
QD_TEST = ["squiggle", "lightning", "moon", "smiley face", "envelope", "key", "umbrella", "diamond"]
QD_ADET_EGITIM, QD_ADET_TEST = 1800, 500


# ----------------------------------------------------------------------------------------------- veriler
def quickdraw_indir(kategori, adet):
    """Quick, Draw! (Google, CC BY 4.0) sadelestirilmis vurus dosyasinin yalnizca basini okur."""
    url = ("https://storage.googleapis.com/quickdraw_dataset/full/simplified/" + urllib.parse.quote(kategori)
           + ".ndjson")
    x = []
    with urllib.request.urlopen(url, timeout=120) as yanit:
        for satir in yanit:
            o = json.loads(satir)
            if not o.get("recognized"):
                continue
            v = [list(zip(s[0], s[1])) for s in o["drawing"] if len(s[0])]
            if v:
                x.append(mnist_formatina_getir(vuruslari_ciz(v)))
            if len(x) >= adet:
                break
    return np.stack(x)


def rakam_olmayanlar(deneme=False):
    """EMNIST harfleri ve Quick, Draw! cizimleri, uygulamadaki 28x28 bicimde. Sonuc veri/rakam_degil.npz'ye saklanir."""
    yol = os.path.join(VERI, "rakam_degil.npz")
    if os.path.exists(yol) and not deneme:
        return dict(np.load(yol))
    if deneme:     # internet olmadan denemek icin: sentetik cizimler gercek verilerin yerine konur
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "testler"))
        from sentetik_cizim import sentetik
        x, g, ad = sentetik(120)
        harf = np.isin(g, ["Harf", "Rakama benzeyen harf"])
        sira = np.random.default_rng(0).permutation(len(x))
        x, g, ad, harf = x[sira], g[sira], ad[sira], harf[sira]
        h = np.flatnonzero(harf); d = np.flatnonzero(~harf)
        sekil_ad = np.unique(ad[d]); egitim_ad = set(sekil_ad[: len(sekil_ad) * 2 // 3])
        qd_e = d[np.isin(ad[d], list(egitim_ad))]; qd_t = d[~np.isin(ad[d], list(egitim_ad))]
        v = {"emnist_egitim_x": x[h[: len(h) // 2]], "emnist_egitim_harf": ad[h[: len(h) // 2]],
             "emnist_test_x": x[h[len(h) // 2:]], "emnist_test_harf": ad[h[len(h) // 2:]],
             "qd_egitim_x": x[qd_e], "qd_egitim_ad": ad[qd_e], "qd_test_x": x[qd_t], "qd_test_ad": ad[qd_t]}
        return v
    from torchvision.datasets import EMNIST
    v = {}
    print("EMNIST harfleri hazirlaniyor (EMNIST daha once indirildiyse tekrar indirilmez)...")
    for parca, egitim in (("egitim", True), ("test", False)):
        ds = EMNIST(root=HAM, split="letters", train=egitim, download=True)
        x = ds.data.transpose(1, 2).numpy().astype(np.uint8)
        harf = np.array([chr(ord("A") + int(t) - 1) for t in ds.targets.numpy()])
        if egitim:                                   # egitimde 30.000 harf yeterli
            sec = np.random.default_rng(0).choice(len(x), 30000, replace=False); x, harf = x[sec], harf[sec]
        t0 = time.time()
        v[f"emnist_{parca}_x"] = np.stack([mnist_formatina_getir(g) for g in x]); v[f"emnist_{parca}_harf"] = harf
        print(f"   {parca}: {len(x)} harf ({time.time() - t0:.0f} sn)")
    print("Quick, Draw! cizimleri indiriliyor...")
    for parca, kategoriler, adet in (("egitim", QD_EGITIM, QD_ADET_EGITIM), ("test", QD_TEST, QD_ADET_TEST)):
        xs, ads = [], []
        for k in kategoriler:
            x = quickdraw_indir(k, adet); xs.append(x); ads += [k] * len(x)
            print(f"   {k}: {len(x)} cizim")
        v[f"qd_{parca}_x"] = np.concatenate(xs); v[f"qd_{parca}_ad"] = np.array(ads)
    np.savez_compressed(yol, **v)
    return v


# ----------------------------------------------------------------------------------------------- egitim
def egit(tohum, x_tr, y_tr, x_val, y_val, aykiri=None, epoch=10, parti=128, agirlik=0.5):
    """4. haftadaki ayarlar (Adam 0,001, 128'lik parti, 10 epoch, dogrulamada en iyi epoch). aykiri verilirse her adimda
    ayni sayida rakam olmayan cizim eklenir ve bunlarda dagilimin duzgun (her rakama 1/10) olmasi istenir."""
    model = baslangic(LeNet5Benzeri, tohum)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    gen = torch.Generator().manual_seed(tohum)
    en_iyi, en_iyi_durum = -1, None
    for _ in range(epoch):
        model.train()
        sira = torch.randperm(len(x_tr), generator=gen)
        for i in range(0, len(sira), parti):
            idx = sira[i:i + parti]
            xb = kayan_nokta(x_tr[idx])
            if aykiri is not None:
                aidx = torch.randint(len(aykiri), (len(idx),), generator=gen)
                z = model(torch.cat([xb, kayan_nokta(aykiri[aidx])]))
                kayip = F.cross_entropy(z[:len(idx)], y_tr[idx]) - agirlik * F.log_softmax(z[len(idx):], 1).mean()
            else:
                kayip = F.cross_entropy(model(xb), y_tr[idx])
            opt.zero_grad(); kayip.backward(); opt.step()
        dogruluk = float((skorlar(model, x_val).argmax(1) == y_val.numpy()).mean())
        if dogruluk > en_iyi:
            en_iyi, en_iyi_durum = dogruluk, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(en_iyi_durum); model.eval()
    return model


def skorlar(model, x_u8, parti=4096):
    model.eval()
    with torch.no_grad():
        x = torch.as_tensor(x_u8)
        return torch.cat([model(kayan_nokta(x[i:i + parti])) for i in range(0, len(x), parti)]).numpy()


def msp(z):
    e = np.exp(z - z.max(1, keepdims=True)); return (e / e.sum(1, keepdims=True)).max(1)


# ----------------------------------------------------------------------------------------------- olcum
def olc(z_fn, rakam, degil):
    """z_fn(x) -> skorlar. Esik dokunmatik dogrulamada %5 yanlis rete gore secilir."""
    esik = float(np.quantile(msp(z_fn(rakam["Dokunmatik doğrulama"][0])), YANLIS_RET))
    s = {"esik": esik}
    for k, (x, y) in rakam.items():
        z = z_fn(x); p = z.argmax(1); kabul = msp(z) >= esik
        s[f"{k}|yanlis_ret_%"] = float((~kabul).mean()) * 100
        s[f"{k}|dogruluk_%"] = float((p == y).mean()) * 100
        s[f"{k}|kabul_edilenlerde_dogruluk_%"] = float((p[kabul] == y[kabul]).mean()) * 100
    for k, (x, _) in degil.items():
        s[f"{k}|ret_%"] = float((msp(z_fn(x)) < esik).mean()) * 100
    return s


def sekiller_ciz():
    """Sonuc tablolarindan (cikti/*.csv) iki sekil: yontemlerin 5 tohumda karsilastirilmasi ve harf/kategori bazinda ret."""
    t = pd.read_csv(os.path.join(CIKTI, "hafta6_rakam_degil_tohumlar.csv"), encoding="utf-8-sig")
    renk = {"Normal eğitim": "#9AA3AF", "Aykırı örnekle eğitim": "#2F4B7C"}
    paneller = [("Rakam olmayanlarda ret (%)", [("EMNIST harfleri (rakama benzeyenler hariç)|ret_%", "Harf"),
                                                 ("Quick, Draw! (görülmemiş 8 kategori)|ret_%", "Quick, Draw!\n(görülmemiş)"),
                                                 ("EMNIST harfleri (rakama benzeyenler)|ret_%", "Rakama\nbenzeyen harf")], (0, 100), 80),
                ("Rakamlarda yanlış ret (%)", [("Dokunmatik test|yanlis_ret_%", "Dokunmatik\ntest"),
                                                       ("Toplanan veri|yanlis_ret_%", "Toplanan\nveri")], (0, 14), 5),
                ("Rakamlarda doğruluk (%)", [("Dokunmatik test|dogruluk_%", "Dokunmatik\ntest"),
                                                     ("Toplanan veri|dogruluk_%", "Toplanan\nveri")], (88, 100), None)]
    fig, axs = plt.subplots(1, 3, figsize=(6.4, 2.5), gridspec_kw={"width_ratios": [3, 2, 2]})
    for a, (baslik, kollar, sinir, hedef) in zip(axs, paneller):
        for i, (kol, ad) in enumerate(kollar):
            for j, (y, r) in enumerate(renk.items()):
                d = t[t.yontem == y][kol].values; x = i + (j - 0.5) * 0.36
                a.scatter(np.full(len(d), x), d, s=10, color=r, alpha=0.8, zorder=3, label=y if i == 0 else None)
                a.plot([x - 0.13, x + 0.13], [d.mean()] * 2, color=r, lw=2)
        if hedef is not None:
            a.axhline(hedef, color="#C0392B", ls="--", lw=0.8)
        a.set_xticks(range(len(kollar))); a.set_xticklabels([ad for _, ad in kollar], fontsize=6.5)
        a.set_ylim(*sinir); a.set_title(baslik, fontsize=7.5)
        for k in ("top", "right"):
            a.spines[k].set_visible(False)
    axs[0].legend(fontsize=6.5, frameon=False, loc="lower left")
    fig.tight_layout(); fig.savefig(os.path.join(CIKTI, "sekil_hafta6_karsilastirma.png"), dpi=200); plt.close(fig)

    ay = pd.read_csv(os.path.join(CIKTI, "hafta6_rakam_degil_ayrinti.csv"), encoding="utf-8-sig")
    fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.2), gridspec_kw={"width_ratios": [26, 8]}, sharey=True)
    for a, grup in zip(axs, ("Harf", "Quick, Draw!")):
        d = ay[ay.grup == grup].sort_values("ad"); xs = np.arange(len(d))
        a.bar(xs, d["ret_uygulama_%"], 0.7, color=["#C9D3E3" if b else "#2F4B7C" for b in d["rakama_benzer"]])
        a.axhline(80, color="#C0392B", ls="--", lw=0.8)
        a.set_xticks(xs); a.set_xticklabels(d["ad"], fontsize=5.8, rotation=0 if grup == "Harf" else 60,
                                             ha="center" if grup == "Harf" else "right")
        a.set_ylim(0, 100); a.set_title(grup if grup != "Harf" else "EMNIST harfleri (açık renk: rakama benzeyen)", fontsize=7.5)
        for k in ("top", "right"):
            a.spines[k].set_visible(False)
    axs[0].set_ylabel("Ret (%)")
    fig.tight_layout(); fig.savefig(os.path.join(CIKTI, "sekil_hafta6_harf_kategori.png"), dpi=200); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deneme", action="store_true", help="internetsiz deneme: sentetik veri, 2 tohum, 2 epoch")
    ap.add_argument("--sadece-sekil", action="store_true", help="yalnizca cikti/ klasorundeki sonuclardan sekilleri ciz")
    args = ap.parse_args()
    if args.sadece_sekil:
        sekiller_ciz(); return
    tohumlar, epoch = (TOHUMLAR[:2], 2) if args.deneme else (TOHUMLAR, 10)
    t_bas = time.time()

    # --- rakamlar
    q = npz_yukle("qmnist.npz")
    tr, val = dogrulama_ayir(len(q["x_train"]))
    x_tr, y_tr = torch.from_numpy(q["x_train"][tr]), torch.from_numpy(q["y_train"][tr].astype(np.int64))
    x_val, y_val = torch.from_numpy(q["x_train"][val]), torch.from_numpy(q["y_train"][val].astype(np.int64))
    if args.deneme:
        x_tr, y_tr = x_tr[:6000], y_tr[:6000]
    d = npz_yukle("dokunmatik.npz")
    ayrim = pd.read_csv(os.path.join(CIKTI, "hafta4_dokunmatik_kisi_ayrimi.csv"), encoding="utf-8-sig")
    m_val = np.isin(d["kisi"], ayrim[ayrim.kume == "dogrulama"].kisi)
    m_tst = np.isin(d["kisi"], ayrim[ayrim.kume == "test"].kisi)
    rakam = {"Dokunmatik doğrulama": (d["x"][m_val], d["y"][m_val]), "Dokunmatik test": (d["x"][m_tst], d["y"][m_tst]),
             "MNIST test": (q["x_test10k"], q["y_test10k"])}
    if os.path.exists(os.path.join(VERI, "toplanan.npz")):
        t = np.load(os.path.join(VERI, "toplanan.npz"), allow_pickle=True)
        rakam["Toplanan veri"] = (t["x"], t["y"])

    # --- rakam olmayanlar
    v = rakam_olmayanlar(args.deneme)
    benzer_e = np.isin(v["emnist_egitim_harf"], list(RAKAMA_BENZEYEN))
    benzer_t = np.isin(v["emnist_test_harf"], list(RAKAMA_BENZEYEN))
    # egitim harflerinin %20'si ve egitim kategorilerindeki cizimlerin %20'si dogrulama icin ayrilir
    rng = np.random.default_rng(0)
    e_idx = rng.permutation(np.flatnonzero(~benzer_e)); e_val, e_tr = e_idx[: len(e_idx) // 5], e_idx[len(e_idx) // 5:]
    q_idx = rng.permutation(len(v["qd_egitim_x"])); q_val, q_tr = q_idx[: len(q_idx) // 5], q_idx[len(q_idx) // 5:]
    aykiri = torch.from_numpy(np.concatenate([v["emnist_egitim_x"][e_tr], v["qd_egitim_x"][q_tr]]))
    degil_val = {"Doğrulama: harf": v["emnist_egitim_x"][e_val], "Doğrulama: Quick, Draw!": v["qd_egitim_x"][q_val]}
    degil_test = {"EMNIST harfleri (rakama benzeyenler hariç)": (v["emnist_test_x"][~benzer_t], None),
                  "EMNIST harfleri (rakama benzeyenler)": (v["emnist_test_x"][benzer_t], None),
                  "Quick, Draw! (görülmemiş 8 kategori)": (v["qd_test_x"], None)}
    print(f"Egitimde kullanilan rakam olmayan cizim: {len(aykiri)}")

    # --- egitim ve olcum (5 tohum)
    satirlar, modeller = [], {}
    for yontem in ("Normal eğitim", "Aykırı örnekle eğitim"):
        for tohum in tohumlar:
            t0 = time.time()
            m = egit(tohum, x_tr, y_tr, x_val, y_val, aykiri if yontem.startswith("Aykırı") else None, epoch=epoch)
            z_fn = lambda x, m=m: skorlar(m, x)
            s = olc(z_fn, rakam, {**{k: (x, None) for k, x in degil_val.items()}, **degil_test})
            s.update({"yontem": yontem, "tohum": tohum, "egitim_sn": round(time.time() - t0, 1)})
            satirlar.append(s); modeller[(yontem, tohum)] = m
            print(f"{yontem}, tohum {tohum}: harf ret %{s['EMNIST harfleri (rakama benzeyenler hariç)|ret_%']:.1f}, "
                  f"Quick, Draw! ret %{s['Quick, Draw! (görülmemiş 8 kategori)|ret_%']:.1f}, "
                  f"dok. test yanlis ret %{s['Dokunmatik test|yanlis_ret_%']:.1f} ({s['egitim_sn']} sn)")
    tablo = pd.DataFrame(satirlar)
    tablo.to_csv(os.path.join(CIKTI, "hafta6_rakam_degil_tohumlar.csv"), index=False, encoding="utf-8-sig")

    # --- secim kurali (yalnizca dogrulama kumeleri)
    tablo["dogrulama_ret_%"] = tablo[["Doğrulama: harf|ret_%", "Doğrulama: Quick, Draw!|ret_%"]].mean(1)
    ort = tablo.groupby("yontem")[["dogrulama_ret_%", "Dokunmatik doğrulama|dogruluk_%"]].mean()
    fark = ort.loc["Aykırı örnekle eğitim", "Dokunmatik doğrulama|dogruluk_%"] - ort.loc["Normal eğitim", "Dokunmatik doğrulama|dogruluk_%"]
    secilen = ("Aykırı örnekle eğitim" if ort.loc["Aykırı örnekle eğitim", "dogrulama_ret_%"] > ort.loc["Normal eğitim", "dogrulama_ret_%"]
               and fark >= -0.5 else "Normal eğitim")
    from scipy import stats
    karsilastirma = []
    for kol in [c for c in tablo.columns if c.endswith("_%") and "|" in c]:
        a = tablo[tablo.yontem == "Normal eğitim"][kol].values; b = tablo[tablo.yontem == "Aykırı örnekle eğitim"][kol].values
        p = float(stats.ttest_rel(b, a).pvalue) if len(a) > 1 and np.std(b - a) > 0 else float("nan")
        karsilastirma.append({"olcut": kol, "normal_ort": a.mean(), "normal_ss": a.std(ddof=1) if len(a) > 1 else 0,
                              "aykiri_ort": b.mean(), "aykiri_ss": b.std(ddof=1) if len(b) > 1 else 0, "p": p})
    kar = pd.DataFrame(karsilastirma).round(3)
    kar.to_csv(os.path.join(CIKTI, "hafta6_rakam_degil.csv"), index=False, encoding="utf-8-sig")

    # --- uygulama modeli: secilen yontemin 42 numarali tohumu -> ONNX (32 bit ve 8 bit)
    model = modeller[(secilen, tohumlar[0])]
    torch.save(model.state_dict(), os.path.join(MODELLER, "uygulama_modeli.pt"))
    with open(os.path.join(MODELLER, "uygulama_modeli.txt"), "w", encoding="utf-8") as f:
        f.write(f"LeNet-5 benzeri CNN\negitim: {secilen}\ntohum: {tohumlar[0]}\n")
    onnx_mod = importlib.import_module("6_onnx_donusum")
    import onnxruntime as ort_
    os.makedirs("onnx", exist_ok=True)
    yol32, yol8 = os.path.join("onnx", "secilen.onnx"), os.path.join("onnx", "secilen_int8.onnx")
    onnx_mod.onnx_yaz(model, yol32)
    kal = (q["x_train"][np.random.default_rng(0).choice(len(q["x_train"]), 512, replace=False)].astype(np.float32)[:, None] / 255.0)
    onnx_mod.int8_yaz(yol32, yol8, kal)
    bilgi_yolu = os.path.join("onnx", "bilgi.json")
    bilgi = json.load(open(bilgi_yolu, encoding="utf-8")) if os.path.exists(bilgi_yolu) else {}
    uygulama = {}
    for dosya, yol in (("secilen.onnx", yol32), ("secilen_int8.onnx", yol8)):
        o = ort_.InferenceSession(yol, providers=["CPUExecutionProvider"])
        z_fn = lambda x, o=o: np.concatenate([o.run(None, {"giris": x[i:i + 4096].astype(np.float32)[:, None] / 255})[0]
                                              for i in range(0, len(x), 4096)])
        s = olc(z_fn, rakam, degil_test)
        uygulama[dosya] = s
        eski = bilgi.get(dosya, {})
        bilgi[dosya] = {**eski, "model": "LeNet-5 benzeri CNN", "kisa_ad": "secilen", "parametre": 61706,
                        "sayi_bicimi": "8 bit (int8)" if "int8" in dosya else "32 bit",
                        "dosya_KB": round(os.path.getsize(yol) / 1024, 1),
                        "mnist_test_%": round(s["MNIST test|dogruluk_%"], 2),
                        "dokunmatik_test_%": round(s["Dokunmatik test|dogruluk_%"], 2),
                        "egitim": secilen, "secilen": True,
                        "rakam_degil": {"yontem": "msp", "esik": round(s["esik"], 4), "dogrulama_yanlis_ret": YANLIS_RET}}
    # derin CNN (yedek aday): yeniden egitilmez, yalnizca esik belirlenir
    if os.path.exists(os.path.join("onnx", "derin_int8.onnx")):
        for dosya in ("derin.onnx", "derin_int8.onnx"):
            o = ort_.InferenceSession(os.path.join("onnx", dosya), providers=["CPUExecutionProvider"])
            z_fn = lambda x, o=o: np.concatenate([o.run(None, {"giris": x[i:i + 4096].astype(np.float32)[:, None] / 255})[0]
                                                  for i in range(0, len(x), 4096)])
            s = olc(z_fn, rakam, degil_test); uygulama[dosya] = s
            if dosya in bilgi:
                bilgi[dosya]["rakam_degil"] = {"yontem": "msp", "esik": round(s["esik"], 4), "dogrulama_yanlis_ret": YANLIS_RET}
    with open(bilgi_yolu, "w", encoding="utf-8") as f:
        json.dump(bilgi, f, ensure_ascii=False, indent=2)
    with open(os.path.join(CIKTI, "hafta6_uygulama_modeli.json"), "w", encoding="utf-8") as f:
        json.dump({"secilen_yontem": secilen, "dogruluk_farki_puan": round(float(fark), 3),
                   "uygulama": {k: {a: round(b, 4) for a, b in s.items()} for k, s in uygulama.items()}}, f,
                  ensure_ascii=False, indent=2)

    # --- harf ve kategori bazinda ret orani (uygulama modeli, 8 bit) ve normal egitimle karsilastirma
    esik8 = uygulama["secilen_int8.onnx"]["esik"]
    o8 = ort_.InferenceSession(yol8, providers=["CPUExecutionProvider"])
    z8 = lambda x: o8.run(None, {"giris": x.astype(np.float32)[:, None] / 255})[0]
    normal = modeller[("Normal eğitim", tohumlar[0])]
    esik_n = float(np.quantile(msp(skorlar(normal, rakam["Dokunmatik doğrulama"][0])), YANLIS_RET))
    ayrinti = []
    for grup, x, ad in (("Harf", v["emnist_test_x"], v["emnist_test_harf"]), ("Quick, Draw!", v["qd_test_x"], v["qd_test_ad"])):
        r_u, r_n = msp(z8(x)) < esik8, msp(skorlar(normal, x)) < esik_n
        for a in np.unique(ad):
            m = ad == a
            ayrinti.append({"grup": grup, "ad": a, "cizim": int(m.sum()), "rakama_benzer": a in RAKAMA_BENZEYEN,
                            "ret_normal_%": round(float(r_n[m].mean()) * 100, 1), "ret_uygulama_%": round(float(r_u[m].mean()) * 100, 1)})
    ay = pd.DataFrame(ayrinti); ay.to_csv(os.path.join(CIKTI, "hafta6_rakam_degil_ayrinti.csv"), index=False, encoding="utf-8-sig")

    # --- sekiller
    sekiller_ciz()

    fig, axs = plt.subplots(2, 10, figsize=(6.4, 1.7))
    ornek = np.concatenate([np.random.default_rng(3).choice(np.flatnonzero(~benzer_t), 10, replace=False)])
    q_orn = [int(np.flatnonzero(v["qd_test_ad"] == k)[0]) for k in np.unique(v["qd_test_ad"])][:10]
    for j in range(10):
        for i, (x, ad, idx) in enumerate(((v["emnist_test_x"], v["emnist_test_harf"], ornek),
                                          (v["qd_test_x"], v["qd_test_ad"], q_orn))):
            a = axs[i, j]; a.set_xticks([]); a.set_yticks([])
            if j < len(idx):
                g = x[idx[j]]; ret = msp(z8(g[None]))[0] < esik8
                a.imshow(g, cmap="gray_r"); a.set_title(("✗ " if ret else "✓ ") + str(ad[idx[j]])[:9], fontsize=5.5, pad=1.5,
                                                         color="#2F4B7C" if ret else "#C0392B")
    fig.tight_layout(h_pad=0.2, w_pad=0.2); fig.savefig(os.path.join(CIKTI, "sekil_hafta6_rakam_degil_ornekler.png"), dpi=200)
    plt.close(fig)

    # --- ozet
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 20); pd.set_option("display.max_rows", 200)
    u = uygulama["secilen_int8.onnx"]
    hedef_ret = np.mean([u["EMNIST harfleri (rakama benzeyenler hariç)|ret_%"], u["Quick, Draw! (görülmemiş 8 kategori)|ret_%"]])
    metin = (f"HAFTA 6 - RAKAM DEGIL UYARISI (H5)\nTarih: {time.strftime('%Y-%m-%d %H:%M')}  Sure: {(time.time() - t_bas) / 60:.1f} dk"
             f"{'  (DENEME CALISTIRMASI - sonuclar kullanilmaz)' if args.deneme else ''}\n\n"
             f"Egitimde kullanilan rakam olmayan cizim: {len(aykiri)} (EMNIST harfi {len(e_tr)}, Quick, Draw! {len(q_tr)})\n"
             f"Secilen yontem: {secilen} (dokunmatik dogrulama dogrulugu farki: {fark:+.2f} puan)\n\n"
             "5 TOHUMUN ORTALAMASI (normal / aykiri ornekle egitim, p: eslestirilmis t testi)\n" + kar.to_string(index=False) +
             "\n\nUYGULAMA MODELI (8 bit ONNX)\n" + "\n".join(f"  {k}: {b:.2f}" for k, b in u.items()) +
             f"\n\nH5: rakam olmayanlarda ret (harf + Quick, Draw! ortalamasi) %{hedef_ret:.1f} (hedef >= %80), "
             f"dokunmatik testte yanlis ret %{u['Dokunmatik test|yanlis_ret_%']:.1f} (hedef <= %5)\n\n"
             "HARF VE KATEGORI BAZINDA RET (%)\n" + ay.to_string(index=False) + "\n")
    open(os.path.join(CIKTI, "ozet_hafta6_rakam_degil.txt"), "w", encoding="utf-8").write(metin)
    print(metin)

    # --- tarayici testi icin kucuk bir ornek (her kaynaktan 300 cizim)
    rng2 = np.random.default_rng(5)
    np.savez_compressed(os.path.join(CIKTI, "hafta6_ornek_rakam_degil.npz"),
                        x=np.concatenate([v["emnist_test_x"][rng2.choice(len(v["emnist_test_x"]), 300, replace=False)],
                                          v["qd_test_x"][rng2.choice(len(v["qd_test_x"]), 300, replace=False)]]))

    # --- gonderilecek dosyalar tek zip'te
    with zipfile.ZipFile(os.path.join(CIKTI, "hafta6_sonuclar.zip"), "w", zipfile.ZIP_DEFLATED) as z:
        for f in ("hafta6_rakam_degil.csv", "hafta6_rakam_degil_tohumlar.csv", "hafta6_rakam_degil_ayrinti.csv",
                  "hafta6_uygulama_modeli.json", "ozet_hafta6_rakam_degil.txt", "sekil_hafta6_karsilastirma.png",
                  "sekil_hafta6_harf_kategori.png", "sekil_hafta6_rakam_degil_ornekler.png", "hafta6_ornek_rakam_degil.npz"):
            z.write(os.path.join(CIKTI, f), f)
        z.write(os.path.join(MODELLER, "uygulama_modeli.pt"), "uygulama_modeli.pt")
        z.write(os.path.join(MODELLER, "uygulama_modeli.txt"), "uygulama_modeli.txt")
        z.write(bilgi_yolu, "bilgi.json")
        z.write(yol32, "secilen.onnx"); z.write(yol8, "secilen_int8.onnx")
    print(f"Bitti. Sonuclar: {os.path.join(CIKTI, 'hafta6_sonuclar.zip')}")


if __name__ == "__main__":
    main()
