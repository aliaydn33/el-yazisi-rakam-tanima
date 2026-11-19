"""
HAFTA 7: Kullanici duzeltmeleriyle modelin ince ayari (E3)

E3 hedefi (1. hafta): kullanicilarin yaptigi duzeltmelerin kaydedilmesi ve modelin bu verilerle ince ayar yapilarak
guncellenmesi. Duzeltmeler uygulamadan iki yolla gelir: sunucudaki geri_bildirim/*.jsonl dosyalari ve sunucu
yokken kullanicinin "Duzeltmeleri indir" ile aldigi duzeltmeler_*.json dosyalari.

Ince ayar: uygulama modeli (modeller/uygulama_modeli.pt) duzeltmelerle kisa bir sure, dusuk ogrenme oraniyla
egitilir. Modelin eski bildiklerini unutmamasi icin her partide duzeltmelerin yanina QMNIST egitim kumesinden de
ornekler konur (partinin %75'i). Ayarlar deneyden once secildi: Adam, ogrenme orani 0,0001, 100 adim,
32 duzeltme + 96 QMNIST ornegi.

Calistirma:
  python 11_ince_ayar.py --kayitlar geri_bildirim duzeltmeler_20261119-2140.json
        Kayitlari okur, ince ayar yapar, dokunmatik dogrulama kumesinde eski modelle karsilastirir ve
        modeller/ince_ayar_modeli.pt dosyasina yazar. Uygulamadaki model kendiliginden degismez.
  python 11_ince_ayar.py --deney
        Yontemin ise yarayip yaramadigini olcer. Henuz gercek kullanici duzeltmesi olmadigi icin ekibin topladigi
        veri kullanilir: 20 katilimci iki gruba ayrilir, bir grubun cizimleri duzeltme gibi kullanilir, olcum
        diger gruptaki kisilerde yapilir (ve tersi). Iki durum denenir:
          (a) yalnizca modelin yanlis tahmin ettigi ya da reddettigi cizimler (kullanicinin gercekten duzeltecegi)
          (b) o kisilerin butun cizimleri
        5 tohum x 2 grup = 10 kosu. Karar kurali onceden: ince ayar, ayrilan kisilerde dogrulugu artiriyor ve
        dokunmatik dogrulama kumesinde 0,5 puandan fazla dusurmuyorsa ise yarar kabul edilir.
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import stats

from ortak import CIKTI, MODELLER, VERI, npz_yukle
from model_yapisi import MODEL_SINIFLARI, kayan_nokta, tahmin_et

TOHUMLAR = [42, 43, 44, 45, 46]
ADIM, PARTI_DUZELTME, PARTI_QMNIST, OGRENME_ORANI = 100, 32, 96, 1e-4


def kayitlari_oku(yollar):
    """geri_bildirim klasorleri, .jsonl ya da duzeltmeler_*.json dosyalari -> (x: N x 28 x 28 uint8, y: N)"""
    dosyalar = []
    for y in yollar:
        dosyalar += sorted(glob.glob(os.path.join(y, "*.jsonl")) + glob.glob(os.path.join(y, "*.json"))) if os.path.isdir(y) else [y]
    kayitlar = []
    for d in dosyalar:
        if d.endswith(".jsonl"):
            kayitlar += [json.loads(s) for s in open(d, encoding="utf-8") if s.strip()]
        else:
            o = json.load(open(d, encoding="utf-8"))
            kayitlar += o.get("kayitlar", []) if isinstance(o, dict) else o
    kayitlar = [k for k in kayitlar if len(k.get("goruntu", [])) == 784 and 0 <= int(k.get("dogru", -1)) <= 9]
    if not kayitlar:
        return np.zeros((0, 28, 28), np.uint8), np.zeros(0, np.int64)
    x = np.array([k["goruntu"] for k in kayitlar], np.uint8).reshape(-1, 28, 28)
    return x, np.array([int(k["dogru"]) for k in kayitlar], np.int64)


def model_yukle():
    sinif = open(os.path.join(MODELLER, "uygulama_modeli.txt"), encoding="utf-8").readline().strip()
    m = MODEL_SINIFLARI[sinif]()
    m.load_state_dict(torch.load(os.path.join(MODELLER, "uygulama_modeli.pt"), map_location="cpu"))
    return m.eval()


def ince_ayar(model, x_d, y_d, x_q, y_q, tohum):
    gen = torch.Generator().manual_seed(tohum)
    x_d, y_d = torch.from_numpy(x_d), torch.from_numpy(y_d)
    opt = torch.optim.Adam(model.parameters(), lr=OGRENME_ORANI)
    model.train()
    for _ in range(ADIM):
        i = torch.randint(len(x_d), (PARTI_DUZELTME,), generator=gen)
        j = torch.randint(len(x_q), (PARTI_QMNIST,), generator=gen)
        xb = torch.cat([kayan_nokta(x_d[i]), kayan_nokta(x_q[j])])
        yb = torch.cat([y_d[i], y_q[j]])
        kayip = F.cross_entropy(model(xb), yb)
        opt.zero_grad(); kayip.backward(); opt.step()
    return model.eval()


def dogruluk(model, x, y):
    return float((tahmin_et(model, torch.from_numpy(x), "cpu") == y).mean()) * 100


def veriler():
    q = npz_yukle("qmnist.npz")
    d = npz_yukle("dokunmatik.npz")
    ayrim = pd.read_csv(os.path.join(CIKTI, "hafta4_dokunmatik_kisi_ayrimi.csv"), encoding="utf-8-sig")
    val = np.isin(d["kisi"], ayrim[ayrim.kume == "dogrulama"].kisi)
    return (torch.from_numpy(q["x_train"]), torch.from_numpy(q["y_train"].astype(np.int64)),
            d["x"][val], d["y"][val], q["x_test10k"], q["y_test10k"])


def deney():
    x_q, y_q, x_v, y_v, x_m, y_m = veriler()
    t = np.load(os.path.join(VERI, "toplanan.npz"), allow_pickle=True)
    x_t, y_t, kat = t["x"], t["y"].astype(np.int64), t["katilimci"]
    kisiler = sorted(set(kat))
    gruplar = [kisiler[0::2], kisiler[1::2]]
    temel = model_yukle()
    esik = json.load(open(os.path.join("onnx", "bilgi.json"), encoding="utf-8"))["secilen.onnx"]["rakam_degil"]["esik"]
    with torch.no_grad():
        p = F.softmax(temel(kayan_nokta(torch.from_numpy(x_t))), 1).numpy()
    duzeltilecek = (p.argmax(1) != y_t) | (p.max(1) < esik)        # yanlis tahmin edilen ya da reddedilen
    once_v, once_m = dogruluk(temel, x_v, y_v), dogruluk(temel, x_m, y_m)
    satirlar = []
    for g, (egitim_grubu, olcum_grubu) in enumerate([(gruplar[0], gruplar[1]), (gruplar[1], gruplar[0])]):
        e = np.isin(kat, egitim_grubu); o = np.isin(kat, olcum_grubu)
        once_o = dogruluk(temel, x_t[o], y_t[o])
        for durum, secim in (("a) yalnızca düzeltilecek çizimler", e & duzeltilecek), ("b) bütün çizimler", e)):
            for tohum in TOHUMLAR:
                m = ince_ayar(model_yukle(), x_t[secim], y_t[secim], x_q, y_q, tohum)
                satirlar.append({"durum": durum, "grup": g + 1, "tohum": tohum, "duzeltme_sayisi": int(secim.sum()),
                                 "ayrilan_kisiler_once_%": round(once_o, 2), "ayrilan_kisiler_sonra_%": round(dogruluk(m, x_t[o], y_t[o]), 2),
                                 "dokunmatik_dogrulama_once_%": round(once_v, 2), "dokunmatik_dogrulama_sonra_%": round(dogruluk(m, x_v, y_v), 2),
                                 "mnist_test_once_%": round(once_m, 2), "mnist_test_sonra_%": round(dogruluk(m, x_m, y_m), 2)})
                print(satirlar[-1])
    f = pd.DataFrame(satirlar)
    f.to_csv(os.path.join(CIKTI, "hafta7_ince_ayar.csv"), index=False, encoding="utf-8-sig")
    ozet = {}
    for durum, gr in f.groupby("durum"):
        kazanc = gr["ayrilan_kisiler_sonra_%"] - gr["ayrilan_kisiler_once_%"]
        kayip_v = gr["dokunmatik_dogrulama_sonra_%"] - gr["dokunmatik_dogrulama_once_%"]
        kayip_m = gr["mnist_test_sonra_%"] - gr["mnist_test_once_%"]
        ozet[durum] = {"kosu": len(gr), "duzeltme_sayisi_grup_basina": sorted(set(gr.duzeltme_sayisi.tolist())),
                       "ayrilan_kisiler_degisim": [round(kazanc.mean(), 2), round(kazanc.std(), 2)],
                       "ayrilan_kisiler_p": round(float(stats.ttest_1samp(kazanc, 0).pvalue), 4),
                       "dokunmatik_dogrulama_degisim": [round(kayip_v.mean(), 2), round(kayip_v.std(), 2)],
                       "mnist_test_degisim": [round(kayip_m.mean(), 2), round(kayip_m.std(), 2)],
                       "ise_yarar": bool(kazanc.mean() > 0 and kayip_v.mean() >= -0.5)}
    with open(os.path.join(CIKTI, "hafta7_ince_ayar.json"), "w", encoding="utf-8") as d:
        json.dump(ozet, d, ensure_ascii=False, indent=2)
    print(json.dumps(ozet, ensure_ascii=False, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kayitlar", nargs="*", help="geri_bildirim klasoru ya da duzeltmeler_*.json dosyalari")
    ap.add_argument("--deney", action="store_true")
    ap.add_argument("--tohum", type=int, default=42)
    args = ap.parse_args()
    if args.deney:
        return deney()
    if not args.kayitlar:
        raise SystemExit("Kullanim: python 11_ince_ayar.py --kayitlar geri_bildirim [duzeltmeler_*.json]  ya da  --deney")
    x_d, y_d = kayitlari_oku(args.kayitlar)
    if len(x_d) == 0:
        raise SystemExit("Okunabilir duzeltme kaydi bulunamadi.")
    print(f"{len(x_d)} duzeltme kaydi okundu (rakamlara gore: {np.bincount(y_d, minlength=10).tolist()})")
    x_q, y_q, x_v, y_v, x_m, y_m = veriler()
    temel = model_yukle()
    m = ince_ayar(model_yukle(), x_d, y_d, x_q, y_q, args.tohum)
    print(f"Dokunmatik dogrulama: %{dogruluk(temel, x_v, y_v):.2f} -> %{dogruluk(m, x_v, y_v):.2f}")
    print(f"MNIST test          : %{dogruluk(temel, x_m, y_m):.2f} -> %{dogruluk(m, x_m, y_m):.2f}")
    print(f"Duzeltmelerin kendisi: %{dogruluk(temel, x_d, y_d):.2f} -> %{dogruluk(m, x_d, y_d):.2f} (egitimde kullanildi)")
    torch.save(m.state_dict(), os.path.join(MODELLER, "ince_ayar_modeli.pt"))
    print("modeller/ince_ayar_modeli.pt yazildi. Uygulamaya alinmasi, ince ayarda kullanilmayan yeni kisilerin "
          "cizimlerinde olculduktan sonra karar verilir.")


if __name__ == "__main__":
    main()
