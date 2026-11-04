"""
HAFTA 5 - ADIM 1: Modelleri tarayicida calisacak ONNX bicimine donustur ve dogrulugun korundugunu kontrol et

Ne yapar?
  1) 4. haftada secilen modeli (modeller/secilen_model_hafta4.pt) ve varsa yedek aday Derin CNN'i ONNX'e cevirir
  2) Her model icin bir de 8 bit nicemlenmis (int8) surum uretir (dosya yaklasik 4 kat kuculur)
  3) PyTorch, ONNX ve int8 ONNX'i ayni veride karsilastirir:
       - MNIST test, dokunmatik test (4. haftadaki kisi bazli ayrim) ve varsa ekibin topladigi veri
       - dogruluk, PyTorch ile ayni tahmini verme orani, cikis degerlerindeki en buyuk fark
       - dosya boyutu ve islemcide tek goruntu icin tahmin suresi
  4) ONNX dosyalarini ve uygulamanin gosterecegi model bilgisini 'onnx' klasorune yazar
     (7_uygulama_olustur.py bu klasordeki modelleri uygulama sayfasina ekler)

Gereksinim: python -m pip install onnx onnxruntime
Calistirma:  python 6_onnx_donusum.py
"""
import json
import os
import time
import warnings

import numpy as np
import pandas as pd
import torch

from ortak import CIKTI, MODELLER, VERI, npz_yukle
from model_yapisi import MODEL_SINIFLARI

warnings.filterwarnings("ignore")
HEDEF = "onnx"
os.makedirs(HEDEF, exist_ok=True)

ADAYLAR = []   # (kisa ad, model sinifi adi, .pt yolu)
_bilgi = os.path.join(MODELLER, "secilen_model_hafta4.txt")
if os.path.exists(_bilgi):
    ADAYLAR.append(("secilen", open(_bilgi, encoding="utf-8").readline().strip(),
                    os.path.join(MODELLER, "secilen_model_hafta4.pt")))
_derin = os.path.join(MODELLER, "hafta4", "B_Derin_CNN_42.pt")
if os.path.exists(_derin):
    ADAYLAR.append(("derin", "Derin CNN", _derin))


def onnx_yaz(model, yol):
    """PyTorch modelini tek dosyalik ONNX olarak yazar (giris: (n, 1, 28, 28), 0-1 arasi; cikis: 10 sinifin skoru)."""
    x = torch.zeros(1, 1, 28, 28)
    try:
        torch.onnx.export(model, x, yol, input_names=["giris"], output_names=["cikis"], opset_version=17,
                          dynamic_axes={"giris": {0: "n"}, "cikis": {0: "n"}}, dynamo=False)
    except Exception:
        # eski disa aktarici yoksa yenisini kullan ve agirliklari ayni dosyada tut
        import onnx
        torch.onnx.export(model, x, yol, input_names=["giris"], output_names=["cikis"], opset_version=18, dynamo=True)
        m = onnx.load(yol, load_external_data=True)
        onnx.save(m, yol, save_as_external_data=False)
        if os.path.exists(yol + ".data"):
            os.remove(yol + ".data")


class Kalibrasyon:
    """int8 nicemleme icin olcek bulmakta kullanilan ornekler (egitim kumesinden 512 goruntu)."""
    def __init__(self, x):
        self.parcalar = iter([{"giris": x[i:i + 64]} for i in range(0, len(x), 64)])

    def get_next(self):
        return next(self.parcalar, None)


def int8_yaz(kaynak, hedef, kalibrasyon_x):
    """Statik nicemleme: agirliklar kanal bazinda int8, ara katman cikislari uint8.
    Son katman 32 bit birakilir. Boylece cikis skorlari, dolayisiyla ekranda gosterilen olasiliklar yuvarlanmaz."""
    import onnx
    from onnxruntime.quantization import QuantFormat, QuantType, quantize_static
    from onnxruntime.quantization.shape_inference import quant_pre_process
    ara = hedef + ".on.onnx"
    quant_pre_process(kaynak, ara)
    son_katman = [n.name for n in onnx.load(ara).graph.node if "cikis" in n.output]
    quantize_static(ara, hedef, Kalibrasyon(kalibrasyon_x), quant_format=QuantFormat.QDQ,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8, per_channel=True,
                    nodes_to_exclude=son_katman)
    os.remove(ara)


def ort_tahmin(oturum, x_f32, parti=1024):
    skor = np.concatenate([oturum.run(None, {"giris": x_f32[i:i + parti]})[0] for i in range(0, len(x_f32), parti)])
    return skor


def sure_olc(fonk, n=200):
    for _ in range(20):
        fonk()
    s = []
    for _ in range(n):
        t0 = time.perf_counter(); fonk(); s.append((time.perf_counter() - t0) * 1000)
    return float(np.median(s))


def main():
    import onnxruntime as ort
    if not ADAYLAR:
        raise SystemExit("modeller/secilen_model_hafta4.pt bulunamadi. Once 5_model_karsilastirma.py calistirilmali.")

    q = npz_yukle("qmnist.npz")
    d = npz_yukle("dokunmatik.npz")
    ayrim_yolu = os.path.join(CIKTI, "hafta4_dokunmatik_kisi_ayrimi.csv")
    if os.path.exists(ayrim_yolu):
        ayrim = pd.read_csv(ayrim_yolu, encoding="utf-8-sig")
        test_kisi = set(ayrim[ayrim.kume == "test"].kisi)
    else:                                   # 4. haftadaki ayrimin aynisi
        kisiler = np.unique(d["kisi"]); kar = np.random.default_rng(42).permutation(kisiler)
        test_kisi = set(kar[int(round(len(kisiler) * 0.2)):].tolist())
    dok_test = np.isin(d["kisi"], list(test_kisi))
    kumeler = {"MNIST test": (q["x_test10k"], q["y_test10k"]),
               "Dokunmatik test": (d["x"][dok_test], d["y"][dok_test])}
    if os.path.exists(os.path.join(VERI, "toplanan.npz")):
        t = dict(np.load(os.path.join(VERI, "toplanan.npz"), allow_pickle=True))
        kumeler[f"Toplanan veri ({len(np.unique(t['katilimci']))} kişi)"] = (t["x"], t["y"])
    kalibrasyon = (q["x_train"][np.random.default_rng(0).choice(len(q["x_train"]), 512, replace=False)]
                   .astype(np.float32)[:, None] / 255.0)

    satirlar, uygulama_bilgisi = [], {}
    for kisa, sinif, pt in ADAYLAR:
        model = MODEL_SINIFLARI[sinif]()
        model.load_state_dict(torch.load(pt, map_location="cpu")); model.eval()
        yol32 = os.path.join(HEDEF, f"{kisa}.onnx"); yol8 = os.path.join(HEDEF, f"{kisa}_int8.onnx")
        onnx_yaz(model, yol32)
        int8_yaz(yol32, yol8, kalibrasyon)
        oturumlar = {"ONNX": ort.InferenceSession(yol32, providers=["CPUExecutionProvider"]),
                     "ONNX int8": ort.InferenceSession(yol8, providers=["CPUExecutionProvider"])}
        boyut = {"PyTorch": os.path.getsize(pt) / 1024, "ONNX": os.path.getsize(yol32) / 1024,
                 "ONNX int8": os.path.getsize(yol8) / 1024}
        tek = np.random.default_rng(1).random((1, 1, 28, 28), dtype=np.float32)
        with torch.no_grad():
            hiz_pt = sure_olc(lambda: model(torch.from_numpy(tek)))
        hiz = {"PyTorch": hiz_pt,
               "ONNX": sure_olc(lambda: oturumlar["ONNX"].run(None, {"giris": tek})),
               "ONNX int8": sure_olc(lambda: oturumlar["ONNX int8"].run(None, {"giris": tek}))}
        for kume, (x, y) in kumeler.items():
            x_f = x.astype(np.float32)[:, None] / 255.0
            with torch.no_grad():
                skor_pt = torch.cat([model(torch.from_numpy(x_f[i:i + 1024])) for i in range(0, len(x_f), 1024)]).numpy()
            p_pt = skor_pt.argmax(1)
            for surum in ("PyTorch", "ONNX", "ONNX int8"):
                if surum == "PyTorch":
                    p, fark = p_pt, 0.0
                else:
                    skor = ort_tahmin(oturumlar[surum], x_f)
                    p, fark = skor.argmax(1), float(np.abs(skor - skor_pt).max())
                satirlar.append({"model": sinif, "surum": surum, "test_kumesi": kume, "ornek": len(y),
                                 "dogruluk_%": round(float((p == y).mean()) * 100, 2),
                                 "pytorch_ile_ayni_tahmin_%": round(float((p == p_pt).mean()) * 100, 2),
                                 "en_buyuk_skor_farki": round(fark, 6),
                                 "dosya_KB": round(boyut[surum], 1), "tahmin_ms_cpu": round(hiz[surum], 3)})
        # uygulamanin model listesinde gosterilecek bilgiler
        parametre = sum(p.numel() for p in model.parameters())
        for surum, dosya in (("ONNX", f"{kisa}.onnx"), ("ONNX int8", f"{kisa}_int8.onnx")):
            r = {s["test_kumesi"]: s for s in satirlar if s["model"] == sinif and s["surum"] == surum}
            uygulama_bilgisi[dosya] = {
                "model": sinif, "kisa_ad": kisa, "parametre": parametre,
                "sayi_bicimi": "8 bit (int8)" if "int8" in surum else "32 bit",
                "dosya_KB": round(boyut[surum], 1),
                "mnist_test_%": r["MNIST test"]["dogruluk_%"],
                "dokunmatik_test_%": r["Dokunmatik test"]["dogruluk_%"],
                "pytorch_ile_ayni_%": r["Dokunmatik test"]["pytorch_ile_ayni_tahmin_%"],
                "secilen": kisa == "secilen"}

    with open(os.path.join(HEDEF, "bilgi.json"), "w", encoding="utf-8") as f:
        json.dump(uygulama_bilgisi, f, ensure_ascii=False, indent=2)
    tablo = pd.DataFrame(satirlar)
    tablo.to_csv(os.path.join(CIKTI, "hafta5_onnx.csv"), index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 20)
    metin = (f"HAFTA 5 - ONNX DONUSUMU\nTarih: {time.strftime('%Y-%m-%d %H:%M')}\n"
             f"PyTorch {torch.__version__}, onnxruntime {ort.__version__}\n\n" + tablo.to_string(index=False) + "\n\n"
             "Uygulamaya eklenecek dosyalar:\n" +
             "".join(f"  {d}: {b['model']}, {b['sayi_bicimi']}, {b['dosya_KB']} KB\n" for d, b in uygulama_bilgisi.items()))
    open(os.path.join(CIKTI, "ozet_hafta5_onnx.txt"), "w", encoding="utf-8").write(metin)
    print(metin)
    print(f"ONNX dosyalari '{HEDEF}' klasorune, sonuclar '{CIKTI}' klasorune kaydedildi.")


if __name__ == "__main__":
    main()
