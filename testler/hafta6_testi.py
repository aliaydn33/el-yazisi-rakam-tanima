"""
6. hafta testi: uygulamadaki Grad-CAM ve "rakam degil" kararinin Python ile karsilastirilmasi

1) Grad-CAM: 10_gradcam.py'nin kaydettigi orneklerde (testler/gradcam_ornekleri.npz) tarayicidaki JavaScript isi
   haritasi PyTorch'taki isi haritasiyla karsilastirilir (en buyuk fark ve korelasyon).
2) "Rakam degil": dokunmatik test, EMNIST harfleri ve Quick, Draw! cizimlerinden secilen orneklerde tarayicinin karari, Python'da ayni esikle
   verilen kararla karsilastirilir.

Calistirma (proje klasorunden, once 9_rakam_degil.py ve 10_gradcam.py): python testler/hafta6_testi.py
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ortak import CIKTI, VERI, npz_yukle   # noqa: E402

ORT_ADRES = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/"
SIRAYLA = """async ([g, m, a]) => { const s = []; for (const x of g) s.push(await Uygulama.skorlar(x, m, a)); return s; }"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sayfa", default="index.html")
    ap.add_argument("--ort-klasoru", default=None)
    args = ap.parse_args()
    from playwright.sync_api import sync_playwright
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from uygulama_testi import yerel_ort

    g = np.load(os.path.join("testler", "gradcam_ornekleri.npz"))
    s = np.load(os.path.join(CIKTI, "hafta6_ornek_rakam_degil.npz"))
    d = npz_yukle("dokunmatik.npz")
    rng = np.random.default_rng(0)
    xr = np.concatenate([d["x"][rng.choice(len(d["x"]), 300, replace=False)], s["x"][rng.choice(len(s["x"]), 300, replace=False)]])
    bilgi = json.load(open(os.path.join("onnx", "bilgi.json"), encoding="utf-8"))
    import onnxruntime as ort
    sonuc = {}
    with sync_playwright() as p:
        tarayici = p.chromium.launch(executable_path=os.environ.get("CHROMIUM") or None)
        baglam = tarayici.new_context()
        if args.ort_klasoru:
            baglam.route(ORT_ADRES + "**", yerel_ort(args.ort_klasoru))
        sayfa = baglam.new_page(); hatalar = []
        sayfa.on("pageerror", lambda h: hatalar.append(str(h)))
        sayfa.goto("file://" + os.path.abspath(args.sayfa))
        motor = sayfa.evaluate("Uygulama.hazir")
        # 1) Grad-CAM
        js = sayfa.evaluate("v => v.map(x => Uygulama.gradcam(x, 'secilen_int8.onnx'))", g["x"].reshape(len(g["x"]), -1).tolist())
        cam_js = np.array([j["cam"] for j in js], np.float32).reshape(-1, 28, 28)
        sinif_js = np.array([j["sinif"] for j in js])
        fark = np.abs(cam_js - g["cam"])
        kor = [float(np.corrcoef(a.ravel(), b.ravel())[0, 1]) for a, b in zip(cam_js, g["cam"]) if a.std() > 0 and b.std() > 0]
        sonuc["gradcam"] = {"ornek": int(len(g["x"])), "ayni_sinif_%": round(float((sinif_js == g["tahmin"]).mean()) * 100, 2),
                            "en_buyuk_fark": round(float(fark.max()), 6), "ortalama_fark": round(float(fark.mean()), 7),
                            "en_dusuk_korelasyon": round(min(kor), 6)}
        # 2) rakam degil karari
        for dosya in [k for k in ("secilen_int8.onnx", "derin_int8.onnx") if k in bilgi and "rakam_degil" in bilgi[k]]:
            rd = bilgi[dosya]["rakam_degil"]
            o = ort.InferenceSession(os.path.join("onnx", dosya), providers=["CPUExecutionProvider"])
            z = o.run(None, {"giris": xr.astype(np.float32)[:, None] / 255})[0]
            m = z.max(1)
            e = 1 / np.exp(z - m[:, None]).sum(1) if rd["yontem"] == "msp" else m + np.log(np.exp(z - m[:, None]).sum(1))
            py_ret = e < rd["esik"]
            skor = sayfa.evaluate(SIRAYLA, [xr.reshape(len(xr), -1).tolist(), motor, dosya])
            js_ret = np.array(sayfa.evaluate("([s, a]) => s.map(x => Uygulama.rakamDegil(x, a))", [skor, dosya]))
            sonuc[f"rakam_degil_{dosya}"] = {"ornek": int(len(xr)), "hesaplama": motor, "yontem": rd["yontem"],
                                              "ayni_karar_%": round(float((js_ret == py_ret).mean()) * 100, 2),
                                              "tarayicida_ret_%": round(float(js_ret.mean()) * 100, 1)}
        sonuc["sayfa_hatalari"] = hatalar
        tarayici.close()
    with open(os.path.join(CIKTI, "hafta6_tarayici_testi.json"), "w", encoding="utf-8") as f:
        json.dump(sonuc, f, ensure_ascii=False, indent=2)
    print(json.dumps(sonuc, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
