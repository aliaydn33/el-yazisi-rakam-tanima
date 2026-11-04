"""
Uygulama sayfasinin (index.html) testi

1) Dogruluk: Dokunmatik test kumesindeki kisilerden secilen cizimler tarayicida iki yoldan tahmin edilir:
     - onnxruntime-web (tarayicidaki ONNX calistiricisi)
     - yedek hesaplayici (sayfadaki JavaScript kodu)
   Her ikisi Python'daki onnxruntime ile ayni ONNX dosyasi uzerinde karsilastirilir (ayni tahmin orani,
   skorlardaki en buyuk fark). Ayrica cizimin vuruslari tarayicida on islemeden gecirilerek uctan uca
   dogruluk olculur.
2) Hiz: Sayfadaki "Hiz testi" dugmesine basilir. Istenirse islemci yavaslatilarak (--yavaslatma 4) orta
   seviye bir telefonun hizina yaklasilir. Bu yalnizca bir benzetimdir, gercek olcum telefonda yapilir.
3) Toplanan veri: 'toplanan_veri' klasorundeki cizimler (veri toplama sayfasindan) varsa vuruslari tarayicida
   on islemeden gecirilip her modelle tahmin edilir. Bu, H2 hedefinin uygulamanin kendisinde olculmesidir.
4) Internetsiz calisma: onnxruntime-web'in indirildigi adres engellenerek sayfanin yedek hesaplayiciyla
   calismaya devam ettigi kontrol edilir.

Gereksinim: pip install playwright onnxruntime, ardindan: playwright install chromium
Calistirma (proje klasorunden): python testler/uygulama_testi.py
"""
import argparse
import json
import mimetypes
import os
import re
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ortak import CIKTI, HAM, mnist_formatina_getir, vuruslari_ciz   # noqa: E402

ORT_ADRES = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/"
# onnxruntime-web ayni modelde ayni anda tek tahmin calistirir, bu yuzden cizimler sirayla verilir
SIRAYLA = """async ([g, m, a]) => { const s = []; for (const x of g) s.push(await Uygulama.skorlar(x, m, a)); return s; }"""


def test_cizimleri(n, tohum=0):
    """Dokunmatik veri setinden, 4. haftadaki ayrimda test kumesine dusen kisilerin cizimleri."""
    k = os.path.join(HAM, "dokunmatik")
    glyph = pd.read_csv(os.path.join(k, "glyph.csv"))
    stroke = pd.read_csv(os.path.join(k, "stroke.csv"))
    touch = pd.read_csv(os.path.join(k, "touch.csv"))
    ayrim = pd.read_csv(os.path.join(CIKTI, "hafta4_dokunmatik_kisi_ayrimi.csv"), encoding="utf-8-sig")
    test_kisi = set(ayrim[ayrim.kume == "test"].kisi)
    glyph = glyph[(glyph.ZVALID == 1) & glyph.ZSUBJECT.isin(test_kisi)]
    touch = touch.sort_values(["ZSTROKE", "Z_FOK_STROKE"])
    nokta = {s: list(zip(g.ZX.values.tolist(), g.ZY.values.tolist())) for s, g in touch.groupby("ZSTROKE")}
    stroke = stroke.sort_values(["ZGLYPH", "Z_FOK_GLYPH"])
    vurus = {g_: [nokta[s] for s in g.Z_PK.values if s in nokta] for g_, g in stroke.groupby("ZGLYPH")}
    secim = glyph.sample(n=n, random_state=tohum)
    return [(vurus[r.Z_PK], int(r.ZCHARACTER)) for r in secim.itertuples() if vurus.get(r.Z_PK)]


def toplanan_cizimler(klasor="toplanan_veri"):
    """Veri toplama sayfasinin JSON dosyalari (4_toplanan_veri.py ile ayni okuma: ayni cizim iki kez sayilmaz)."""
    import glob
    cizimler, gorulen = [], set()
    for yol in sorted(glob.glob(os.path.join(klasor, "*.json"))):
        o = json.load(open(yol, encoding="utf-8"))
        for orn in o.get("ornekler", []):
            anahtar = (o["katilimci_no"], orn["sira"])
            vurus = [[[p[0], p[1]] for p in v] for v in orn["vuruslar"] if len(v)]
            if anahtar in gorulen or not vurus:
                continue
            gorulen.add(anahtar)
            cizimler.append((vurus, int(orn["rakam"]), o["katilimci_no"]))
    return cizimler


def yerel_ort(klasor):
    """--ort-klasoru verilirse onnxruntime-web dosyalari internetten degil bu klasorden verilir."""
    def yonlendir(route):
        ad = route.request.url.split("/dist/")[-1].split("?")[0]
        yol = os.path.join(klasor, ad)
        if not os.path.exists(yol):
            return route.fulfill(status=404, body="")
        tur = "application/wasm" if ad.endswith(".wasm") else (mimetypes.guess_type(ad)[0] or "text/javascript")
        route.fulfill(status=200, body=open(yol, "rb").read(), headers={
            "Content-Type": tur, "Access-Control-Allow-Origin": "*"})
    return yonlendir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500, help="denenecek cizim sayisi")
    ap.add_argument("--sayfa", default="index.html")
    ap.add_argument("--yavaslatma", type=float, default=4, help="hiz benzetimi icin islemci yavaslatma kati (0 = yok)")
    ap.add_argument("--ort-klasoru", default=None, help="onnxruntime-web dosyalari icin yerel klasor (istege bagli)")
    args = ap.parse_args()
    import onnxruntime as ort
    from playwright.sync_api import sync_playwright

    ornekler = test_cizimleri(args.n)
    vuruslar = [[[list(q) for q in s] for s in v] for v, _ in ornekler]
    y = np.array([e for _, e in ornekler])
    py28 = np.stack([mnist_formatina_getir(vuruslari_ciz(v)) for v, _ in ornekler])
    adres = "file://" + os.path.abspath(args.sayfa)

    satirlar, hiz = [], []
    with sync_playwright() as p:
        tarayici = p.chromium.launch(executable_path=os.environ.get("CHROMIUM") or None)
        baglam = tarayici.new_context(viewport={"width": 1100, "height": 900})
        if args.ort_klasoru:
            baglam.route(ORT_ADRES + "**", yerel_ort(args.ort_klasoru))
        sayfa = baglam.new_page()
        hatalar = []
        sayfa.on("pageerror", lambda h: hatalar.append(str(h)))
        sayfa.goto(adres)
        motor = sayfa.evaluate("Uygulama.hazir")
        modeller = sayfa.evaluate("Uygulama.modeller")
        print(f"Sayfa acildi. onnxruntime-web durumu: {sayfa.evaluate('Uygulama.ortDurum()')}, modeller: {modeller}")
        if motor != "ort":
            print("UYARI: onnxruntime-web yuklenemedi, yalnizca yedek hesaplayici denenecek.")

        js28 = np.array(sayfa.evaluate("v => v.map(c => Uygulama.cizimden28(c))", vuruslar), np.uint8).reshape(-1, 28, 28)
        for model in modeller:
            oturum = ort.InferenceSession(os.path.join("onnx", model), providers=["CPUExecutionProvider"])
            ref = oturum.run(None, {oturum.get_inputs()[0].name: py28.astype(np.float32)[:, None] / 255})[0]
            for m in (["ort", "js"] if motor == "ort" else ["js"]):
                t0 = time.time()
                skor = np.array(sayfa.evaluate(SIRAYLA, [py28.reshape(len(y), -1).tolist(), m, model]), np.float32)
                uctan = np.array(sayfa.evaluate(SIRAYLA, [js28.reshape(len(y), -1).tolist(), m, model]), np.float32)
                satirlar.append({
                    "model_dosyasi": model, "hesaplama": "onnxruntime-web" if m == "ort" else "yedek hesaplayici",
                    "cizim": len(y),
                    "python_onnx_ile_ayni_tahmin_%": round(float((skor.argmax(1) == ref.argmax(1)).mean()) * 100, 2),
                    "en_buyuk_skor_farki": round(float(np.abs(skor - ref).max()), 6),
                    "dogruluk_python_on_isleme_%": round(float((skor.argmax(1) == y).mean()) * 100, 2),
                    "dogruluk_tarayici_on_isleme_%": round(float((uctan.argmax(1) == y).mean()) * 100, 2),
                    "sure_s": round(time.time() - t0, 1)})
                print(satirlar[-1], flush=True)

        # toplanan veri: vuruslar tarayicida on islenir ve tahmin edilir (uctan uca)
        toplanan, toplanan_satir = toplanan_cizimler(), []
        if toplanan:
            tv = [v for v, _, _ in toplanan]; ty = np.array([e for _, e, _ in toplanan])
            tk = np.array([k for _, _, k in toplanan])
            t28 = sayfa.evaluate("v => v.map(c => Uygulama.cizimden28(c))", tv)
            for model in modeller:
                for m in (["ort", "js"] if motor == "ort" and model == modeller[0] else [motor]):
                    skor = np.array(sayfa.evaluate(SIRAYLA, [t28, m, model]), np.float32)
                    dogru = skor.argmax(1) == ty
                    kisi = pd.Series(dogru).groupby(tk).mean() * 100
                    toplanan_satir.append({
                        "model_dosyasi": model, "hesaplama": "onnxruntime-web" if m == "ort" else "yedek hesaplayici",
                        "katilimci": len(kisi), "cizim": len(ty), "dogru": int(dogru.sum()),
                        "dogruluk_%": round(float(dogru.mean()) * 100, 2),
                        "en_dusuk_katilimci_%": round(float(kisi.min()), 1),
                        "medyan_katilimci_%": round(float(kisi.median()), 1)})
                    print(toplanan_satir[-1], flush=True)
            pd.DataFrame(toplanan_satir).to_csv(os.path.join(CIKTI, "hafta5_tarayici_toplanan.csv"), index=False,
                                                encoding="utf-8-sig")

        # hiz testi: sayfadaki dugmeyle, her model ve hesaplama yolu icin
        cdp = baglam.new_cdp_session(sayfa)
        for yavas in sorted({1, args.yavaslatma} - {0}):
            cdp.send("Emulation.setCPUThrottlingRate", {"rate": yavas})
            for model in modeller:
                for m in (["ort", "js"] if motor == "ort" else ["js"]):
                    sayfa.select_option("#modelSecim", model)
                    sayfa.select_option("#motorSecim", m)
                    sayfa.click("#temizle") if sayfa.is_enabled("#temizle") else None
                    sayfa.click("#hizTesti")
                    sayfa.wait_for_function("!document.getElementById('hizTesti').disabled", timeout=600000)
                    sayfa.wait_for_function("!document.getElementById('kopyala').disabled", timeout=5000)
                    metin = sayfa.evaluate("document.getElementById('olcumSonucu').innerText")
                    sayi = [float(s.replace(",", ".")) for s in re.findall(r"(\d+,\d+) ms", metin)]
                    hiz.append({"islemci_yavaslatma": yavas, "model_dosyasi": model,
                                "hesaplama": "onnxruntime-web" if m == "ort" else "yedek hesaplayici",
                                "on_isleme_ort_ms": sayi[0], "model_ort_ms": sayi[3], "model_medyan_ms": sayi[4],
                                "model_p95_ms": sayi[5], "toplam_ort_ms": sayi[6]})
                    print(hiz[-1], flush=True)
        cdp.send("Emulation.setCPUThrottlingRate", {"rate": 1})
        sayfa.close()

        # internet yokken: onnxruntime-web adresi engellenir
        cevrimdisi = baglam.new_page()
        cevrimdisi.route(ORT_ADRES + "**", lambda r: r.abort())
        cevrimdisi.goto(adres)
        cevrimdisi_motor = cevrimdisi.evaluate("Uygulama.hazir")
        skor = cevrimdisi.evaluate("([v, a]) => Uygulama.skorlar(Uygulama.cizimden28(v), 'js', a)", [vuruslar[0], modeller[0]])
        internetsiz = {"onnxruntime_web": cevrimdisi_motor, "yedek_hesaplayici_tahmin": int(np.argmax(skor)),
                       "gercek_etiket": int(y[0]),
                       "durum_mesaji": cevrimdisi.evaluate("document.getElementById('motorDurum').textContent")}
        print("Internetsiz:", internetsiz)
        tarayici_surum = tarayici.version
        tarayici.close()

    on_fark = np.abs(py28.astype(int) - js28.astype(int))
    sonuc = {"tarih": time.strftime("%Y-%m-%d %H:%M"), "tarayici": "Chromium " + tarayici_surum,
             "cizim_sayisi": int(len(y)), "on_isleme_ayni_piksel_%": round(float((on_fark == 0).mean()) * 100, 2),
             "on_isleme_en_buyuk_fark": int(on_fark.max()), "sayfa_hatalari": hatalar, "internetsiz": internetsiz}
    pd.DataFrame(satirlar).to_csv(os.path.join(CIKTI, "hafta5_tarayici_dogruluk.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(hiz).to_csv(os.path.join(CIKTI, "hafta5_tarayici_hiz.csv"), index=False, encoding="utf-8-sig")
    with open(os.path.join(CIKTI, "hafta5_tarayici_testi.json"), "w", encoding="utf-8") as f:
        json.dump(sonuc, f, ensure_ascii=False, indent=2)
    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 20)
    print("\n" + pd.DataFrame(satirlar).to_string(index=False) + "\n\n" + pd.DataFrame(hiz).to_string(index=False))
    print(json.dumps(sonuc, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
