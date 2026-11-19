"""
7. hafta testi: sunucu (FastAPI), telefona kurulum ve internetsiz calisma (PWA), ek ozellikler

Sunucu bu betik icinde ayri bir is parcaciginda baslatilir (geri bildirim kayitlari gecici bir klasore yazilir).
  1) Sunucu: /api/saglik, /api/model, /api/tahmin (2.000 dokunmatik test cizimi, Python'daki onnxruntime ile
     karsilastirma ve sure), /api/geri-bildirim (gecerli ve hatali kayitlar), proje dosyalarinin disariya kapali
     olmasi.
  2) Fotograftan okuma (IG-11, E2): hazir_testler.py'nin urettigi fotograflar /api/fotograf'a gonderilir.
  3) PWA (IOG-08, H4): service worker'in kurulmasi, tarayicinin kurulabilirlik denetimi (Chrome DevTools
     Protocol, Page.getInstallabilityErrors), internet kesildikten sonra sayfanin yeniden acilmasi ve tahmin
     yapilmasi.
  4) Duzeltme (IG-09, E3): tuvale cizilen rakamda "Tahmin yanlis mi?" ile dogru rakam secilir; sunucu varken
     kaydin sunucuya, internet yokken cihaza yazildigi kontrol edilir.
  5) Sayi ve islem okuma (IG-10, E1): hazirlanan sayilar ve islemler tarayicida okunur. Ayrica tek rakamlarin
     yanlislikla birden fazla sembole bolunme ve isaret sanilma orani olculur.

Kurallar 1 numarali tohumla uretilen orneklerde ayarlandi; olcum farkli tohumla (2026) uretilen orneklerde yapilir.
Calistirma (proje klasorunden, once 7_uygulama_olustur.py): python testler/hafta7_testi.py
"""
import argparse
import json
import os
import socket
import sys
import tempfile
import threading
import time
import urllib.request

import numpy as np
import pandas as pd

KOK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, KOK); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ortak import CIKTI, npz_yukle   # noqa: E402
import hazir_testler as ht            # noqa: E402

ORT_ADRES = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/"
TOHUM = 2026


def bos_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close()
    return p


def istek(adres, veri=None, tur="application/json", yontem=None):
    """Kucuk bir HTTP istemcisi (standart kutuphane). Donus: (durum kodu, govde, sure ms)."""
    if veri is not None and tur == "application/json":
        veri = json.dumps(veri).encode()
    r = urllib.request.Request(adres, data=veri, method=yontem or ("POST" if veri is not None else "GET"))
    if veri is not None:
        r.add_header("Content-Type", tur)
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(r, timeout=60) as y:
            govde, kod = y.read(), y.status
    except urllib.error.HTTPError as h:
        govde, kod = h.read(), h.code
    sure = (time.perf_counter() - t0) * 1000
    try:
        govde = json.loads(govde)
    except ValueError:
        pass
    return kod, govde, sure


def cok_parcali(alan, dosya_adi, bayt, tur="image/jpeg"):
    sinir = "----rakam" + os.urandom(8).hex()
    govde = (f"--{sinir}\r\nContent-Disposition: form-data; name=\"{alan}\"; filename=\"{dosya_adi}\"\r\n"
             f"Content-Type: {tur}\r\n\r\n").encode() + bayt + f"\r\n--{sinir}--\r\n".encode()
    return govde, f"multipart/form-data; boundary={sinir}"


def sunucuyu_baslat(port, klasor):
    os.environ["GERI_BILDIRIM_KLASORU"] = klasor
    import uvicorn
    import sunucu
    s = uvicorn.Server(uvicorn.Config(sunucu.app, host="127.0.0.1", port=port, log_level="error"))
    threading.Thread(target=s.run, daemon=True).start()
    for _ in range(100):
        if s.started:
            return s
        time.sleep(0.1)
    raise SystemExit("Sunucu baslatilamadi")


def ozet(x):
    x = np.asarray(x, float)
    return {"ortalama": round(float(x.mean()), 2), "medyan": round(float(np.median(x)), 2),
            "en_yavas_%5": round(float(np.percentile(x, 95)), 2)}


def sayi_ogeleri(dk, tk):
    """Olcumde kullanilan sayi ve islem ornekleri (dokunmatik: 300 sayi + 150 islem, toplanan: 120 + 60)."""
    rng = np.random.default_rng(TOHUM)
    ogeler = []
    for kaynak, kisiler, n_sayi, n_islem in (("dokunmatik", dk, 300, 150), ("toplanan", tk, 120, 60)):
        adlar = sorted(kisiler)
        for m in ht.hazir_metinler(n_sayi, n_islem, rng):
            kisi = adlar[int(rng.integers(len(adlar)))]
            ogeler.append({"kaynak": kaynak, "metin": m, "vuruslar": ht.sayi_ornegi(m, kisiler[kisi], rng)})
    return ogeler


def sekiller():
    """Rapordaki sekiller: basamak sayisina gore dogruluk ve hazirlanan orneklerden bazilari."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import cv2
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})
    sf = pd.read_csv(os.path.join(CIKTI, "hafta7_sayi_testi.csv"), encoding="utf-8-sig", dtype={"metin": str, "okunan": str, "derin_okunan": str})
    gruplar = [(f"{b} basamak", sf[(sf.tur == "sayı") & (sf.basamak == b)]) for b in range(1, 7)] + [("İşlem", sf[sf.tur == "işlem"])]
    fig, ax = plt.subplots(figsize=(6.2, 2.5))
    x = np.arange(len(gruplar))
    for k, (sutun, ad, renk) in enumerate((("tam_dogru", "LeNet-5 benzeri CNN (uygulama modeli)", "#2F4B7C"),
                                           ("derin_tam_dogru", "Derin CNN", "#9AB0D0"))):
        d = [g[sutun].mean() * 100 for _, g in gruplar]
        ax.bar(x + (k - 0.5) * 0.38, d, 0.36, color=renk, label=ad)
        for xi, v in zip(x, d):
            ax.text(xi + (k - 0.5) * 0.38, v + 0.8, f"{v:.0f}", ha="center", fontsize=6.5)
    ax.axhline(90, ls="--", c="#C2410C", lw=1); ax.text(5, 91.2, "E1 hedefi %90", ha="center", color="#C2410C", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels([g for g, _ in gruplar]); ax.set_ylim(50, 104); ax.set_ylabel("Tamamen doğru okunan (%)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=7, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(CIKTI, "sekil_hafta7_sayi_dogrulugu.png"), dpi=200); plt.close(fig)

    # hazirlanan ornekler: ust satirda sayi/islem cizimleri, altta fotograflar
    ogeler = sayi_ogeleri(ht.dokunmatik_kisiler(), ht.toplanan_kisiler())
    secim = [i for i in range(len(sf)) if sf.tur[i] == "işlem"][:2] + [i for i in range(len(sf)) if sf.basamak[i] == 5][:1] + \
            [i for i in range(len(sf)) if not sf.tam_dogru[i]][:1]
    from fotograf_okuma import FotografOkuyucu
    bilgi = json.load(open(os.path.join("onnx", "bilgi.json"), encoding="utf-8"))
    okuyucu = FotografOkuyucu(os.path.join("onnx", "secilen_int8.onnx"), bilgi["secilen_int8.onnx"].get("rakam_degil"))
    fig = plt.figure(figsize=(6.4, 3.3))
    for j, i in enumerate(secim):
        ax = fig.add_axes([0.01 + j * 0.25, 0.6, 0.23, 0.26])
        for v in ogeler[i]["vuruslar"]:
            a = np.array(v); ax.plot(a[:, 0], a[:, 1], c="#111", lw=1.4, solid_capstyle="round")
        ax.invert_yaxis(); ax.set_aspect("equal"); ax.axis("off")
        renk = "#1F2937" if sf.tam_dogru[i] else "#C2410C"
        ax.set_title(f"Beklenen: {sf.metin[i]}\nOkunan: {sf.okunan[i]}", fontsize=7, color=renk)
    for j in range(2):
        yol = os.path.join(CIKTI, f"hafta7_ornek_fotograf_{j + 1}.jpg")
        img = cv2.imread(yol)
        s = okuyucu.oku(open(yol, "rb").read())
        kutular = [r["kutu"] for satir in s["satirlar"] for r in satir["rakamlar"]]
        for x0, y0, w, h in kutular:
            cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), (235, 99, 37), max(2, img.shape[1] // 400))
        k = np.array(kutular); hh = int(np.median(k[:, 3]))          # rakamlarin cevresi kirpilir
        x0, y0 = max(0, k[:, 0].min() - 2 * hh), max(0, k[:, 1].min() - hh)
        x1, y1 = min(img.shape[1], (k[:, 0] + k[:, 2]).max() + 2 * hh), min(img.shape[0], (k[:, 1] + k[:, 3]).max() + hh)
        ax = fig.add_axes([0.02 + j * 0.5, 0.02, 0.46, 0.46])
        ax.imshow(img[y0:y1, x0:x1, ::-1]); ax.axis("off")
        ax.set_title("Okunan: " + s["metin"].replace("\n", " / "), fontsize=7)
    fig.savefig(os.path.join(CIKTI, "sekil_hafta7_hazir_ornekler.png"), dpi=200); plt.close(fig)
    print("Sekiller cikti/ klasorune yazildi.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ort-klasoru", default=None, help="onnxruntime-web dosyalari icin yerel klasor (istege bagli)")
    ap.add_argument("--fotograf", type=int, default=200)
    ap.add_argument("--sadece-sekil", action="store_true", help="testi calistirmadan sekilleri yeniden ciz")
    args = ap.parse_args()
    os.chdir(KOK)
    if args.sadece_sekil:
        return sekiller()
    from playwright.sync_api import sync_playwright
    from uygulama_testi import yerel_ort
    import onnxruntime as ort

    port = bos_port(); adres = f"http://127.0.0.1:{port}"
    gecici = tempfile.mkdtemp(prefix="geri_bildirim_")
    sunucuyu_baslat(port, gecici)
    sonuc = {"ortam": "Otomatik test: Chromium (Playwright) ve sunucu ayni bilgisayarda; süreler bu bilgisayara aittir"}

    # ------------------------------------------------------------------ 1) sunucu
    a = {}
    kod, g, _ = istek(adres + "/api/saglik"); a["saglik"] = {"kod": kod, "durum": g.get("durum")}
    kod, g, _ = istek(adres + "/api/model"); a["model"] = {"kod": kod, "model_sayisi": len(g)}
    d = npz_yukle("dokunmatik.npz")
    ayrim = pd.read_csv(os.path.join(CIKTI, "hafta4_dokunmatik_kisi_ayrimi.csv"), encoding="utf-8-sig")
    test = np.flatnonzero(np.isin(d["kisi"], ayrim[ayrim.kume == "test"].kisi))
    sec = np.random.default_rng(TOHUM).choice(test, 2000, replace=False)
    x, y = d["x"][sec], d["y"][sec]
    sunucu_skor, sunucu_sure = [], []
    for i in range(0, len(x), 250):
        kod, g, _ = istek(adres + "/api/tahmin", {"goruntuler": x[i:i + 250].reshape(-1, 784).tolist()})
        sunucu_skor += [r["skorlar"] for r in g["sonuclar"]]
    sunucu_skor = np.array(sunucu_skor)
    o = ort.InferenceSession(os.path.join("onnx", "secilen_int8.onnx"), providers=["CPUExecutionProvider"])
    py = o.run(None, {o.get_inputs()[0].name: x.astype(np.float32)[:, None] / 255})[0]
    tek = []
    for i in range(100):
        kod, g, s = istek(adres + "/api/tahmin", {"goruntuler": [x[i].ravel().tolist()]})
        tek.append(s); sunucu_sure.append(g["sure_ms"])
    a["tahmin"] = {"cizim": len(x), "python_ile_ayni_%": round(float((sunucu_skor.argmax(1) == py.argmax(1)).mean()) * 100, 2),
                   "skor_en_buyuk_fark": float(np.abs(sunucu_skor - py).max()),
                   "dogruluk_%": round(float((sunucu_skor.argmax(1) == y).mean()) * 100, 2),
                   "tek_cizim_istek_ms": ozet(tek), "tek_cizim_model_ms": ozet(sunucu_sure)}
    hatali = {"783 değer": {"goruntuler": [[0] * 783]}, "300 değeri": {"goruntuler": [[300] + [0] * 783]},
              "boş liste": {"goruntuler": []}}
    a["tahmin_hatali_istek"] = {k: istek(adres + "/api/tahmin", v)[0] for k, v in hatali.items()}
    # geri bildirim
    kayit = {"goruntu": x[0].ravel().tolist(), "tahmin": int(py[0].argmax()), "dogru": int(y[0]), "guven": 0.9,
             "model": "secilen_int8.onnx", "surum": "test", "vuruslar": [[[1, 2], [3, 4]]]}
    kod, g, _ = istek(adres + "/api/geri-bildirim", kayit)
    yazilan = [json.loads(s) for f in os.listdir(gecici) for s in open(os.path.join(gecici, f), encoding="utf-8")]
    a["geri_bildirim"] = {"kod": kod, "kaydedildi": g.get("kaydedildi"), "dosyadaki_kayit": len(yazilan),
                          "goruntu_ayni": yazilan[-1]["goruntu"] == kayit["goruntu"] if yazilan else False,
                          "kayitta_ip_var": any(k in yazilan[-1] for k in ("ip", "istemci", "client")) if yazilan else None}
    hatali = {"dogru=12": dict(kayit, dogru=12), "783 değer": dict(kayit, goruntu=kayit["goruntu"][:-1]),
              "güven 1,5": dict(kayit, guven=1.5), "eksik alan": {k: v for k, v in kayit.items() if k != "dogru"}}
    a["geri_bildirim_hatali"] = {k: istek(adres + "/api/geri-bildirim", v)[0] for k, v in hatali.items()}
    a["disariya_kapali"] = {p: istek(adres + p)[0] for p in
                            ("/onnx/bilgi.json", "/sunucu.py", "/veri/qmnist.npz", "/modeller/uygulama_modeli.pt",
                             "/geri_bildirim/", "/toplanan_veri/BENIOKU.txt")}
    sonuc["sunucu"] = a
    print(json.dumps(a, ensure_ascii=False, indent=1))

    # ------------------------------------------------------------------ 2) fotograftan okuma
    print("Fotograflar hazirlaniyor...")
    fotolar = ht.hazir_fotograflar(args.fotograf, tohum=TOHUM)
    satirlar = []
    for i, (jpg, metin, kaynak) in enumerate(fotolar):
        govde, tur = cok_parcali("fotograf", "f.jpg", jpg)
        kod, g, s = istek(adres + "/api/fotograf", govde, tur)
        okunan = g["metin"].replace(" ", "")
        hedef = metin.replace("\n", ""); oku = okunan.replace("\n", "")
        satirlar.append({"no": i, "kaynak": kaynak, "beklenen": metin.replace("\n", "/"), "okunan": okunan.replace("\n", "/"),
                         "tam_dogru": okunan == metin, "rakam_sayisi_dogru": len(oku) == len(hedef),
                         "satir_sayisi_dogru": okunan.count("\n") == metin.count("\n"),
                         "rakam": len(hedef), "dogru_rakam": sum(p == q for p, q in zip(oku, hedef)) if len(oku) == len(hedef) else None,
                         "sunucu_ms": g["sure_ms"], "istek_ms": round(s, 1), "kb": round(len(jpg) / 1024, 1)})
        if i < 3:
            open(os.path.join(CIKTI, f"hafta7_ornek_fotograf_{i + 1}.jpg"), "wb").write(jpg)
    f = pd.DataFrame(satirlar)
    f.to_csv(os.path.join(CIKTI, "hafta7_fotograf_testi.csv"), index=False, encoding="utf-8-sig")
    fs = {}
    for k, gr in [("tümü", f)] + list(f.groupby("kaynak")):
        tam = gr[gr.rakam_sayisi_dogru]
        fs[k] = {"fotograf": len(gr), "tam_dogru_%": round(gr.tam_dogru.mean() * 100, 1),
                 "rakam_sayisi_dogru_%": round(gr.rakam_sayisi_dogru.mean() * 100, 1),
                 "satir_sayisi_dogru_%": round(gr.satir_sayisi_dogru.mean() * 100, 1),
                 "rakam_dogrulugu_%": round(tam.dogru_rakam.sum() / tam.rakam.sum() * 100, 2),
                 "rakam_toplam": int(gr.rakam.sum())}
    fs["sure"] = {"sunucu_ms": ozet(f.sunucu_ms), "istek_ms": ozet(f.istek_ms), "fotograf_kb": ozet(f.kb)}
    kod, _, _ = istek(adres + "/api/fotograf", *cok_parcali("fotograf", "x.txt", b"fotograf degil", "text/plain"))
    fs["fotograf_olmayan_dosya_kodu"] = kod
    kod, _, _ = istek(adres + "/api/fotograf", *cok_parcali("fotograf", "b.jpg", os.urandom(10 * 1024 * 1024 + 10)))
    fs["10_MB_ustu_kodu"] = kod
    sonuc["fotograf"] = fs
    print(json.dumps(fs, ensure_ascii=False, indent=1))

    # ------------------------------------------------------------------ 5) sayi ve islem ornekleri
    dk, tk = ht.dokunmatik_kisiler(), ht.toplanan_kisiler()
    ogeler = sayi_ogeleri(dk, tk)
    tek_rakamlar = [(r, v, "dokunmatik") for k in dk.values() for r, l in k.items() for v in l] + \
                   [(r, v, "toplanan") for k in tk.values() for r, l in k.items() for v in l]

    # ------------------------------------------------------------------ 3, 4, 5) tarayici
    profil = tempfile.mkdtemp(prefix="profil_")
    t = {}
    toplanan = ht.toplanan_kisiler()
    ornek_cizim = toplanan[sorted(toplanan)[0]][3][0]           # bir katilimcinin 3'u
    with sync_playwright() as p:
        b = p.chromium.launch_persistent_context(profil, executable_path=os.environ.get("CHROMIUM") or None,
                                                 viewport={"width": 1000, "height": 900}, accept_downloads=True)
        if args.ort_klasoru:
            b.route(ORT_ADRES + "**", yerel_ort(args.ort_klasoru))
        s = b.new_page(); hatalar = []
        s.on("pageerror", lambda h: hatalar.append(str(h)))
        t0 = time.perf_counter(); s.goto(adres + "/"); motor = s.evaluate("Uygulama.hazir")
        t["ilk_acilis_sn"] = round(time.perf_counter() - t0, 2)
        s.wait_for_function("document.getElementById('pwaDurum').textContent.indexOf('Hazır') === 0", timeout=30000)
        s.wait_for_function("document.getElementById('sunucuDurum').textContent.indexOf('Bağlı') === 0", timeout=10000)
        cdp = b.new_cdp_session(s)
        m = cdp.send("Page.getAppManifest")
        t["pwa"] = {"motor": motor, "onbellek": s.evaluate("Uygulama.pwa()"), "service_worker": s.evaluate("!!navigator.serviceWorker.controller") or
                    s.evaluate("navigator.serviceWorker.ready.then(r => !!r.active)"),
                    "kurulabilirlik_hatalari": cdp.send("Page.getInstallabilityErrors")["installabilityErrors"],
                    "manifest_hatalari": m.get("errors", []), "sunucu_bulundu": s.evaluate("!!Uygulama.sunucu()"),
                    "fotograf_bolumu_gorunur": s.is_visible("#fotoKart")}

        # tarayici ile sunucu ayni tahmini veriyor mu (500 cizim)
        sira = "async ([g, m]) => { const s = []; for (const x of g) s.push(await Uygulama.skorlar(x, m)); return s; }"
        tz = np.array(s.evaluate(sira, [x[:500].reshape(500, -1).tolist(), "ort"]))
        t["tarayici_sunucu_ayni_%"] = round(float((tz.argmax(1) == sunucu_skor[:500].argmax(1)).mean()) * 100, 2)

        def ciz(vuruslar):
            k = s.locator("#tuval").bounding_box()
            p_ = np.array([q for v in vuruslar for q in v]); x0, y0 = p_.min(0); w = max(np.ptp(p_[:, 0]), np.ptp(p_[:, 1]), 1e-6)
            olc = k["height"] * 0.6 / w
            ox = k["x"] + (k["width"] - np.ptp(p_[:, 0]) * olc) / 2; oy = k["y"] + (k["height"] - np.ptp(p_[:, 1]) * olc) / 2
            for v in vuruslar:
                s.mouse.move(ox + (v[0][0] - x0) * olc, oy + (v[0][1] - y0) * olc); s.mouse.down()
                for q in v[1:]:
                    s.mouse.move(ox + (q[0] - x0) * olc, oy + (q[1] - y0) * olc)
                s.mouse.up()
            s.wait_for_timeout(400)

        # duzeltme: sunucu varken
        ciz(ornek_cizim)
        tahmin = s.text_content("#rakam")
        s.click("#duzeltAc"); s.click("#duzeltRakamlar button[data-rakam='3']")
        s.wait_for_function("document.getElementById('duzeltDurum').textContent.indexOf('kaydedildi') >= 0", timeout=10000)
        kod, oz, _ = istek(adres + "/api/geri-bildirim/ozet")
        t["duzeltme_sunucu"] = {"ekrandaki_tahmin": tahmin, "mesaj": s.text_content("#duzeltDurum"),
                                "sunucudaki_kayit": oz["toplam"], "dogru_3_sayisi": oz["dogru_rakam"]["3"]}

        # fotograf: arayuzden yukleme
        yol = os.path.join(CIKTI, "hafta7_ornek_fotograf_1.jpg")
        s.set_input_files("#fotoSec", yol)
        s.wait_for_function("document.getElementById('fotoMetin').textContent.length > 0", timeout=30000)
        t["fotograf_arayuz"] = {"beklenen": fotolar[0][1].replace("\n", "/"), "okunan": s.text_content("#fotoMetin").replace("\n", "/"),
                                "durum": s.text_content("#fotoDurum")}
        s.click("#temizle")

        # internetsiz calisma
        b.set_offline(True)
        s.wait_for_timeout(500); etiket = s.is_visible("#cevrimdisi")      # 'offline' olayiyla gorunmeli
        s.reload(); motor2 = s.evaluate("Uygulama.hazir")
        s.wait_for_function("document.getElementById('sunucuDurum').textContent.indexOf('Yok') === 0", timeout=10000)
        ciz(ornek_cizim)
        t["internetsiz"] = {"sayfa_acildi": s.evaluate("!!window.Uygulama"), "motor": motor2, "ort_durumu": s.evaluate("Uygulama.ortDurum()"),
                            "tahmin": s.text_content("#rakam"), "guven": s.text_content("#guven"),
                            "internet_yok_etiketi": etiket,
                            "sunucu": s.text_content("#sunucuDurum"), "fotograf_bolumu_gizli": s.is_hidden("#fotoKart")}
        s.click("#duzeltAc"); s.click("#duzeltRakamlar button[data-rakam='3']")
        s.wait_for_function("document.getElementById('duzeltDurum').textContent.indexOf('cihazda') >= 0", timeout=10000)
        t["duzeltme_cihazda"] = {"mesaj": s.text_content("#duzeltDurum"), "cihazdaki_kayit": s.text_content("#yerelSayi")}
        with s.expect_download() as ind:
            s.click("#duzeltmeIndir")
        dosya = json.load(open(ind.value.path(), encoding="utf-8"))
        t["duzeltme_cihazda"]["indirilen_kayit"] = len(dosya["kayitlar"])
        t["duzeltme_cihazda"]["indirilen_dogru"] = dosya["kayitlar"][0]["dogru"]
        b.set_offline(False)

        # sayi ve islem okuma (E1)
        s.click("#modSayi")
        oku = "async ([v, m]) => { const s = []; for (const x of v) s.push(await Uygulama.sayiOku(x, m)); return s; }"
        r = s.evaluate(oku, [[o_["vuruslar"] for o_ in ogeler], "ort"])
        s.select_option("#modelSecim", "derin_int8.onnx")              # karsilastirma icin derin CNN
        r_derin = s.evaluate(oku, [[o_["vuruslar"] for o_ in ogeler], "ort"])
        s.select_option("#modelSecim", "secilen_int8.onnx")
        tr = s.evaluate("v => v.map(x => { const g = SayiOkuma.grupla(x); return [g.length, g.map(y => SayiOkuma.isaret(y))]; })",
                        [v for _, v, _ in tek_rakamlar])
        t["sayfa_hatalari"] = hatalar
        b.close()
    sonuc["tarayici"] = t
    print(json.dumps(t, ensure_ascii=False, indent=1))

    sat = []
    for o_, x_, xd in zip(ogeler, r, r_derin):
        okunan = "".join(z["deger"] for z in x_["semboller"])
        beklenen = ht.islem_sonucu(o_["metin"])
        sat.append({"kaynak": o_["kaynak"], "tur": "sayı" if beklenen is None else "işlem", "metin": o_["metin"], "okunan": okunan,
                    "basamak": len(o_["metin"]) if beklenen is None else None, "sembol": len(o_["metin"]),
                    "tam_dogru": okunan == o_["metin"], "sembol_sayisi_dogru": len(okunan) == len(o_["metin"]),
                    "isaretler_dogru": [c for c in okunan if not c.isdigit()] == [c for c in o_["metin"] if not c.isdigit()],
                    "sonuc_dogru": (x_["sonuc"] == beklenen) if beklenen is not None else None,
                    "supheli": sum(z["supheli"] for z in x_["semboller"]),
                    "derin_okunan": "".join(z["deger"] for z in xd["semboller"]),
                    "derin_tam_dogru": "".join(z["deger"] for z in xd["semboller"]) == o_["metin"]})
    sf = pd.DataFrame(sat)
    sf.to_csv(os.path.join(CIKTI, "hafta7_sayi_testi.csv"), index=False, encoding="utf-8-sig")
    ss = {}
    for k, gr in [("tümü", sf)] + list(sf.groupby("kaynak")):
        ss[k] = {"örnek": len(gr), "tam_dogru_%": round(gr.tam_dogru.mean() * 100, 1),
                 "sayılar_%": round(gr[gr.tur == "sayı"].tam_dogru.mean() * 100, 1),
                 "işlemler_%": round(gr[gr.tur == "işlem"].tam_dogru.mean() * 100, 1),
                 "sembol_sayisi_dogru_%": round(gr.sembol_sayisi_dogru.mean() * 100, 1),
                 "isaretler_dogru_%": round(gr[gr.tur == "işlem"].isaretler_dogru.mean() * 100, 1),
                 "islem_sonucu_dogru_%": round(gr[gr.tur == "işlem"].sonuc_dogru.astype(float).mean() * 100, 1),
                 "derin_cnn_tam_dogru_%": round(gr.derin_tam_dogru.mean() * 100, 1)}
    ss["basamak_sayisina_gore_%"] = {int(k): round(v * 100, 1) for k, v in sf[sf.tur == "sayı"].groupby("basamak").tam_dogru.mean().items()}
    ss["basamak_sayisina_gore_derin_%"] = {int(k): round(v * 100, 1) for k, v in sf[sf.tur == "sayı"].groupby("basamak").derin_tam_dogru.mean().items()}
    from scipy.stats import binomtest
    ss["mcnemar_lenet_derin"] = {"yalniz_lenet_dogru": int((sf.tam_dogru & ~sf.derin_tam_dogru).sum()),
                                 "yalniz_derin_dogru": int((~sf.tam_dogru & sf.derin_tam_dogru).sum())}
    n01, n10 = ss["mcnemar_lenet_derin"]["yalniz_lenet_dogru"], ss["mcnemar_lenet_derin"]["yalniz_derin_dogru"]
    ss["mcnemar_lenet_derin"]["p"] = round(float(binomtest(n01, n01 + n10, 0.5).pvalue), 4) if n01 + n10 else 1.0
    bol = {"dokunmatik": [0, 0, 0], "toplanan": [0, 0, 0]}
    for (rk, v, k), (ng, isa) in zip(tek_rakamlar, tr):
        bol[k][0] += 1; bol[k][1] += ng > 1; bol[k][2] += any(isa)
    ss["tek_rakamlar"] = {k: {"rakam": n, "bolunen_%": round(b_ / n * 100, 2), "isaret_sanilan_%": round(i_ / n * 100, 2)}
                          for k, (n, b_, i_) in bol.items()}
    sonuc["sayi_islem"] = ss
    print(json.dumps(ss, ensure_ascii=False, indent=1))
    with open(os.path.join(CIKTI, "hafta7_testi.json"), "w", encoding="utf-8") as f:
        json.dump(sonuc, f, ensure_ascii=False, indent=2)
    print(f"Sonuclar '{CIKTI}/hafta7_testi.json' dosyasina yazildi.")
    sekiller()


if __name__ == "__main__":
    main()
