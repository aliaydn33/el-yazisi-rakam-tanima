"""
8. hafta testi: kullanici testi ekraninin otomatik denetimi, farkli ekran boyutlari ve erisilebilirlik

Gercek kullanici testi insanlarla yapilir (12_kullanici_testi.py sonuclari analiz eder). Bu betik testi
katilimcilara gondermeden once uygulamanin dogru calistigini otomatik olarak denetler:
  1) Kullanici testi akisi: uygulama "?test=<kimlik>" ile acilir. Sanal bir katilimci onay verir, 26 gorevi
     (20 rakam, 2 sayi, 2 islem, 2 rakam olmayan cizim, Aciklama dugmesi, fotograf) yapar ve SUS anketini doldurur.
     Akis telefon boyutunda dokunarak (Chrome DevTools Protocol, Input.dispatchTouchEvent) ve bilgisayar boyutunda
     fareyle iki kez yapilir. Gonderilen kayit, Google Apps Script'teki Kod.gs'nin kaydet() kurallarina gore
     denetlenir (katilimci_no, ornekler, dosya adi, en fazla 3 MB). Apps Script'e gercekten gonderilmez: istek
     tarayicida yakalanir.
     Ayrica: sayfa test ortasinda yeniden yuklenince kaldigi yerden devam etmesi, gonderme basarisiz olunca
     dosya indirme secenegi, SUS puaninin dogru hesaplanmasi.
  2) Ekran boyutlari ve erisilebilirlik (IOG-10, IOG-11, IOG-12): 360, 412, 768 ve 1280 piksel genislikte
     yatay kaydirma, gorunen butun dugmelerin boyutu (en az 44x44 CSS piksel), metinlerin renk kontrasti
     (WCAG 2.1: normal metin 4,5:1, buyuk metin 3:1) ve sayfa dili.

Bu ortamda yalnizca Chromium vardir. Firefox ve Safari (iOS'ta butun tarayicilar Safari'nin motorunu kullanir)
kullanici testindeki katilimcilarin kendi cihazlarinda denenir.
Calistirma (proje klasorunden, once 7_uygulama_olustur.py): python testler/hafta8_testi.py
"""
import argparse
import base64
import json
import os
import re
import sys
import tempfile
import time

import numpy as np

KOK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, KOK)
sys.path.insert(0, os.path.join(KOK, "testler"))
from ortak import CIKTI  # noqa: E402
import hazir_testler as ht  # noqa: E402
from hafta7_testi import ORT_ADRES, bos_port, sunucuyu_baslat  # noqa: E402

KIMLIK = "AKfycbTEST" + "x" * 40          # Apps Script dagitim kimligi bicimi; istek tarayicida yakalanir
TOHUM = 2026


def kod_gs_kaydet(metin):
    """apps_script/Kod.gs'deki kaydet() ile ayni kurallar."""
    if len(metin) > 3 * 1024 * 1024:
        raise ValueError("Gecersiz veri")
    v = json.loads(metin)
    if not v.get("katilimci_no") or not v.get("ornekler"):
        raise ValueError("Eksik veri")
    ad = re.sub(r"[^A-Za-z0-9_.-]", "", str(v.get("dosya_adi") or ("rakamlar_" + v["katilimci_no"] + ".json")))
    if not ad.endswith(".json"):
        ad += ".json"
    return {"durum": "tamam", "dosya": ad, "cizim": len(v["ornekler"])}


def yildiz(r=1.0):
    p = []
    for i in range(11):
        a = -np.pi / 2 + i * 4 * np.pi / 5
        p.append([r * np.cos(a), r * np.sin(a)])
    return [p]


def kalp():
    t = np.linspace(0, 2 * np.pi, 40)
    return [[[16 * np.sin(x) ** 3, -(13 * np.cos(x) - 5 * np.cos(2 * x) - 2 * np.cos(3 * x) - np.cos(4 * x))] for x in t]]


class Katilimci:
    """Uygulamayi kullanan sanal katilimci: dokunarak (telefon) ya da fareyle (bilgisayar) cizer."""

    def __init__(self, sayfa, dokunma):
        self.s, self.dokunma = sayfa, dokunma
        self.cdp = sayfa.context.new_cdp_session(sayfa) if dokunma else None

    def _nokta(self, tur, x, y):
        if self.dokunma:
            self.cdp.send("Input.dispatchTouchEvent", {"type": tur, "touchPoints": [] if tur == "touchEnd" else
                                                       [{"x": float(x), "y": float(y), "id": 1}]})
        elif tur == "touchStart":
            self.s.mouse.move(x, y); self.s.mouse.down()
        elif tur == "touchMove":
            self.s.mouse.move(x, y)
        else:
            self.s.mouse.up()

    def ciz(self, vuruslar, doluluk=0.6, genis=False):
        k = self.s.locator("#tuval").bounding_box()
        p = np.array([q for v in vuruslar for q in v], float); x0, y0 = p.min(0)
        w, h = max(np.ptp(p[:, 0]), 1e-6), max(np.ptp(p[:, 1]), 1e-6)
        olc = min(k["width"] * (0.85 if genis else doluluk) / w, k["height"] * doluluk / h)
        ox = k["x"] + (k["width"] - w * olc) / 2; oy = k["y"] + (k["height"] - h * olc) / 2
        for v in vuruslar:
            self._nokta("touchStart", ox + (v[0][0] - x0) * olc, oy + (v[0][1] - y0) * olc)
            for q in v[1:]:
                self._nokta("touchMove", ox + (q[0] - x0) * olc, oy + (q[1] - y0) * olc)
            self._nokta("touchEnd", 0, 0)
        self.s.wait_for_timeout(150)

    def dokun(self, secici):
        self.s.locator(secici).scroll_into_view_if_needed()
        if self.dokunma:
            self.s.locator(secici).tap()
        else:
            self.s.locator(secici).click()


def akis(tarayici, adres, cihaz, dokunma, rakamlar, foto, ekran_klasoru, ad, yeniden_yukle=True):
    """Bir katilimcinin testi bastan sona yapmasi. Donus: (gonderilen kayit, denetim sonuclari)."""
    b = tarayici.new_context(**cihaz, accept_downloads=True, locale="tr-TR")
    gonderilen, d = [], {"hatalar": []}

    def apps_script(route):
        metin = route.request.post_data or ""
        try:
            cevap = kod_gs_kaydet(metin); gonderilen.append(json.loads(metin))
        except Exception as h:
            cevap = {"durum": "hata", "mesaj": str(h)}
        route.fulfill(status=200, content_type="application/json", headers={"Access-Control-Allow-Origin": "*"},
                      body=json.dumps(cevap))
    b.route("https://script.google.com/macros/s/**", apps_script)
    s = b.new_page(); s.on("pageerror", lambda h: d["hatalar"].append(str(h)))
    k = Katilimci(s, dokunma)
    s.goto(f"{adres}/?test={KIMLIK}")
    s.wait_for_selector("#testOnay")
    s.screenshot(path=os.path.join(ekran_klasoru, f"{ad}_1_onay.png"))
    d["onaysiz_baslanamiyor"] = s.is_disabled("#testBasla")
    k.dokun("#oncekiSecim label:nth-child(1)"); k.dokun("label.onay")
    k.dokun("#testBasla")
    s.wait_for_selector("#testSerit:not([hidden])")
    rng = np.random.default_rng(TOHUM)
    gorevler = s.evaluate("Uygulama.test.durum().adimlar")
    d["gorev_sayisi"] = len(gorevler)
    for i, g in enumerate(gorevler):
        if yeniden_yukle and i == 7:                 # test ortasinda sayfa yeniden yuklenir
            s.reload(); s.wait_for_selector("#testSerit:not([hidden])")
            d["yeniden_yuklenince_sira"] = s.evaluate("Uygulama.test.durum().sira")
        if g["tur"] == "rakam":
            kisi = rakamlar[int(g["hedef"])]
            k.ciz(kisi[int(rng.integers(len(kisi)))])
            if i == 0:
                s.screenshot(path=os.path.join(ekran_klasoru, f"{ad}_2_rakam.png"))
            if i == 1:                               # bir tahmini duzeltme
                k.dokun("#duzeltAc"); k.dokun(f"#duzeltRakamlar button[data-rakam='{g['hedef']}']")
        elif g["tur"] in ("sayi", "islem"):
            if s.evaluate("document.body.className") != "sayi-modu":
                k.dokun("#modSayi")
            metin = g["hedef"].replace(" ", "")
            kisi = {r: rakamlar[r] for r in range(10)}
            k.ciz(ht.sayi_ornegi(metin, kisi, rng), doluluk=0.55, genis=True)
            if g["tur"] == "islem" and "+" in metin:
                s.screenshot(path=os.path.join(ekran_klasoru, f"{ad}_3_islem.png"))
        elif g["tur"] == "rakamdegil":
            if s.evaluate("document.body.className") == "sayi-modu":
                k.dokun("#modRakam")
            k.ciz(yildiz() if i % 2 else kalp())
        elif g["tur"] == "acikla":
            k.ciz(rakamlar[7][0]); k.dokun("#acikla")
            s.wait_for_timeout(200)
        else:
            s.set_input_files("#fotoSec", files=[{"name": "foto.jpg", "mimeType": "image/jpeg", "buffer": foto[0]}])
            s.wait_for_selector("#testYazilan:not([hidden])", timeout=20000)
            s.screenshot(path=os.path.join(ekran_klasoru, f"{ad}_4_foto.png"))
            s.fill("#testYazilan", foto[1].replace("\n", ""))
        s.wait_for_function("!document.getElementById('testSonraki').disabled", timeout=10000)
        k.dokun("#testSonraki")
        s.wait_for_function(f"Uygulama.test.durum().sira === {i + 1}", timeout=10000)
    s.wait_for_selector(".olcek")
    s.screenshot(path=os.path.join(ekran_klasoru, f"{ad}_5_anket.png"))
    d["eksik_anketle_gonderilemiyor"] = s.is_disabled("#testGonder")
    cevap = [4, 2, 5, 1, 4, 2, 5, 1, 4, 2]          # beklenen SUS puani: (3+3+4+4+3+3+4+4+3+3) * 2,5 = 85
    for m, c in enumerate(cevap):
        k.dokun(f".olcek[data-madde='{m}'] label:nth-child({c})")
    s.fill("#testYorum", "Deneme yorumu")
    k.dokun("#testGonder")
    s.wait_for_selector("#testKapat", timeout=60000)
    s.screenshot(path=os.path.join(ekran_klasoru, f"{ad}_6_tesekkur.png"))
    d["gonderim_sayisi"] = len(gonderilen)
    kayit = gonderilen[-1] if gonderilen else None
    if kayit and os.environ.get("HAFTA8_KAYIT_KLASORU"):        # analiz betigini denemek icin (depoya girmez)
        json.dump(kayit, open(os.path.join(os.environ["HAFTA8_KAYIT_KLASORU"], kayit["dosya_adi"]), "w", encoding="utf-8"), ensure_ascii=False)
    # tesekkur ekranindan sonra sayfa yeniden acilinca test tekrar baslamamali
    s.reload(); s.wait_for_selector("#testYeni"); d["gonderildikten_sonra_tekrar_acilinca"] = "teşekkür ekranı"
    k.dokun("#testKapat"); d["test_kapatilinca_adres"] = s.evaluate("location.search")
    b.close()
    return kayit, d


def kayit_denetimi(k, foto_metin):
    o = k["ornekler"]
    return {
        "dosya_adi": k["dosya_adi"], "boyut_KB": round(len(json.dumps(k)) / 1024, 1),
        "apps_script_kabul": kod_gs_kaydet(json.dumps(k))["durum"] == "tamam",
        "rakam_cizimi": len(o), "her_rakamdan_2": sorted(int(x["hedef"]) for x in o) == sorted(list(range(10)) * 2),
        "goruntu_784": all(len(x["goruntu"]) == 784 for x in o), "vurus_var": all(len(x["vuruslar"]) > 0 for x in o),
        "rakam_dogruluk_%": round(100 * np.mean([str(x["tahmin"]) == x["hedef"] for x in o]), 1),
        "duzeltme_kaydi": [x["duzeltme"] for x in o[:2]],
        "sayi_islem": [(x["hedef"], x["okunan"], x["dogru"]) for x in k["sayilar"]],
        "rakam_degil": [(x["tahmin"], x["rakam_degil"]) for x in k["rakam_degil"]],
        "acikla_bulundu": k["acikla"]["bulundu"],
        "fotograf": {"okunan": k["fotograf"]["okunan"], "yazilan": k["fotograf"]["yazilan"], "dogru": k["fotograf"]["dogru"],
                     "beklenen": foto_metin.replace("\n", "")},
        "sus": k["sus"], "sus_puani": k["sus_puani"], "sus_puani_dogru": k["sus_puani"] == 85.0,
        "yorum": k["yorum"], "onceki_katilim": k["onceki_katilim"],
        "cihaz": {x: k["cihaz"][x] for x in ("cihaz_turu", "isletim_sistemi", "tarayici", "ekran")},
        "giris_turleri": k["giris_turleri"], "acilis": k["acilis"],
        "hiz_ms": {"on_isleme": k["hiz"]["on_isleme_ms"]["medyan"], "uygulama_modeli": k["hiz"]["uygulama_modeli_ms"]["medyan"],
                   "derin": k["hiz"].get("derin_ms", {}).get("medyan"), "hesaplayan": k["hiz"]["hesaplayan"]},
        "kayitta_kisisel_alan_yok": not any(a in json.dumps(k, ensure_ascii=False).lower() for a in ("e-posta", "email", "telefon_no", "ad_soyad")),
    }


def gonderilemezse(tarayici, adres, rakamlar):
    """Apps Script'e ulasilamazsa: hata ekrani, yeniden deneme ve dosya indirme."""
    b = tarayici.new_context(viewport={"width": 412, "height": 900}, accept_downloads=True)
    b.route("https://script.google.com/macros/s/**", lambda r: r.abort())
    s = b.new_page()
    s.goto(f"{adres}/?test={KIMLIK}")
    s.wait_for_selector("#testOnay")
    s.click("#oncekiSecim label:nth-child(1)"); s.click("label.onay"); s.click("#testBasla")
    k = Katilimci(s, False)
    n = s.evaluate("Uygulama.test.durum().adimlar.length")
    for i in range(n):
        g = s.evaluate(f"Uygulama.test.durum().adimlar[{i}]")
        if g["tur"] == "rakam" or g["tur"] == "rakamdegil":
            if s.evaluate("document.body.className") == "sayi-modu":
                s.click("#modRakam")
            k.ciz(rakamlar[1][0])
        elif g["tur"] in ("sayi", "islem"):
            if s.evaluate("document.body.className") != "sayi-modu":
                s.click("#modSayi")
            k.ciz(rakamlar[1][0])
        elif g["tur"] == "acikla":
            s.click("#testAtla"); s.wait_for_function(f"Uygulama.test.durum().sira === {i + 1}"); continue
        else:
            s.click("#testAtla"); s.wait_for_function(f"Uygulama.test.durum().sira === {i + 1}"); continue
        s.wait_for_function("!document.getElementById('testSonraki').disabled")
        s.click("#testSonraki"); s.wait_for_function(f"Uygulama.test.durum().sira === {i + 1}")
    for m in range(10):
        s.click(f".olcek[data-madde='{m}'] label:nth-child(3)")
    s.click("#testGonder")
    s.wait_for_selector("#testIndir", timeout=90000)
    d = {"hata_ekrani": s.text_content("#testBaslik"), "yeniden_dene_var": s.is_visible("#testTekrar")}
    with s.expect_download() as indirme:
        s.click("#testIndir")
    yol = indirme.value.path(); k_ = json.load(open(yol, encoding="utf-8"))
    d["indirilen_dosya"] = indirme.value.suggested_filename
    d["indirilen_sus_puani"] = k_["sus_puani"]; d["atlanan_gorevler"] = {"acikla_bulundu": k_["acikla"]["bulundu"], "foto_atlandi": k_["fotograf"]["atlandi"]}
    b.close()
    return d


# ------------------------------------------------------------------------------------------ erisilebilirlik
KONTRAST_JS = r"""
() => {
  function rgb(s) { const m = s.match(/[\d.]+/g); return m ? m.map(Number) : [0, 0, 0, 1]; }
  function lum(c) { const a = c.slice(0, 3).map(v => { v /= 255; return v <= 0.03928 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4); });
                    return 0.2126 * a[0] + 0.7152 * a[1] + 0.0722 * a[2]; }
  function zemin(e) {            // ust ogelerde ilk saydam olmayan zemin (yari saydamlar ustteki renkle karistirilir)
    let renk = [255, 255, 255], katmanlar = [];
    for (let x = e; x; x = x.parentElement) {
      const c = rgb(getComputedStyle(x).backgroundColor); const a = c.length > 3 ? c[3] : 1;
      if (a > 0) { katmanlar.push([c, a]); if (a >= 1) break; }
    }
    for (let i = katmanlar.length - 1; i >= 0; i--) { const [c, a] = katmanlar[i]; renk = renk.map((v, k) => c[k] * a + v * (1 - a)); }
    return renk;
  }
  function gorunur(e) { const r = e.getBoundingClientRect(), s = getComputedStyle(e);
    return r.width > 0 && r.height > 0 && s.visibility !== "hidden" && s.display !== "none" && Number(s.opacity) > 0.05 && !e.closest("[hidden]"); }
  function saydamlik(e) { let o = 1; for (let x = e; x; x = x.parentElement) o *= Number(getComputedStyle(x).opacity); return o; }
  const sonuc = [];
  const ogeler = Array.from(document.querySelectorAll("body *")).filter(e => !["SCRIPT", "STYLE", "CANVAS", "TEXTAREA", "INPUT", "SELECT", "OPTION"].includes(e.tagName));
  for (const e of ogeler) {
    const metin = Array.from(e.childNodes).filter(n => n.nodeType === 3).map(n => n.textContent.trim()).join(" ").trim();
    if (!metin || !gorunur(e)) continue;
    if (e.closest("button:disabled")) continue;            // devre disi ogeler WCAG'de kapsam disi
    const s = getComputedStyle(e), c = rgb(s.color), a = (c.length > 3 ? c[3] : 1) * saydamlik(e), z = zemin(e);
    const on = z.map((v, k) => c[k] * a + v * (1 - a));
    const L1 = lum(on), L2 = lum(z), oran = (Math.max(L1, L2) + 0.05) / (Math.min(L1, L2) + 0.05);
    const px = parseFloat(s.fontSize), kalin = Number(s.fontWeight) >= 700 || s.fontWeight === "bold";
    const buyuk = px >= 24 || (kalin && px >= 18.66);
    sonuc.push({ metin: metin.slice(0, 40), oran: Math.round(oran * 100) / 100, gereken: buyuk ? 3 : 4.5, px: px });
  }
  return sonuc;
}
"""
HEDEF_JS = r"""
() => {
  const secici = "button, select, input:not([type=radio]):not([type=checkbox]):not([type=file]), label.dugme, summary, .olcek label, .secenekler label, label.onay, textarea";
  return Array.from(document.querySelectorAll(secici)).filter(e => {
    const r = e.getBoundingClientRect(), s = getComputedStyle(e);
    return r.width > 0 && r.height > 0 && s.visibility !== "hidden" && !e.closest("[hidden]") && !e.closest("[aria-hidden=true]");
  }).map(e => { const r = e.getBoundingClientRect();
    return { oge: e.id || e.textContent.trim().slice(0, 30) || e.tagName, genislik: Math.round(r.width * 10) / 10, yukseklik: Math.round(r.height * 10) / 10 }; });
}
"""
TASMA_JS = "() => ({ sayfa: document.documentElement.scrollWidth, pencere: window.innerWidth })"


def erisilebilirlik(tarayici, adres, ekran_klasoru):
    sonuc = {}
    for gen, yuk in [(360, 740), (412, 915), (768, 1024), (1280, 900)]:
        b = tarayici.new_context(viewport={"width": gen, "height": yuk}, has_touch=gen < 1000, is_mobile=gen < 1000)
        s = b.new_page()
        olcum = {}
        for durum, adres_, hazirla in [
            ("rakam", f"{adres}/", None), ("sayi", f"{adres}/", "#modSayi"),
            ("test_onay", f"{adres}/?test={KIMLIK}", None), ("test_gorev", f"{adres}/?test={KIMLIK}", "gorev"),
            ("test_anket", f"{adres}/?test={KIMLIK}", "anket")]:
            s.goto(adres_); s.wait_for_timeout(300)
            if durum.startswith("test"):
                s.evaluate("localStorage.clear()"); s.reload(); s.wait_for_selector("#testOnay")
            if hazirla == "#modSayi":
                s.click("#modSayi")
            if hazirla in ("gorev", "anket"):
                s.click("#oncekiSecim label:nth-child(1)"); s.click("label.onay"); s.click("#testBasla")
                s.wait_for_selector("#testSerit:not([hidden])")
            if hazirla == "anket":
                s.evaluate("""() => { const d = JSON.parse(localStorage.getItem('rakam-tanima-kullanici-testi'));
                                      d.sira = d.adimlar.length; d.bitti = true;
                                      localStorage.setItem('rakam-tanima-kullanici-testi', JSON.stringify(d)); }""")
                s.reload(); s.wait_for_selector(".olcek")
            for ac in ("#ayrinti", "#hakkinda"):
                s.evaluate(f"document.querySelector('{ac}').open = true")
            s.wait_for_timeout(150)
            t = s.evaluate(TASMA_JS)
            h = s.evaluate(HEDEF_JS)
            k = s.evaluate(KONTRAST_JS)
            kucuk = [x for x in h if x["genislik"] < 44 or x["yukseklik"] < 44]
            dusuk = [x for x in k if x["oran"] < x["gereken"]]
            olcum[durum] = {"yatay_kaydirma": t["sayfa"] > t["pencere"], "dugme_sayisi": len(h), "kucuk_dugmeler": kucuk,
                            "metin_sayisi": len(k), "dusuk_kontrast": dusuk,
                            "en_dusuk_kontrast": min((x["oran"] for x in k), default=None)}
            if gen == 360 and durum in ("rakam", "sayi", "test_gorev", "test_anket"):
                s.screenshot(path=os.path.join(ekran_klasoru, f"genislik360_{durum}.png"), full_page=durum == "rakam")
        olcum["dil"] = s.evaluate("document.documentElement.lang")
        sonuc[str(gen)] = olcum
        b.close()
    return sonuc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ort-klasoru", default=None, help="onnxruntime-web dosyalari icin yerel klasor (istege bagli)")
    args = ap.parse_args()
    os.chdir(KOK)
    from playwright.sync_api import sync_playwright
    from uygulama_testi import yerel_ort

    port = bos_port(); adres = f"http://127.0.0.1:{port}"
    sunucuyu_baslat(port, tempfile.mkdtemp(prefix="geri_bildirim_"))
    ekran_klasoru = os.path.join(CIKTI, "hafta8_ekran"); os.makedirs(ekran_klasoru, exist_ok=True)
    toplanan = ht.toplanan_kisiler()
    rakamlar = toplanan[sorted(toplanan)[3]]          # bir katilimcinin cizimleri
    foto = ht.hazir_fotograflar(1, tohum=TOHUM)[0][:2]
    sonuc = {"ortam": "Otomatik test: Chromium (Playwright), uygulama ve sunucu ayni bilgisayarda"}
    with sync_playwright() as p:
        t = p.chromium.launch(executable_path=os.environ.get("CHROMIUM") or None)
        if args.ort_klasoru:
            orig = t.new_context

            def new_context(**kw):
                c = orig(**kw); c.route(ORT_ADRES + "**", yerel_ort(args.ort_klasoru)); return c
            t.new_context = new_context
        telefon = dict(p.devices["Pixel 7"])
        telefon.pop("default_browser_type", None)
        for ad, cihaz, dokunma in [("telefon", telefon, True), ("bilgisayar", {"viewport": {"width": 1280, "height": 900}}, False)]:
            t0 = time.perf_counter()
            kayit, d = akis(t, adres, cihaz, dokunma, rakamlar, foto, ekran_klasoru, ad)
            d["sure_sn"] = round(time.perf_counter() - t0, 1)
            d["kayit"] = kayit_denetimi(kayit, foto[1]) if kayit else None
            sonuc["akis_" + ad] = d
            if ad == "telefon" and kayit:
                ornek = dict(kayit); ornek["ornekler"] = ornek["ornekler"][:2]
                json.dump(ornek, open(os.path.join(CIKTI, "hafta8_ornek_test_kaydi.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
            print(ad, json.dumps({k: v for k, v in d.items() if k != "kayit"}, ensure_ascii=False))
            print(json.dumps(d["kayit"], ensure_ascii=False)[:1500])
        sonuc["gonderilemezse"] = gonderilemezse(t, adres, rakamlar)
        print(sonuc["gonderilemezse"])
        sonuc["erisilebilirlik"] = erisilebilirlik(t, adres, ekran_klasoru)
        t.close()
    with open(os.path.join(CIKTI, "hafta8_testi.json"), "w", encoding="utf-8") as f:
        json.dump(sonuc, f, ensure_ascii=False, indent=2)
    for gen, o in sonuc["erisilebilirlik"].items():
        for durum, x in o.items():
            if isinstance(x, dict):
                print(gen, durum, "kaydirma:", x["yatay_kaydirma"], "kucuk:", [(y["oge"], y["genislik"], y["yukseklik"]) for y in x["kucuk_dugmeler"]],
                      "dusuk:", [(y["metin"], y["oran"]) for y in x["dusuk_kontrast"]])


if __name__ == "__main__":
    main()
