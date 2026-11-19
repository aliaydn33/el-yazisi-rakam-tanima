"""
HAFTA 5 - ADIM 2: Uygulama sayfasini (index.html) olustur
(7. haftada eklenen: telefona kurulum icin manifest.webmanifest, sw.js ve simgeler)

uygulama_sablon.html dosyasina 'onnx' klasorundeki modelleri (6_onnx_donusum.py'nin urettigi) ekler ve
index.html dosyasini yazar. Modeller sayfanin icine base64 metni olarak gomulur. Boylece:
  - sayfa tek bir dosyadir, cift tiklayarak acilabilir ya da GitHub Pages gibi bir yerde yayinlanabilir
  - model dosyalarini ayrica indirmek gerekmez, internet olmadan da (yedek hesaplayiciyla) calisir

7. hafta (PWA): Uygulamanin telefona kurulabilmesi ve internet olmadan da acilabilmesi icin
  - manifest.webmanifest : uygulamanin adi, simgeleri, renkleri ve tam ekran acilmasi
  - sw.js                : service worker. Ilk acilista sayfayi, simgeleri ve onnxruntime-web dosyalarini
                           cihazdaki onbellege kaydeder. Internet yokken sayfa onbellekten acilir.
                           Icindeki surum numarasi index.html'in ozetidir: sayfa degisince telefondaki
                           kopya da yenilenir.
  - simgeler/            : ana ekran simgeleri (192, 512, maskelenebilir 512, 180)

Calistirma: python 7_uygulama_olustur.py
            python 7_uygulama_olustur.py --sadece-secilen   (Derin CNN eklenmez, sayfa kuculur)
"""
import argparse
import base64
import hashlib
import json
import os

SABLON = "uygulama_sablon.html"
CIKIS = "index.html"
KLASOR = "onnx"
SIMGELER = "simgeler"
ORT_ADRES = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/"
LACIVERT, ZEMIN = (47, 75, 124), "#F5F6F8"

MANIFEST = {
    "name": "Rakam Tanıma",
    "short_name": "Rakam Tanıma",
    "description": "Ekrana çizilen rakamları tanıyan uygulama (El Yazısı Rakam Tanıma Projesi, 41.GRUP)",
    "lang": "tr",
    "start_url": "./",
    "scope": "./",
    "display": "standalone",
    "background_color": ZEMIN,
    "theme_color": "#2F4B7C",
    "icons": [
        {"src": "simgeler/simge-192.png", "sizes": "192x192", "type": "image/png", "purpose": "any"},
        {"src": "simgeler/simge-512.png", "sizes": "512x512", "type": "image/png", "purpose": "any"},
        {"src": "simgeler/simge-maskeli-512.png", "sizes": "512x512", "type": "image/png", "purpose": "maskable"},
    ],
}

# Service worker. __SURUM__ index.html'in ozetiyle degistirilir.
SW = r"""// Rakam Tanima - service worker (7_uygulama_olustur.py uretir, elle degistirilmez)
// Uygulamanin internetsiz acilabilmesi icin sayfa, simgeler ve onnxruntime-web dosyalari onbellege alinir.
var SURUM = "__SURUM__";
var SAYFA = "rakam-sayfa-" + SURUM;              // sayfa her degistiginde yeni onbellek
var ORT = "__ORT__";
var ORT_ONBELLEK = "rakam-ort-1.20.1";           // onnxruntime-web surumu degismedikce yeniden indirilmez
var SAYFA_DOSYALARI = ["./", "manifest.webmanifest", "simgeler/simge-192.png", "simgeler/simge-512.png",
                       "simgeler/simge-maskeli-512.png", "simgeler/simge-180.png"];
var ORT_DOSYALARI = ["ort.wasm.min.js", "ort-wasm-simd-threaded.mjs", "ort-wasm-simd-threaded.wasm"];

self.addEventListener("install", function (e) {
  e.waitUntil(caches.open(SAYFA).then(function (c) { return c.addAll(SAYFA_DOSYALARI); }).then(function () {
    // onnxruntime-web indirilemezse kurulum yine tamamlanir: uygulama yedek hesaplayiciyla calisir
    return caches.open(ORT_ONBELLEK).then(function (c) {
      return Promise.all(ORT_DOSYALARI.map(function (d) {
        return c.match(ORT + d).then(function (var_) {
          if (var_) return;
          return fetch(ORT + d, { mode: "cors" }).then(function (r) { if (r.ok) return c.put(ORT + d, r); }).catch(function () { });
        });
      }));
    });
  }).then(function () { return self.skipWaiting(); }));
});

self.addEventListener("activate", function (e) {
  e.waitUntil(caches.keys().then(function (adlar) {
    return Promise.all(adlar.filter(function (a) { return a.indexOf("rakam-sayfa-") === 0 && a !== SAYFA; })
                            .map(function (a) { return caches.delete(a); }));
  }).then(function () { return self.clients.claim(); }));
});

function agdanSonraOnbellek(istek, anahtar) {   // once internet (guncel surum), yoksa onbellek
  return new Promise(function (coz, reddet) {
    var bitti = false, zaman = setTimeout(function () {
      caches.match(anahtar).then(function (r) { if (r && !bitti) { bitti = true; coz(r); } });
    }, 4000);
    fetch(istek).then(function (r) {
      if (r.ok) { var k = r.clone(); caches.open(SAYFA).then(function (c) { c.put(anahtar, k); }); }
      if (!bitti) { bitti = true; clearTimeout(zaman); coz(r); }
    }).catch(function () {
      clearTimeout(zaman);
      caches.match(anahtar).then(function (r) { if (!bitti) { bitti = true; r ? coz(r) : reddet(new Error("internet yok")); } });
    });
  });
}

self.addEventListener("fetch", function (e) {
  var istek = e.request, url = new URL(istek.url);
  if (istek.method !== "GET" || url.pathname.indexOf("/api/") >= 0) return;   // sunucu istekleri onbellege alinmaz
  var kapsam = new URL(self.registration.scope);
  if (istek.mode === "navigate" && url.origin === kapsam.origin) {
    var anaSayfa = url.pathname === kapsam.pathname || url.pathname === kapsam.pathname + "index.html";
    e.respondWith(agdanSonraOnbellek(istek, anaSayfa ? "./" : istek));
    return;
  }
  if (istek.url.indexOf(ORT) === 0) {             // onnxruntime-web: once onbellek
    e.respondWith(caches.open(ORT_ONBELLEK).then(function (c) {
      return c.match(istek.url).then(function (r) {
        return r || fetch(istek).then(function (y) { if (y.ok) c.put(istek.url, y.clone()); return y; });
      });
    }));
    return;
  }
  if (url.origin === kapsam.origin && SAYFA_DOSYALARI.some(function (d) { return d !== "./" && url.pathname === kapsam.pathname + d; })) {
    e.respondWith(caches.match(istek).then(function (r) { return r || fetch(istek); }));
  }
});
"""


def simgeleri_olustur():
    """Ana ekran simgeleri: lacivert zemin uzerine elle cizilmis gibi bir 7."""
    from PIL import Image, ImageDraw
    os.makedirs(SIMGELER, exist_ok=True)

    def ciz(boyut, maskeli, yuvarlak):
        b = boyut * 4                                    # once buyuk cizilir, sonra kucultulur (yumusak kenar)
        img = Image.new("RGBA", (b, b), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        if yuvarlak:
            d.rounded_rectangle([0, 0, b - 1, b - 1], radius=int(b * 0.22), fill=LACIVERT + (255,))
        else:
            d.rectangle([0, 0, b, b], fill=LACIVERT + (255,))
        alan = 0.55 if maskeli else 0.66                 # maskeli simgede rakam ortadaki guvenli alanda kalir
        o, k = b * (1 - alan) / 2, b * alan
        nokta = lambda x, y: (o + x * k, o + y * k)
        kal = int(k * 0.15)
        beyaz, acik = (255, 255, 255, 255), (201, 211, 227, 255)
        cizgi = [nokta(0.12, 0.15), nokta(0.50, 0.11), nokta(0.86, 0.12), nokta(0.50, 0.90)]
        orta = [nokta(0.36, 0.53), nokta(0.84, 0.51)]
        for yol, renk, kalinlik in ((cizgi, beyaz, kal), (orta, acik, int(kal * 0.8))):
            d.line(yol, fill=renk, width=kalinlik, joint="curve")
            for p in yol:                                # uclarda ve koselerde yuvarlak kalem izi
                r = kalinlik / 2
                d.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r], fill=renk)
        return img.resize((boyut, boyut), Image.LANCZOS)

    ciz(192, False, True).save(os.path.join(SIMGELER, "simge-192.png"))
    ciz(512, False, True).save(os.path.join(SIMGELER, "simge-512.png"))
    ciz(512, True, False).save(os.path.join(SIMGELER, "simge-maskeli-512.png"))
    ciz(180, False, False).convert("RGB").save(os.path.join(SIMGELER, "simge-180.png"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sadece-secilen", action="store_true", help="yalnizca 4. haftada secilen modeli ekle")
    args = ap.parse_args()

    bilgi_yolu = os.path.join(KLASOR, "bilgi.json")
    if not os.path.exists(bilgi_yolu):
        raise SystemExit("onnx/bilgi.json bulunamadi. Once 6_onnx_donusum.py calistirilmali.")
    bilgi = json.load(open(bilgi_yolu, encoding="utf-8"))

    modeller = {}
    for dosya, b in bilgi.items():
        if args.sadece_secilen and not b.get("secilen"):
            continue
        with open(os.path.join(KLASOR, dosya), "rb") as f:
            modeller[dosya] = {"bilgi": b, "veri": base64.b64encode(f.read()).decode("ascii")}

    sablon = open(SABLON, encoding="utf-8").read()
    if "/*MODELLER*/" not in sablon:
        raise SystemExit("Sablonda /*MODELLER*/ isareti bulunamadi.")
    sayfa = sablon.replace("/*MODELLER*/", json.dumps(modeller, ensure_ascii=False, separators=(",", ":")))
    with open(CIKIS, "w", encoding="utf-8", newline="\n") as f:
        f.write(sayfa)

    # PWA dosyalari
    if not os.path.exists(os.path.join(SIMGELER, "simge-512.png")):
        simgeleri_olustur()
    with open("manifest.webmanifest", "w", encoding="utf-8", newline="\n") as f:
        json.dump(MANIFEST, f, ensure_ascii=False, indent=2)
    ozet = hashlib.sha256(sayfa.encode("utf-8"))
    for ad in sorted(os.listdir(SIMGELER)):
        ozet.update(open(os.path.join(SIMGELER, ad), "rb").read())
    surum = ozet.hexdigest()[:12]
    with open("sw.js", "w", encoding="utf-8", newline="\n") as f:
        f.write(SW.replace("__SURUM__", surum).replace("__ORT__", ORT_ADRES))

    print(f"{CIKIS} olusturuldu ({os.path.getsize(CIKIS) / 1024:.0f} KB). Eklenen modeller:")
    for dosya, m in modeller.items():
        b = m["bilgi"]
        print(f"  {dosya:20s} {b['model']}, {b['sayi_bicimi']}, {b['dosya_KB']} KB")
    print(f"manifest.webmanifest, sw.js (surum {surum}) ve {SIMGELER}/ yazildi.")
    print("Sayfa cift tiklanarak tarayicida acilabilir. Telefona kurulum icin sayfa https ile (GitHub Pages) "
          "ya da sunucu.py ile acilmalidir.")


if __name__ == "__main__":
    main()
