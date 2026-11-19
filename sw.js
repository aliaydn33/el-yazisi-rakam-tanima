// Rakam Tanima - service worker (7_uygulama_olustur.py uretir, elle degistirilmez)
// Uygulamanin internetsiz acilabilmesi icin sayfa, simgeler ve onnxruntime-web dosyalari onbellege alinir.
var SURUM = "98ccc1a7f7d2";
var SAYFA = "rakam-sayfa-" + SURUM;              // sayfa her degistiginde yeni onbellek
var ORT = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
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
