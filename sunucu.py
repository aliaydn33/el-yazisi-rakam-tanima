"""
HAFTA 7: Uygulamanin sunucu tarafi (FastAPI)

Gereksinim analizine gore tahmin kullanicinin tarayicisinda yapilir (IG-16). Fotograftan okuma da telefonda
yapilir (uygulamanin icindeki FotografOkuma). Sunucunun uygulama icin yaptigi is, kullanicilarin duzeltmelerini
bir yerde toplamaktir:
  GET  /api/saglik           sunucunun calistigini ve yuklu modeli bildirir (uygulama sunucuyu bununla bulur)
  GET  /api/model            uygulamadaki modellerin bilgileri (onnx/bilgi.json)
  POST /api/geri-bildirim    kullanicinin duzelttigi tahminlerin kaydi (IG-09)
  GET  /api/geri-bildirim/ozet   kayit sayilari (rakam bazinda), cizimlerin kendisi verilmez
  POST /api/tahmin           28x28 goruntulerin tahmini       } uygulama kullanmaz; testler (tarayicidaki sonucla
  POST /api/fotograf         fotograftan okuma (Python)       }  karsilastirma) ve baska istemciler icin

Ayrica uygulamanin kendisini (index.html, manifest.webmanifest, sw.js, simgeler/) yayinlar. Proje klasorundeki
diger dosyalar (veri, modeller, geri bildirim kayitlari) disariya acilmaz.

Geri bildirim kayitlari geri_bildirim/ klasorune her gun icin bir dosya olarak (JSON Lines) yazilir. Kayitta
kisiyi tanimlayan bilgi (IP adresi, cihaz adi) tutulmaz (IOG-13, IOG-14).

Calistirma:  python sunucu.py                    bu bilgisayardan: http://localhost:8000
             python sunucu.py --host 0.0.0.0     ayni agdaki telefondan: http://<bilgisayarin-IP-adresi>:8000
"""
import argparse
import datetime as dt
import json
import os
import threading
import time
import uuid
from typing import List, Optional

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from fotograf_okuma import FotografOkuyucu

KLASOR = os.path.dirname(os.path.abspath(__file__))
ONNX = os.path.join(KLASOR, "onnx")
GERI_BILDIRIM = os.environ.get("GERI_BILDIRIM_KLASORU", os.path.join(KLASOR, "geri_bildirim"))   # testler baska klasor verir
MODEL = "secilen_int8.onnx"                  # uygulamadaki model (8 bit LeNet-5 benzeri CNN)
EN_BUYUK_FOTOGRAF = 10 * 1024 * 1024         # 10 MB
SURUM = "7. hafta"

bilgi = json.load(open(os.path.join(ONNX, "bilgi.json"), encoding="utf-8"))
oturum = ort.InferenceSession(os.path.join(ONNX, MODEL), providers=["CPUExecutionProvider"])
giris_adi = oturum.get_inputs()[0].name
rakam_degil = bilgi[MODEL].get("rakam_degil")
okuyucu = FotografOkuyucu(os.path.join(ONNX, MODEL), rakam_degil)
yazma_kilidi = threading.Lock()
baslangic = time.time()

app = FastAPI(title="Rakam Tanıma sunucusu", version=SURUM,
              description="El Yazısı Rakam Tanıma Projesi (41.GRUP) - fotoğraftan okuma ve düzeltme kaydı")


# ------------------------------------------------------------------------------------------- veri bicimleri
def _goruntu_dogrula(v):
    if len(v) != 784:
        raise ValueError("görüntü 28x28 = 784 değerden oluşmalı")
    if min(v) < 0 or max(v) > 255:
        raise ValueError("görüntü değerleri 0-255 arasında olmalı")
    return v


class TahminIstegi(BaseModel):
    goruntuler: List[List[int]] = Field(..., min_length=1, max_length=512,
                                        description="28x28 görüntüler, satır satır 784 değer (0 zemin, 255 mürekkep)")

    @field_validator("goruntuler")
    @classmethod
    def _kontrol(cls, g):
        for v in g:
            _goruntu_dogrula(v)
        return g


class GeriBildirim(BaseModel):
    goruntu: List[int] = Field(..., description="Modelin gördüğü 28x28 görüntü (784 değer)")
    tahmin: int = Field(..., ge=0, le=9)
    dogru: int = Field(..., ge=0, le=9, description="Kullanıcının seçtiği doğru rakam")
    guven: float = Field(..., ge=0, le=1)
    rakam_degil: bool = False
    model: str = Field("", max_length=40)
    hesaplayan: str = Field("", max_length=30)
    giris_turu: str = Field("", max_length=20)
    surum: str = Field("", max_length=20)
    zaman: str = Field("", max_length=40, description="Tarayıcıdaki kayıt zamanı")
    vuruslar: Optional[List[List[List[float]]]] = Field(None, max_length=60)

    @field_validator("goruntu")
    @classmethod
    def _kontrol(cls, v):
        return _goruntu_dogrula(v)

    @field_validator("vuruslar")
    @classmethod
    def _vurus(cls, v):
        if v is not None:
            if sum(len(s) for s in v) > 5000:
                raise ValueError("çok fazla nokta")
            if any(len(p) < 2 for s in v for p in s):
                raise ValueError("her nokta (x, y) olmalı")
        return v


# ------------------------------------------------------------------------------------------- yardimcilar
def skorlar(x):
    x = np.asarray(x, np.float32).reshape(-1, 1, 28, 28) / 255
    return oturum.run(None, {giris_adi: x})[0]


def olasilik(z):
    p = np.exp(z - z.max(1, keepdims=True))
    return p / p.sum(1, keepdims=True)


def kayitlar():
    if not os.path.isdir(GERI_BILDIRIM):
        return []
    sonuc = []
    for ad in sorted(os.listdir(GERI_BILDIRIM)):
        if ad.endswith(".jsonl"):
            with open(os.path.join(GERI_BILDIRIM, ad), encoding="utf-8") as f:
                sonuc += [json.loads(s) for s in f if s.strip()]
    return sonuc


# ------------------------------------------------------------------------------------------- API
@app.get("/api/saglik")
def saglik():
    return {"durum": "hazir", "surum": SURUM, "model": bilgi[MODEL]["model"], "model_dosyasi": MODEL,
            "calisma_suresi_sn": round(time.time() - baslangic, 1)}


@app.get("/api/model")
def model_bilgisi():
    return bilgi


@app.post("/api/tahmin")
def tahmin(istek: TahminIstegi):
    t0 = time.perf_counter()
    z = skorlar(istek.goruntuler)
    p = olasilik(z)
    sure = (time.perf_counter() - t0) * 1000
    sonuc = []
    for zi, pi in zip(z, p):
        c = int(pi.argmax())
        ret = bool(rakam_degil and rakam_degil["yontem"] == "msp" and pi.max() < rakam_degil["esik"])
        sonuc.append({"rakam": c, "guven": round(float(pi.max()), 6), "rakam_degil": ret,
                      "skorlar": [round(float(s), 6) for s in zi]})
    return {"model": MODEL, "sure_ms": round(sure, 3), "sonuclar": sonuc}


@app.post("/api/fotograf")
async def fotograf(fotograf: UploadFile = File(..., description="Kâğıda yazılmış rakamların fotoğrafı")):
    bayt = await fotograf.read(EN_BUYUK_FOTOGRAF + 1)
    if len(bayt) > EN_BUYUK_FOTOGRAF:
        raise HTTPException(413, "Fotoğraf 10 MB'tan büyük olamaz.")
    t0 = time.perf_counter()
    try:
        sonuc = okuyucu.oku(bayt)
    except ValueError as h:
        raise HTTPException(400, str(h))
    sonuc["sure_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return sonuc


@app.post("/api/geri-bildirim")
def geri_bildirim(k: GeriBildirim):
    kayit = {"kayit_no": uuid.uuid4().hex[:12], "sunucu_zamani": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
    kayit.update(k.model_dump())
    os.makedirs(GERI_BILDIRIM, exist_ok=True)
    yol = os.path.join(GERI_BILDIRIM, f"duzeltmeler_{dt.date.today():%Y%m%d}.jsonl")
    with yazma_kilidi:
        with open(yol, "a", encoding="utf-8") as f:
            f.write(json.dumps(kayit, ensure_ascii=False, separators=(",", ":")) + "\n")
        toplam = len(kayitlar())
    return {"kaydedildi": True, "kayit_no": kayit["kayit_no"], "toplam": toplam}


@app.get("/api/geri-bildirim/ozet")
def geri_bildirim_ozet():
    k = kayitlar()
    rakam = {str(r): 0 for r in range(10)}
    for x in k:
        rakam[str(x["dogru"])] += 1
    return {"toplam": len(k), "tahmin_yanlis": sum(x["tahmin"] != x["dogru"] for x in k), "dogru_rakam": rakam}


# ------------------------------------------------------------------------------------------- uygulama dosyalari
def _dosya(ad, tur=None, onbellek="no-cache"):
    yol = os.path.join(KLASOR, ad)
    if not os.path.exists(yol):
        raise HTTPException(404, f"{ad} bulunamadı. Önce 7_uygulama_olustur.py çalıştırılmalı.")
    return FileResponse(yol, media_type=tur, headers={"Cache-Control": onbellek})


@app.get("/", include_in_schema=False)
@app.get("/index.html", include_in_schema=False)
def ana_sayfa():
    return _dosya("index.html", "text/html; charset=utf-8")


@app.get("/manifest.webmanifest", include_in_schema=False)
def manifest():
    return _dosya("manifest.webmanifest", "application/manifest+json")


@app.get("/sw.js", include_in_schema=False)
def service_worker():
    return _dosya("sw.js", "text/javascript; charset=utf-8")


if os.path.isdir(os.path.join(KLASOR, "simgeler")):
    app.mount("/simgeler", StaticFiles(directory=os.path.join(KLASOR, "simgeler")), name="simgeler")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1", help="0.0.0.0 verilirse ayni agdaki cihazlar da baglanabilir")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    import uvicorn
    print(f"Uygulama: http://localhost:{args.port}   API belgeleri: http://localhost:{args.port}/docs")
    if args.host == "0.0.0.0":
        import socket
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.connect(("10.255.255.255", 1)); ip = s.getsockname()[0]; s.close()
            print(f"Ayni agdaki telefondan: http://{ip}:{args.port}")
        except OSError:
            pass
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
