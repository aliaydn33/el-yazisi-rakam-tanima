"""
HAFTA 7: Kagida yazilmis rakamlarin fotografindan okunmasi (IG-11, E2)

Sunucu (sunucu.py) bu dosyadaki fotografi_oku() fonksiyonunu kullanir. Adimlar (OpenCV):
  1) Gri tonlama, en uzun kenari 1600 pikselden buyukse kucultme
  2) Aydinlatma farkinin giderilmesi: kagidin parlakligi buyuk bir morfolojik kapama ile kestirilir (murekkep
     silinir, kagit kalir) ve goruntu buna bolunur. Boylece golgeli ya da bir tarafi karanlik fotograflar da
     ayni esikle ayrilabilir.
  3) Otsu esikleme: murekkep beyaz, kagit siyah olur
     Koyu murekkebe bagli soluk kisimlar da murekkep sayilir (cift esik), rakamin ince yerleri kopmasin diye.
  4) Bagli bilesenler bulunur. Cok kucuk lekeler ve fotografin kenarina degen buyuk golgeler atilir. Rakam
     yuksekligine gore kucuk bir kapama ile cizgideki kucuk kopukluklar kapatilir.
  5) Ayni sutunda ust uste binen parcalar birlestirilir (or. ayri cizilen 5'in sapkasi, kopuk bir cizgi)
  6) Parcalar satirlara ayrilir, her satir soldan saga siralanir. Aralarindaki bosluk rakam yuksekliginden
     buyukse araya bosluk konur (ayri sayilar).
  7) Her rakamin cizgi kalinligi uygulamadaki cizimlerle ayni orana getirilir (yuksekligin yaklasik %13'u),
     sonra ortak.py'deki MNIST bicimine getirme adimlari uygulanir ve ONNX modeliyle tahmin edilir.

Komut satirindan deneme: python fotograf_okuma.py fotograf.jpg
"""
import json
import os
import sys

import cv2
import numpy as np

from ortak import mnist_formatina_getir

EN_BUYUK_KENAR = 1600
HEDEF_KALINLIK = 0.13       # ortak.vuruslari_ciz: 200 piksellik cizimde 26 piksel kalinlik


def murekkep_maskesi(gri):
    """Gri goruntuden murekkep maskesi (murekkep 255, kagit 0)."""
    h, w = gri.shape
    k = max(15, int(min(h, w) / 12)) | 1           # kalin yazilmis rakamlardan da buyuk olmali
    zemin = cv2.morphologyEx(gri, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    duz = cv2.divide(gri, zemin, scale=255)                    # kagit ~255, murekkep koyu
    duz = cv2.GaussianBlur(duz, (3, 3), 0)
    esik, _ = cv2.threshold(duz, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # kagitta neredeyse hic murekkep yoksa Otsu kagit gurultusunu ayirmaya calisir; esik sinirlanir
    esik = min(esik, 200)
    # cift esik: koyu murekkebe bagli olan soluk kisimlar da alinir (rakamin ince ya da soluk yerleri kopmasin)
    guclu = duz < esik
    zayif = (duz < esik + 0.45 * (np.percentile(duz, 90) - esik)).astype(np.uint8)
    _, etiket = cv2.connectedComponents(zayif, connectivity=8)
    tut = np.unique(etiket[guclu]); tut = tut[tut > 0]
    maske = np.isin(etiket, tut).astype(np.uint8) * 255
    return cv2.morphologyEx(maske, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))


def parcalar(maske):
    """Bagli bilesenler -> [x, y, w, h] kutulari ve etiket haritasi."""
    n, etiket, ist, _ = cv2.connectedComponentsWithStats(maske, connectivity=8)
    H, W = maske.shape
    kutular = []
    for i in range(1, n):
        x, y, w, h, alan = ist[i]
        if alan < 12 or max(w, h) < 0.012 * max(H, W):
            continue                                              # toz, leke
        kenarda = x <= 1 or y <= 1 or x + w >= W - 1 or y + h >= H - 1
        if kenarda and (w > 0.5 * W or h > 0.5 * H):
            continue                                              # kagit kenari, golge
        kutular.append([int(x), int(y), int(w), int(h), [i]])
    if not kutular:
        return [], etiket
    # rakama gore cok kisa parcalar (nokta, cizik) atilir: en yuksek parcalarin medyanina gore
    yuk = np.array([k[3] for k in kutular])
    ref = np.median(yuk[yuk >= 0.5 * yuk.max()])
    kutular = [k for k in kutular if k[3] >= 0.25 * ref or k[2] >= 0.5 * ref]
    return kutular, etiket


def rakam_yuksekligi(maske):
    kutular, _ = parcalar(maske)
    if not kutular:
        return 0
    yuk = np.array([k[3] for k in kutular])
    return float(np.median(yuk[yuk >= 0.5 * yuk.max()]))


def birlestir(kutular, ref):
    """Yatayda buyuk olcude cakisan ve dikeyde yakin olan parcalari tek rakam yapar. ref: rakam yuksekligi.
    Birlesen parcanin yuksekligi bir rakamdan belirgin bicimde buyuk olacaksa (ust uste iki satir) birlestirilmez."""
    degisti = True
    while degisti:
        degisti = False
        for i in range(len(kutular)):
            for j in range(i + 1, len(kutular)):
                a, b = kutular[i], kutular[j]
                cakisma = min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])
                dikey_bosluk = max(a[1], b[1]) - min(a[1] + a[3], b[1] + b[3])
                x0, y0 = min(a[0], b[0]), min(a[1], b[1])
                x1, y1 = max(a[0] + a[2], b[0] + b[2]), max(a[1] + a[3], b[1] + b[3])
                yatay_bosluk = -cakisma
                ust_uste = cakisma >= 0.5 * min(a[2], b[2]) and dikey_bosluk < 0.35 * ref
                # birbirine cok yakin parcalar (5'in ayri yazilmis sapkasi gibi), birlesince bir rakam boyutunda kaliyorsa
                yakin = (max(yatay_bosluk, 0) ** 2 + max(dikey_bosluk, 0) ** 2) ** 0.5 < 0.12 * ref and x1 - x0 <= 1.1 * (y1 - y0)
                if (ust_uste or yakin) and y1 - y0 <= 1.3 * ref:
                    kutular[i] = [x0, y0, x1 - x0, y1 - y0, a[4] + b[4]]
                    del kutular[j]
                    degisti = True
                    break
            if degisti:
                break
    return kutular


def satirlara_ayir(kutular):
    """Kutular soldan saga gezilir; her kutu, son elemaninin dikey ortasi kendisine en yakin satira eklenir
    (fark rakam yuksekliginin yarisindan azsa). Boylece hafif egik yazilmis satirlar da ayrilabilir.
    Satirlar yukaridan asagi, kutular soldan saga siralanir."""
    satirlar = []
    for k in sorted(kutular, key=lambda k: k[0]):
        orta = k[1] + k[3] / 2
        aday = [(abs(orta - (s[-1][1] + s[-1][3] / 2)), s) for s in satirlar]
        aday = [(f, s) for f, s in aday if f < 0.5 * max(k[3], s[-1][3])]
        if aday:
            min(aday, key=lambda a: a[0])[1].append(k)
        else:
            satirlar.append([k])
    satirlar = [{"kutular": s, "y0": float(np.mean([k[1] for k in s]))} for s in satirlar]
    return sorted(satirlar, key=lambda s: s["y0"])


def rakam_goruntusu(etiket, kutu):
    """Kutudaki murekkebi alip cizgi kalinligini uygulamadaki orana getirir, MNIST bicimine cevirir."""
    x, y, w, h, ids = kutu
    parca = np.isin(etiket[y:y + h, x:x + w], ids).astype(np.uint8) * 255
    pay = max(w, h) // 4 + 2
    parca = cv2.copyMakeBorder(parca, pay, pay, pay, pay, cv2.BORDER_CONSTANT, value=0)
    # mevcut kalinlik ~ 2 * alan / cevre (ince ve uzun bir cizgi icin)
    konturlar, _ = cv2.findContours(parca, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    cevre = sum(cv2.arcLength(c, True) for c in konturlar)
    kalinlik = 2 * float((parca > 0).sum()) / max(cevre, 1)
    hedef = HEDEF_KALINLIK * max(w, h)
    if kalinlik < hedef:
        k = int(round(hedef - kalinlik)) | 1
        parca = cv2.dilate(parca, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    return mnist_formatina_getir(parca)


class FotografOkuyucu:
    def __init__(self, model_yolu, rakam_degil=None):
        import onnxruntime as ort
        self.oturum = ort.InferenceSession(model_yolu, providers=["CPUExecutionProvider"])
        self.giris = self.oturum.get_inputs()[0].name
        self.rakam_degil = rakam_degil

    def skorlar(self, x28):
        x = np.asarray(x28, np.float32).reshape(-1, 1, 28, 28) / 255
        return self.oturum.run(None, {self.giris: x})[0]

    def oku(self, bayt):
        """bayt: fotograf dosyasinin icerigi (JPEG, PNG...). Donus: sozluk (JSON'a cevrilebilir)."""
        goruntu = cv2.imdecode(np.frombuffer(bayt, np.uint8), cv2.IMREAD_GRAYSCALE)
        if goruntu is None:
            raise ValueError("Dosya bir fotoğraf olarak açılamadı.")
        H0, W0 = goruntu.shape
        olcek = min(1.0, EN_BUYUK_KENAR / max(H0, W0))
        if olcek < 1:
            goruntu = cv2.resize(goruntu, (round(W0 * olcek), round(H0 * olcek)), interpolation=cv2.INTER_AREA)
        maske = murekkep_maskesi(goruntu)
        # cizgideki kucuk kopukluklar (soluk murekkep, ince yerler) rakam boyuna gore kapatilir
        h = rakam_yuksekligi(maske)
        if h:
            k = max(3, int(round(0.08 * h))) | 1
            maske = cv2.morphologyEx(maske, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        kutular, etiket = parcalar(maske)
        kutular = birlestir(kutular, h or 1)
        satirlar = satirlara_ayir(kutular)
        sonuc = {"genislik": W0, "yukseklik": H0, "satirlar": [], "metin": ""}
        if not kutular:
            return sonuc
        x28 = [rakam_goruntusu(etiket, k) for s in satirlar for k in s["kutular"]]
        z = self.skorlar(np.stack(x28))
        p = np.exp(z - z.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
        sira = 0
        metinler = []
        for s in satirlar:
            ref = float(np.median([k[3] for k in s["kutular"]]))
            rakamlar, metin, onceki = [], "", None
            for k in s["kutular"]:
                pi = p[sira]; c = int(pi.argmax())
                emin_degil = bool(self.rakam_degil and self.rakam_degil["yontem"] == "msp"
                                  and pi.max() < self.rakam_degil["esik"])
                if onceki is not None and k[0] - (onceki[0] + onceki[2]) > ref:
                    metin += " "
                metin += str(c)
                rakamlar.append({"rakam": c, "guven": round(float(pi.max()), 4), "emin_degil": emin_degil,
                                 "kutu": [round(v / olcek) for v in k[:4]], "goruntu": x28[sira].ravel().tolist()})
                onceki = k; sira += 1
            sonuc["satirlar"].append({"metin": metin, "rakamlar": rakamlar})
            metinler.append(metin)
        sonuc["metin"] = "\n".join(metinler)
        return sonuc


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Kullanim: python fotograf_okuma.py fotograf.jpg")
    bilgi = json.load(open(os.path.join("onnx", "bilgi.json"), encoding="utf-8"))
    o = FotografOkuyucu(os.path.join("onnx", "secilen_int8.onnx"), bilgi["secilen_int8.onnx"].get("rakam_degil"))
    s = o.oku(open(sys.argv[1], "rb").read())
    print("Okunan:", s["metin"] or "(rakam bulunamadi)")
    for i, satir in enumerate(s["satirlar"], 1):
        print(f"  satir {i}: " + ", ".join(f"{r['rakam']} (%{r['guven'] * 100:.0f})" for r in satir["rakamlar"]))
