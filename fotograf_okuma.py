"""
HAFTA 7: Kagida yazilmis rakamlarin fotografindan okunmasi (IG-11, E2)

Fotograftan okuma uygulamanin icinde (index.html, FotografOkuma) telefonda yapilir. Bu dosya ayni yontemin
Python karsiligidir: sunucu.py'deki /api/fotograf bunu kullanir, testler de tarayicidaki sonucu bununla
karsilastirir. Iki tarafin ayni sonucu vermesi icin her adim JavaScript'te birebir yazilabilecek sekilde
secilmistir (kare pencereli kapama, kendi Otsu ve yuvarlama kodumuz gibi).

Adimlar:
  1) Gri tonlama, en uzun kenari 1600 pikselden buyukse kucultme
  1b) Fotografta kagidin disinda masa ya da baska seyler de gorunuyorsa kagit bulunur (parlak ve renksiz en buyuk
     bolge), yalnizca kagidin ustu okunur.
  2) Aydinlatma farkinin giderilmesi: kagidin parlakligi buyuk bir kapama (en buyuk, sonra en kucuk deger
     suzgeci) ile kestirilir (murekkep silinir, kagit kalir) ve goruntu buna bolunur. Boylece golgeli ya da bir
     tarafi karanlik fotograflar da ayni esikle ayrilabilir. Ardindan 3x3 Gauss yumusatmasi.
  3) Otsu esikleme. Koyu murekkebe bagli soluk kisimlar da murekkep sayilir (cift esik), rakamin ince yerleri
     kopmasin diye. Tek piksellik gurultu 2x2 acma ile silinir.
  4) Bagli bilesenler bulunur. Cok kucuk lekeler ve fotografin kenarina degen buyuk golgeler atilir. Rakam
     yuksekligine gore kucuk bir kapama ile cizgideki kucuk kopukluklar kapatilir.
  5) Ust uste binen ya da cok yakin parcalar birlestirilir (or. ayri cizilen 5'in sapkasi, kopuk bir cizgi)
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
FLT_EPSILON = 1.1920928955078125e-07
KOYU_ORAN = 0.0005          # en koyu %0,05: murekkebin kendisi (rakamlar fotografin cok kucuk bir kismi olabilir)
ZAYIF_ORAN = 0.3            # cift esikte soluk kisimlarin ne kadar acik olabilecegi (defter cizgileri alinmasin)


def yarim_yukari(x):
    """Tam yarida yukari yuvarlama (JavaScript'teki Math.floor(x + 0.5) ile ayni)."""
    return int(np.floor(x + 0.5))


def disk(k):
    """k x k (k tek) daire seklinde yapi elemani."""
    r = (k - 1) / 2
    y, x = np.mgrid[0:k, 0:k]
    return (((x - r) ** 2 + (y - r) ** 2) <= r * r).astype(np.uint8)


def otsu(duz, secim=None):
    """OpenCV'deki Otsu yonteminin aynisi (sinif ici varyansi en kucuk yapan esik). secim: yalnizca bu pikseller."""
    v = duz if secim is None else duz[secim]
    hist = np.bincount(v.astype(np.int64).ravel(), minlength=256)[:256].astype(np.float64)
    olcek = 1.0 / v.size
    mu = float((np.arange(256) * hist).sum()) * olcek
    q1 = mu1 = en = 0.0
    esik = 0
    for i in range(256):
        p_i = hist[i] * olcek
        mu1 *= q1
        q1 += p_i
        q2 = 1.0 - q1
        if min(q1, q2) < FLT_EPSILON or max(q1, q2) > 1.0 - FLT_EPSILON:
            continue
        mu1 = (mu1 + i * p_i) / q1
        mu2 = (mu - q1 * mu1) / q2
        s = q1 * q2 * (mu1 - mu2) * (mu1 - mu2)
        if s > en:
            en, esik = s, i
    return esik


def yuzdelik(duz, oran, secim=None):
    """Degerlerin 'oran' kadarinin altinda kaldigi en kucuk deger (siralamaya gore). secim: yalnizca bu pikseller."""
    v = duz if secim is None else duz[secim]
    hist = np.bincount(v.astype(np.int64).ravel(), minlength=256)
    hedef = int(np.ceil(oran * v.size))
    return int(np.searchsorted(np.cumsum(hist), hedef))


def kagit_maskesi(rgb):
    """Fotografta kagidin disinda masa ya da baska seyler de gorunuyorsa kagidin oldugu bolge, gorunmuyorsa None.
    Kagit: parlak ve renksiz (doygunlugu dusuk) piksellerin olusturdugu en buyuk parca. Uzerindeki yazilar ve defter
    cizgileri delik olarak kalir, delikler doldurulur. Kenardaki golge icin kagit biraz daraltilir."""
    H, W = rgb.shape[:2]
    c = rgb.astype(np.int32)
    en_b, en_k = c.max(2), c.min(2)
    p95 = yuzdelik(en_b, 0.95)
    aday = ((en_b * 100 >= 55 * p95) & ((en_b - en_k) * 100 <= 25 * en_b)).astype(np.uint8)
    k = max(5, min(H, W) // 100) | 1
    aday = cv2.morphologyEx(aday, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    n, et, ist, _ = cv2.connectedComponentsWithStats(aday, connectivity=8)
    if n < 2:
        return None
    en = 1 + int(np.argmax(ist[1:, cv2.CC_STAT_AREA]))
    if ist[en, cv2.CC_STAT_AREA] < 0.15 * H * W:
        return None
    kagit = et == en
    _, et2 = cv2.connectedComponents((~kagit).astype(np.uint8), connectivity=4)
    kenar = np.unique(np.concatenate([et2[0], et2[-1], et2[:, 0], et2[:, -1]]))
    kagit |= (~kagit) & ~np.isin(et2, kenar[kenar > 0])          # delikler (kenara ulasmayan bosluklar)
    if kagit.mean() >= 0.97:
        return None
    e = max(3, min(H, W) // 100) | 1
    return cv2.erode(kagit.astype(np.uint8), np.ones((e, e), np.uint8)).astype(bool)


def murekkep_maskesi(gri, kagit=None):
    """Gri goruntuden murekkep maskesi (murekkep 1, kagit 0). kagit verilirse yalnizca kagidin ustu."""
    h, w = gri.shape
    k = max(15, int(min(h, w) / 12)) | 1            # kalin yazilmis rakamlardan da buyuk olmali
    zemin = cv2.morphologyEx(gri, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8)).astype(np.float64)
    duz = np.where(zemin > 0, np.minimum(np.rint(gri.astype(np.float64) * 255 / np.maximum(zemin, 1)), 255), 0)
    p = np.pad(duz, 1, mode="reflect")              # 3x3 Gauss (1 2 1), kenarda yansitma
    yat = p[:, :-2] + 2 * p[:, 1:-1] + p[:, 2:]
    duz = np.floor((yat[:-2] + 2 * yat[1:-1] + yat[2:]) / 16 + 0.5)
    # Esik: Otsu. Ancak defter cizgileri gibi acik renkli cizgiler de varsa Otsu kagidi cizgilerden ayirmaya calisir.
    # Bu yuzden esik, en koyu piksellerle (murekkep) kagidin ortasindan daha acik olamaz. Kagitta neredeyse hic
    # murekkep yoksa da Otsu kagit gurultusunu ayirmaya calisir; esik 200 ile sinirlanir.
    esik = min(otsu(duz, kagit), (yuzdelik(duz, KOYU_ORAN, kagit) + yuzdelik(duz, 0.5, kagit)) / 2, 200)
    # cift esik: koyu murekkebe bagli olan soluk kisimlar da alinir (rakamin ince ya da soluk yerleri kopmasin)
    guclu = duz < esik
    zayif = (duz < esik + ZAYIF_ORAN * (yuzdelik(duz, 0.9, kagit) - esik)).astype(np.uint8)
    _, etiket = cv2.connectedComponents(zayif, connectivity=8)
    tut = np.unique(etiket[guclu]); tut = tut[tut > 0]
    m = np.isin(etiket, tut)
    # 2x2 acma (tek piksellik gurultu): once en kucuk, sonra en buyuk deger (pencere: kendisi, sol, ust, sol ust)
    a = m.copy()
    a[:, 1:] &= m[:, :-1]; a[1:, :] &= m[:-1, :]; a[1:, 1:] &= m[:-1, :-1]
    b = a.copy()
    b[:, 1:] |= a[:, :-1]; b[1:, :] |= a[:-1, :]; b[1:, 1:] |= a[:-1, :-1]
    if kagit is not None:
        b &= kagit
    return b.astype(np.uint8)


def parcalar(maske):
    """Bagli bilesenler -> [x, y, w, h, [etiket]] kutulari (ilk pikselin satir sirasina gore) ve etiket haritasi."""
    n, etiket, ist, _ = cv2.connectedComponentsWithStats(maske, connectivity=8)
    H, W = maske.shape
    _, ilk = np.unique(etiket.ravel(), return_index=True)
    kutular = []
    for i in sorted(range(1, n), key=lambda i: ilk[i]):
        x, y, w, h, alan = ist[i]
        if alan < 12 or max(w, h) < 0.012 * max(H, W):
            continue                                              # toz, leke
        kenarda = x <= 1 or y <= 1 or x + w >= W - 1 or y + h >= H - 1
        if kenarda and (w > 0.5 * W or h > 0.5 * H):
            continue                                              # kagit kenari, golge
        kutular.append([int(x), int(y), int(w), int(h), [int(i)]])
    if not kutular:
        return [], etiket
    # rakama gore cok kisa parcalar (nokta, cizik) atilir: en yuksek parcalarin medyanina gore
    yuk = np.array([k[3] for k in kutular])
    ref = float(np.median(yuk[yuk >= 0.5 * yuk.max()]))
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
                ust_uste = cakisma >= 0.5 * min(a[2], b[2]) and dikey_bosluk < 0.35 * ref
                # birbirine cok yakin parcalar (5'in ayri yazilmis sapkasi gibi), birlesince bir rakam boyutunda kaliyorsa
                yakin = (max(-cakisma, 0) ** 2 + max(dikey_bosluk, 0) ** 2) ** 0.5 < 0.12 * ref and x1 - x0 <= 1.1 * (y1 - y0)
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
        en, secilen = None, None
        for s in satirlar:
            f = abs(orta - (s[-1][1] + s[-1][3] / 2))
            if f < 0.5 * max(k[3], s[-1][3]) and (en is None or f < en):
                en, secilen = f, s
        if secilen is not None:
            secilen.append(k)
        else:
            satirlar.append([k])
    satirlar = [{"kutular": s, "y0": sum(k[1] for k in s) / len(s)} for s in satirlar]
    return sorted(satirlar, key=lambda s: s["y0"])


def rakam_goruntusu(etiket, kutu):
    """Kutudaki murekkebi alip cizgi kalinligini uygulamadaki orana getirir, MNIST bicimine cevirir."""
    x, y, w, h, ids = kutu
    pay = max(w, h) // 4 + 2
    parca = np.zeros((h + 2 * pay, w + 2 * pay), np.uint8)
    parca[pay:pay + h, pay:pay + w] = np.isin(etiket[y:y + h, x:x + w], ids)
    # mevcut kalinlik ~ 2 * alan / kenar pikseli sayisi (ince ve uzun bir cizgide kenar pikselleri iki yandadir)
    m = parca > 0
    ic = m[1:-1, 1:-1] & m[:-2, 1:-1] & m[2:, 1:-1] & m[1:-1, :-2] & m[1:-1, 2:]
    alan = int(m.sum()); kenar = alan - int(ic.sum())
    kalinlik = 2 * alan / max(kenar, 1)
    hedef = HEDEF_KALINLIK * max(w, h)
    if kalinlik < hedef:
        k = yarim_yukari(hedef - kalinlik) | 1
        parca = cv2.dilate(parca, disk(k))
    return mnist_formatina_getir(parca * 255)


def gri_oku(gri, rgb=None):
    """gri: (H, W) uint8 (en uzun kenar 1600'u gecmeyen), rgb: ayni boyutta renkli goruntu (kagidi bulmak icin).
    Donus: kutular, satirlar ve 28x28 goruntuler."""
    maske = murekkep_maskesi(gri, kagit_maskesi(rgb) if rgb is not None else None)
    h = rakam_yuksekligi(maske)
    if h:
        k = max(3, yarim_yukari(0.08 * h)) | 1
        maske = cv2.morphologyEx(maske, cv2.MORPH_CLOSE, disk(k))
    kutular, etiket = parcalar(maske)
    kutular = birlestir(kutular, h or 1)
    satirlar = satirlara_ayir(kutular)
    x28 = [rakam_goruntusu(etiket, k) for s in satirlar for k in s["kutular"]]
    return satirlar, x28


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
        renkli = cv2.imdecode(np.frombuffer(bayt, np.uint8), cv2.IMREAD_COLOR)
        if renkli is None:
            raise ValueError("Dosya bir fotoğraf olarak açılamadı.")
        H0, W0 = renkli.shape[:2]
        olcek = min(1.0, EN_BUYUK_KENAR / max(H0, W0))
        if olcek < 1:
            renkli = cv2.resize(renkli, (round(W0 * olcek), round(H0 * olcek)), interpolation=cv2.INTER_AREA)
        sonuc = self.gri_oku(cv2.cvtColor(renkli, cv2.COLOR_BGR2GRAY), renkli[..., ::-1])
        sonuc["genislik"], sonuc["yukseklik"] = W0, H0
        for s in sonuc["satirlar"]:
            for r in s["rakamlar"]:
                r["kutu"] = [round(v / olcek) for v in r["kutu"]]
        return sonuc

    def gri_oku(self, gri, rgb=None):
        satirlar, x28 = gri_oku(gri, rgb)
        sonuc = {"genislik": gri.shape[1], "yukseklik": gri.shape[0], "satirlar": [], "metin": ""}
        if not x28:
            return sonuc
        z = self.skorlar(np.stack(x28))
        p = np.exp(z - z.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
        sira, metinler = 0, []
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
                                 "kutu": list(k[:4]), "goruntu": x28[sira].ravel().tolist()})
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
