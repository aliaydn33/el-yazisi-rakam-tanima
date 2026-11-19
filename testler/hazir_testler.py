"""
7. hafta testleri icin hazirlanan sayilar, islemler ve fotograflar (IG-10 / E1 ve IG-11 / E2)

Gereksinim analizinde bu iki ozellik "hazirlanan test sayilari ve fotograflari" ile dogrulanacakti. Elimizde
gercek kisilerin cok basamakli sayi cizimleri ve rakam fotograflari olmadigi icin test ornekleri gercek
kisilerin tek tek cizdigi rakamlardan olusturulur:

  Sayi ve islemler: Dokunmatik veri setinde test kumesine ayrilan 202 kisiden (4. hafta) ve ekibin topladigi
    veriden (20 katilimci) bir kisi secilir, sayinin her rakami icin o kisinin o rakamdan cizdigi orneklerden biri
    alinir. Rakamlar ayni yukseklige getirilip soldan saga, aralarinda rakam yuksekliginin %15-50'si kadar bosluk
    birakilarak dizilir. Isaretler (+, -, x) icin gercek cizim olmadigindan bunlar programla, rastgele egim, boy,
    kayma, titreme, basiklik ve bastaki kucuk kanca eklenerek cizilir.
  Fotograflar: Gercek kagit taramasi olan MNIST test kumesinden (eğitimde kullanilmadi) ve toplanan verinin
    vuruslarindan secilen rakamlar kagit rengindeki bir zemine murekkep rengiyle yazilir. Ardindan fotografa
    benzemesi icin isik farki, golge, hafif donme ve perspektif, bulaniklik, gurultu ve JPEG sikistirmasi eklenir.

Hepsi sabit tohumla uretilir, ayni tohum ayni ornekleri verir.
"""
import glob
import json
import os
import sys

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ortak import CIKTI, HAM, VERI   # noqa: E402

EKSI, CARPI = "−", "×"


# ------------------------------------------------------------------------------------------- kaynak cizimler
def dokunmatik_kisiler():
    """Dokunmatik veri setinin test kisileri: {kisi: {rakam: [vuruslar, ...]}}"""
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
    kisiler = {}
    for r in glyph.itertuples():
        v = vurus.get(r.Z_PK)
        if v:
            kisiler.setdefault(int(r.ZSUBJECT), {}).setdefault(int(r.ZCHARACTER), []).append([[list(p) for p in s] for s in v])
    return kisiler


def toplanan_kisiler(klasor="toplanan_veri"):
    """Veri toplama sayfasinin dosyalari: {katilimci: {rakam: [vuruslar, ...]}}"""
    kisiler, gorulen = {}, set()
    for yol in sorted(glob.glob(os.path.join(klasor, "*.json"))):
        o = json.load(open(yol, encoding="utf-8"))
        for orn in o.get("ornekler", []):
            anahtar = (o["katilimci_no"], orn["sira"])
            v = [[[p[0], p[1]] for p in s] for s in orn["vuruslar"] if len(s)]
            if anahtar in gorulen or not v:
                continue
            gorulen.add(anahtar)
            kisiler.setdefault(o["katilimci_no"], {}).setdefault(int(orn["rakam"]), []).append(v)
    return kisiler


# ------------------------------------------------------------------------------------------- sayi ve islemler
def _cizgi(x0, y0, x1, y1, rng, n=14, titreme=0.012, egrilik=0.05):
    t = np.linspace(0, 1, n)
    x, y = x0 + (x1 - x0) * t, y0 + (y1 - y0) * t
    L = np.hypot(x1 - x0, y1 - y0)
    nx, ny = -(y1 - y0) / max(L, 1e-6), (x1 - x0) / max(L, 1e-6)
    bukum = rng.uniform(-egrilik, egrilik) * L * 4 * t * (1 - t)
    x = x + nx * bukum + rng.normal(0, titreme * L, n)
    y = y + ny * bukum + rng.normal(0, titreme * L, n)
    p = [[float(a), float(b)] for a, b in zip(x, y)]
    if rng.random() < 0.3:                       # kalemin basta yaptigi kucuk kanca
        k = rng.uniform(0.04, 0.08) * L * (1 if rng.random() < 0.5 else -1)
        p = [[p[0][0] + nx * k, p[0][1] + ny * k]] + p
    return p


def isaret_ciz(tur, boy, rng):
    """Merkezi (0, 0) olan, boyu yaklasik 'boy' olan bir isaret. Donus: vuruslar."""
    def don(x, y, aci):
        c, s = np.cos(aci), np.sin(aci)
        return x * c - y * s, x * s + y * c
    egim = np.radians(rng.normal(0, 6))
    r = boy / 2
    if tur == EKSI:
        L = r * rng.uniform(0.8, 1.2)
        a, b = don(-L, 0, egim), don(L, 0, egim)
        return [_cizgi(*a, *b, rng)]
    if tur == "+":
        kx, ky = rng.uniform(-0.15, 0.15) * r, rng.uniform(-0.15, 0.15) * r
        yatay = _cizgi(*don(-r, ky, egim), *don(r, ky, egim), rng)
        dikey = _cizgi(*don(kx, -r, egim), *don(kx, r, egim), rng)
        return [yatay, dikey] if rng.random() < 0.5 else [dikey, yatay]
    if tur == CARPI:
        a1, a2 = np.radians(45 + rng.normal(0, 7)), np.radians(135 + rng.normal(0, 7))
        bas = rng.uniform(0.6, 1.0)              # bazi kisiler carpiyi yatik (basik) yazar
        d1 = _cizgi(-r * np.cos(a1), -r * np.sin(a1) * bas, r * np.cos(a1), r * np.sin(a1) * bas, rng)
        d2 = _cizgi(-r * np.cos(a2), -r * np.sin(a2) * bas, r * np.cos(a2), r * np.sin(a2) * bas, rng)
        return [d1, d2] if rng.random() < 0.5 else [d2, d1]
    raise ValueError(tur)


def sayi_ornegi(metin, kisi, rng, H=100.0):
    """metin: '125+37' gibi (isaretler '+', EKSI, CARPI). kisi: {rakam: [vuruslar]}. Donus: vuruslar."""
    tum, x = [], 20.0
    for ch in metin:
        if ch.isdigit():
            orn = kisi[int(ch)][rng.integers(len(kisi[int(ch)]))]
            p = np.array([q for s in orn for q in s], np.float64)
            x0, y0 = p.min(0); x1, y1 = p.max(0)
            olcek = H * rng.uniform(0.92, 1.08) / max(y1 - y0, 1e-6)
            w = (x1 - x0) * olcek
            dy = 20 + H / 2 + rng.normal(0, 0.04 * H) - (y1 - y0) * olcek / 2
            tum += [[[(q[0] - x0) * olcek + x, (q[1] - y0) * olcek + dy] for q in s] for s in orn]
        else:
            boy = H * rng.uniform(0.35, 0.6)
            w = boy * 1.1
            cx, cy = x + w / 2, 20 + H / 2 + rng.normal(0, 0.05 * H)
            tum += [[[q[0] + cx, q[1] + cy] for q in s] for s in isaret_ciz(ch, boy, rng)]
        x += w + rng.uniform(0.15, 0.5) * H
    return tum


def hazir_metinler(n_sayi, n_islem, rng):
    """En fazla 6 basamakli sayilar (1-6 basamak esit sayida) ve 2-3 sayili islemler."""
    def sayi(basamak):
        s = str(rng.integers(1, 10)) if basamak > 1 else str(rng.integers(0, 10))
        return s + "".join(str(rng.integers(0, 10)) for _ in range(basamak - 1))
    metinler = [sayi(1 + i % 6) for i in range(n_sayi)]
    for i in range(n_islem):
        k = 3 if i % 4 == 3 else 2
        parca = [sayi(int(rng.integers(1, 4)))]
        for _ in range(k - 1):
            parca += [("+", EKSI, CARPI)[int(rng.integers(3))], sayi(int(rng.integers(1, 4)))]
        metinler.append("".join(parca))
    return metinler


def islem_sonucu(metin):
    """Beklenen sonuc (carpma once)."""
    import re
    p = re.findall(r"\d+|[+−×]", metin)
    if len(p) == 1:
        return None
    t = [int(p[0])]
    for i in range(1, len(p), 2):
        if p[i] == CARPI:
            t[-1] *= int(p[i + 1])
        else:
            t += [p[i], int(p[i + 1])]
    s = t[0]
    for i in range(1, len(t), 2):
        s = s + t[i + 1] if t[i] == "+" else s - t[i + 1]
    return s


# ------------------------------------------------------------------------------------------- fotograflar
def _rakam_mnist(x28, h, rng):
    """MNIST goruntusu -> h piksel yuksekliginde murekkep yogunlugu (0-1)."""
    ys, xs = np.where(x28 > 20)
    k = x28[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(np.float32) / 255
    olcek = h / k.shape[0]
    return cv2.resize(k, (max(2, round(k.shape[1] * olcek)), h), interpolation=cv2.INTER_CUBIC).clip(0, 1)


def _rakam_vurus(v, h, rng):
    """Vuruslar -> kalemle yazilmis gibi h piksel yuksekliginde murekkep yogunlugu."""
    p = np.array([q for s in v for q in s], np.float64)
    x0, y0 = p.min(0); x1, y1 = p.max(0)
    olcek = h / max(y1 - y0, 1e-6)
    kal = max(2, int(round(h * rng.uniform(0.05, 0.09))))
    W = int((x1 - x0) * olcek) + 2 * kal + 2
    img = np.zeros((h + 2 * kal + 2, W), np.uint8)
    for s in v:
        q = np.array([[(a - x0) * olcek + kal + 1, (b - y0) * olcek + kal + 1] for a, b in s], np.int32)
        if len(q) == 1:
            cv2.circle(img, tuple(q[0]), kal // 2, 255, -1, cv2.LINE_AA)
        else:
            cv2.polylines(img, [q], False, 255, kal, cv2.LINE_AA)
    return img.astype(np.float32) / 255


def fotograf_ornegi(rakamlar, kaynak, rng):
    """rakamlar: satir listesi, her satir [(rakam, kaynak_ornegi), ...]. Donus: (jpeg baytlari, beklenen metin)."""
    W = int(rng.integers(1100, 1601)); H = int(W * rng.uniform(0.55, 0.8))
    kagit = np.array([rng.uniform(205, 245), rng.uniform(205, 240), rng.uniform(200, 235)], np.float32)
    murekkep = np.array([rng.uniform(15, 70), rng.uniform(15, 60), rng.uniform(20, 90)], np.float32)
    yog = np.zeros((H, W), np.float32)
    h = int(rng.integers(70, 150))
    satir_y = (H - len(rakamlar) * h * 1.6) / 2 + h * 0.3
    metin = []
    for satir in rakamlar:
        boylar = rng.uniform(0.9, 1.1, len(satir)); bosluk = rng.uniform(0.25, 0.6, len(satir))
        for _ in range(5):                       # satir fotografa sigmiyorsa rakamlar kucultulur
            parcalar = [(_rakam_mnist(o, int(h * k), rng) if kaynak == "mnist" else _rakam_vurus(o, int(h * k), rng))
                        for (_, o), k in zip(satir, boylar)]
            gen = sum(p.shape[1] for p in parcalar) + sum(bosluk[1:]) * h
            if gen <= 0.85 * W:
                break
            h = int(h * 0.85 * W / gen)
        x = (W - gen) / 2 + rng.normal(0, 0.02 * W)
        for p, bo in zip(parcalar, bosluk):
            y = int(satir_y + rng.normal(0, 0.04 * h)); xi = int(max(5, min(W - p.shape[1] - 5, x)))
            y = max(5, min(H - p.shape[0] - 5, y))
            yog[y:y + p.shape[0], xi:xi + p.shape[1]] = np.maximum(yog[y:y + p.shape[0], xi:xi + p.shape[1]], p)
            x += p.shape[1] + bo * h
        satir_y += h * 1.6
        metin.append("".join(str(r) for r, _ in satir))
    # isik farki: dogrusal egim + kenarlarda kararma + bir tarafta golge
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    a = rng.uniform(0, 2 * np.pi)
    isik = 1 - rng.uniform(0.05, 0.25) * ((np.cos(a) * xx / W + np.sin(a) * yy / H) + 1) / 2
    isik *= 1 - rng.uniform(0.05, 0.2) * (((xx / W - 0.5) ** 2 + (yy / H - 0.5) ** 2) * 2)
    if rng.random() < 0.5:
        kenar = rng.uniform(0.2, 0.8) * W
        golge = 1 / (1 + np.exp(-(xx - kenar) / (0.03 * W)))
        isik *= 1 - rng.uniform(0.15, 0.35) * (golge if rng.random() < 0.5 else 1 - golge)
    yog *= rng.uniform(0.85, 1.0)
    img = (kagit[None, None] * (1 - yog[..., None]) + murekkep[None, None] * yog[..., None]) * isik[..., None]
    # hafif donme ve perspektif
    d = rng.uniform(-0.025, 0.025, (4, 2)) * [W, H]
    kaynak_k = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
    M = cv2.getPerspectiveTransform(kaynak_k, (kaynak_k + d).astype(np.float32))
    R = cv2.getRotationMatrix2D((W / 2, H / 2), rng.uniform(-3, 3), 1.0)
    M = np.vstack([R, [0, 0, 1]]) @ M
    img = cv2.warpPerspective(img, M, (W, H), borderMode=cv2.BORDER_REPLICATE)
    img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.6, 1.4))
    img = img + rng.normal(0, rng.uniform(2, 7), img.shape)
    ok, jpg = cv2.imencode(".jpg", img.clip(0, 255).astype(np.uint8)[..., ::-1].copy(),
                           [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(60, 91))])
    return jpg.tobytes(), "\n".join(metin)


def hazir_fotograflar(n, tohum=0):
    """n/2 MNIST test kumesinden, n/2 toplanan verinin vuruslarindan. Donus: [(jpeg, metin, kaynak)]"""
    rng = np.random.default_rng(tohum)
    q = np.load(os.path.join(VERI, "qmnist.npz"))
    xm, ym = q["x_test10k"], q["y_test10k"]
    tk = toplanan_kisiler()
    adlar = sorted(tk)
    sonuc = []
    for i in range(n):
        kaynak = "mnist" if i % 2 == 0 else "toplanan"
        satir_sayisi = 2 if rng.random() < 0.2 else 1
        satirlar = []
        kisi = tk[adlar[int(rng.integers(len(adlar)))]]
        for _ in range(satir_sayisi):
            uzunluk = int(rng.integers(3, 9)) if satir_sayisi == 1 else int(rng.integers(2, 6))
            satir = []
            for _ in range(uzunluk):
                r = int(rng.integers(10))
                if kaynak == "mnist":
                    secim = np.flatnonzero(ym == r)
                    satir.append((r, xm[secim[rng.integers(len(secim))]]))
                else:
                    satir.append((r, kisi[r][int(rng.integers(len(kisi[r])))]))
            satirlar.append(satir)
        jpg, metin = fotograf_ornegi(satirlar, kaynak, rng)
        sonuc.append((jpg, metin, kaynak))
    return sonuc
