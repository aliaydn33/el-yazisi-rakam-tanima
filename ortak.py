"""
Ortak yardimci fonksiyonlar (veri hazirlama, egitim ve veri toplama betikleri bunlari kullanir).
"""
import os
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

VERI = "veri"            # hazirlanan .npz dosyalari
HAM = os.path.join(VERI, "ham")   # indirilen ham dosyalar
CIKTI = "cikti"          # sekiller, tablolar, ozet
MODELLER = "modeller"    # egitilen modeller

for _k in (VERI, HAM, CIKTI, MODELLER):
    os.makedirs(_k, exist_ok=True)


def mnist_formatina_getir(murekkep):
    """Murekkebi parlak (arka plan 0) olan bir gri goruntuyu MNIST bicimine getirir.

    MNIST'in hazirlanma adimlari (LeCun vd.):
      1) rakami cevreleyen en kucuk dikdortgeni kirp
      2) en-boy oranini koruyarak 20x20 kutuya sigdir
      3) agirlik merkezine gore 28x28 alanin ortasina yerlestir
    Donus: 28x28 uint8 dizi (0 = arka plan, 255 = murekkep)
    """
    g = np.asarray(murekkep, dtype=np.float32)
    satirlar = np.where(g.max(axis=1) > 30)[0]
    sutunlar = np.where(g.max(axis=0) > 30)[0]
    if len(satirlar) == 0 or len(sutunlar) == 0:          # bos cizim
        return np.zeros((28, 28), np.uint8)
    kirpik = g[satirlar[0]:satirlar[-1] + 1, sutunlar[0]:sutunlar[-1] + 1]

    h, w = kirpik.shape
    olcek = 20.0 / max(h, w)
    yw, yh = max(1, round(w * olcek)), max(1, round(h * olcek))
    kucuk = np.asarray(Image.fromarray(kirpik.astype(np.uint8)).resize((yw, yh), Image.LANCZOS), np.float32)

    kutu = np.zeros((28, 28), np.float32)
    ust, sol = (28 - yh) // 2, (28 - yw) // 2
    kutu[ust:ust + yh, sol:sol + yw] = kucuk

    cy, cx = ndimage.center_of_mass(kutu)
    # kaydirmayi rakam 28x28 disina tasmayacak sekilde sinirla
    dy = float(np.clip(14 - cy, -ust, 28 - (ust + yh)))
    dx = float(np.clip(14 - cx, -sol, 28 - (sol + yw)))
    kutu = ndimage.shift(kutu, (dy, dx), order=1, mode="constant")
    return np.clip(kutu, 0, 255).astype(np.uint8)


def vuruslari_ciz(vuruslar, kenar=200, kalinlik=26, bosluk=30):
    """Dokunmatik ekrandan gelen vuruslari (her vurus [(x, y), ...]) siyah zemine beyaz cizer.

    Rakam once en uzun kenari `kenar` piksel olacak sekilde olceklenir, sonra
    `kalinlik` piksellik yuvarlak uclu cizgilerle cizilir. Bu, bizim uygulamadaki
    tuvalin yaptigi isle ayni: parmak/fare izini goruntuye cevirmek.
    Kalinlik 26 secildi: bu degerde 28x28 goruntulerdeki murekkep orani (~%16)
    MNIST'tekiyle ayni cikiyor, yani cizgiler MNIST kadar kalin oluyor.
    """
    noktalar = np.array([p for v in vuruslar for p in v], dtype=np.float32)
    x0, y0 = noktalar.min(axis=0)
    x1, y1 = noktalar.max(axis=0)
    olcek = kenar / max(x1 - x0, y1 - y0, 1e-6)
    boyut = int(kenar + 2 * bosluk)
    img = Image.new("L", (boyut, boyut), 0)
    d = ImageDraw.Draw(img)
    r = kalinlik / 2
    for v in vuruslar:
        p = [((x - x0) * olcek + bosluk, (y - y0) * olcek + bosluk) for x, y in v]
        if len(p) == 1:
            d.ellipse([p[0][0] - r, p[0][1] - r, p[0][0] + r, p[0][1] + r], fill=255)
        else:
            d.line(p, fill=255, width=kalinlik, joint="curve")
            for q in (p[0], p[-1]):
                d.ellipse([q[0] - r, q[1] - r, q[0] + r, q[1] + r], fill=255)
    return np.asarray(img)


def npz_yukle(ad):
    yol = os.path.join(VERI, ad)
    if not os.path.exists(yol):
        raise SystemExit(f"'{yol}' bulunamadi. Once 1_veri_hazirla.py dosyasini calistirin.")
    return dict(np.load(yol))
