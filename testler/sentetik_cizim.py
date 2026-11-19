"""Sentetik "rakam degil" cizimleri: harf ve sekil sablonlarina rastgele bozulma eklenerek uretilen vuruslar.
Yalnizca 9_rakam_degil.py --deneme ile, internet olmadan betigin calistigini denemek icin kullanilir."""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ortak import mnist_formatina_getir, vuruslari_ciz   # noqa: E402

def yay(cx, cy, rx, ry, a0, a1, n=24):
    a = np.linspace(math.radians(a0), math.radians(a1), n)
    return [(cx + rx * math.cos(t), cy + ry * math.sin(t)) for t in a]


# birim kare icinde (x saga, y asagi) vurus sablonlari
HARF = {
    "A": [[(0.1, 1), (0.5, 0), (0.9, 1)], [(0.28, 0.6), (0.72, 0.6)]],
    "C": [yay(0.55, 0.5, 0.45, 0.5, -50, -310)],
    "E": [[(0.85, 0), (0.15, 0), (0.15, 1), (0.85, 1)], [(0.15, 0.5), (0.7, 0.5)]],
    "F": [[(0.85, 0), (0.15, 0), (0.15, 1)], [(0.15, 0.5), (0.7, 0.5)]],
    "H": [[(0.15, 0), (0.15, 1)], [(0.85, 0), (0.85, 1)], [(0.15, 0.5), (0.85, 0.5)]],
    "K": [[(0.15, 0), (0.15, 1)], [(0.85, 0), (0.15, 0.55)], [(0.35, 0.4), (0.85, 1)]],
    "M": [[(0.05, 1), (0.15, 0), (0.5, 0.65), (0.85, 0), (0.95, 1)]],
    "N": [[(0.15, 1), (0.15, 0), (0.85, 1), (0.85, 0)]],
    "P": [[(0.15, 1), (0.15, 0)] + yay(0.15, 0.27, 0.6, 0.27, -90, 90)],
    "R": [[(0.15, 1), (0.15, 0)] + yay(0.15, 0.27, 0.6, 0.27, -90, 90), [(0.4, 0.54), (0.85, 1)]],
    "T": [[(0.05, 0), (0.95, 0)], [(0.5, 0), (0.5, 1)]],
    "U": [[(0.15, 0)] + yay(0.5, 0.62, 0.35, 0.38, 180, 0) + [(0.85, 0)]],
    "V": [[(0.1, 0), (0.5, 1), (0.9, 0)]],
    "W": [[(0, 0), (0.25, 1), (0.5, 0.35), (0.75, 1), (1, 0)]],
    "X": [[(0.1, 0), (0.9, 1)], [(0.9, 0), (0.1, 1)]],
    "Y": [[(0.1, 0), (0.5, 0.5), (0.9, 0)], [(0.5, 0.5), (0.5, 1)]],
}
RAKAMA_BENZEYEN_HARF = {
    "O": [yay(0.5, 0.5, 0.42, 0.5, -90, 270, 40)],
    "I": [[(0.5, 0), (0.5, 1)]],
    "Z": [[(0.1, 0), (0.9, 0), (0.1, 1), (0.9, 1)]],
    "S": [yay(0.5, 0.27, 0.38, 0.27, -20, -270) + yay(0.5, 0.73, 0.38, 0.27, -90, 160)],
    "B": [[(0.15, 1), (0.15, 0)] + yay(0.15, 0.25, 0.6, 0.25, -90, 90) + yay(0.15, 0.75, 0.7, 0.25, -90, 90)],
    "G": [yay(0.55, 0.5, 0.45, 0.5, -40, -320) + [(0.95, 0.55), (0.6, 0.55)]],
}
def _yildiz():
    p = [(0.5 + 0.5 * math.cos(math.radians(-90 + 144 * k)), 0.5 + 0.5 * math.sin(math.radians(-90 + 144 * k)))
         for k in range(6)]
    return [p]
def _kalp():
    t = np.linspace(0, 2 * math.pi, 50)
    x = 16 * np.sin(t) ** 3; y = -(13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t))
    return [list(zip((x + 17) / 34, (y + 17) / 34))]
SEKIL = {
    "üçgen": [[(0.5, 0), (1, 0.9), (0, 0.9), (0.5, 0)]],
    "kare": [[(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]],
    "yıldız": _yildiz(),
    "kalp": _kalp(),
    "ok": [[(0, 0.5), (1, 0.5)], [(0.7, 0.2), (1, 0.5), (0.7, 0.8)]],
    "ev": [[(0.1, 0.45), (0.1, 1), (0.9, 1), (0.9, 0.45)], [(0, 0.5), (0.5, 0), (1, 0.5)]],
    "gülen yüz": [yay(0.5, 0.5, 0.5, 0.5, 0, 360, 48), [(0.33, 0.35), (0.34, 0.37)], [(0.66, 0.35), (0.67, 0.37)],
                  yay(0.5, 0.55, 0.25, 0.2, 20, 160)],
    "artı": [[(0.5, 0), (0.5, 1)], [(0, 0.5), (1, 0.5)]],
}


def karalama(tur, rng):
    if tur == "rastgele eğri":
        t = np.linspace(0, 1, 120); x = np.zeros_like(t); y = np.zeros_like(t)
        for k in range(1, 5):
            x += rng.normal(0, 1 / k) * np.sin(2 * math.pi * k * t + rng.uniform(0, 6))
            y += rng.normal(0, 1 / k) * np.cos(2 * math.pi * k * t + rng.uniform(0, 6))
        return [list(zip(x, y))]
    if tur == "zikzak":
        n = rng.integers(4, 9)
        return [[(i / n, float(i % 2)) for i in range(n + 1)]]
    if tur == "spiral":
        t = np.linspace(0, rng.uniform(3, 5) * math.pi, 120); r = t / t[-1]
        return [list(zip(r * np.cos(t), r * np.sin(t)))]
    # birden fazla rastgele cizgi
    return [[tuple(rng.uniform(0, 1, 2)), tuple(rng.uniform(0, 1, 2))] for _ in range(rng.integers(3, 7))]


def bozulma(vuruslar, rng):
    """Elle cizim benzeri degisiklik: sik nokta, yumusak titreme, egme, dondurme ve olcekleme."""
    aci = math.radians(rng.uniform(-12, 12)); sx, sy = rng.uniform(0.75, 1.25, 2); kayma = rng.uniform(-0.2, 0.2)
    m = np.array([[math.cos(aci), -math.sin(aci)], [math.sin(aci), math.cos(aci)]]) @ np.array([[sx, kayma], [0, sy]])
    faz = rng.uniform(0, 6, 4); genlik = rng.uniform(0.01, 0.035)
    yeni = []
    for v in vuruslar:
        p = np.array(v, dtype=np.float64)
        if len(p) > 1:                                   # noktalari sik ornekle
            ara = [p[0]]
            for a, b in zip(p[:-1], p[1:]):
                k = max(1, int(np.linalg.norm(b - a) / 0.02))
                ara += [a + (b - a) * (i + 1) / k for i in range(k)]
            p = np.array(ara)
        s = np.linspace(0, 1, len(p))
        p = p + genlik * np.stack([np.sin(2 * math.pi * 2 * s + faz[0]) + np.sin(2 * math.pi * 3 * s + faz[1]),
                                   np.sin(2 * math.pi * 2 * s + faz[2]) + np.sin(2 * math.pi * 3 * s + faz[3])], 1)
        yeni.append([tuple(q) for q in (p @ m.T) * 100])
    return yeni


def sentetik(adet=300, tohum=0):
    rng = np.random.default_rng(tohum)
    x, grup, ad = [], [], []
    kaynaklar = ([("Harf", k, v) for k, v in HARF.items()] +
                 [("Rakama benzeyen harf", k, v) for k, v in RAKAMA_BENZEYEN_HARF.items()] +
                 [("Şekil", k, v) for k, v in SEKIL.items()] +
                 [("Karalama", k, None) for k in ("rastgele eğri", "zikzak", "spiral", "rastgele çizgiler")])
    for g, k, sablon in kaynaklar:
        for _ in range(adet):
            v = bozulma(sablon if sablon is not None else karalama(k, rng), rng)
            x.append(mnist_formatina_getir(vuruslari_ciz(v))); grup.append(g); ad.append(k)
    return np.stack(x), np.array(grup), np.array(ad)


