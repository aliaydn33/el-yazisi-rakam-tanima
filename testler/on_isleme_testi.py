"""
Cizim ekranindaki (cizim.html) JavaScript on islemesinin Python'daki on islemeyle ayni sonucu verip
vermedigini test eder.

Yontem: Dokunmatik ekran veri setinden rastgele secilen cizimlerin vurus koordinatlari hem Python'daki
(ortak.py) hem tarayicidaki (cizim.html) on islemeden gecirilir. Iki 28x28 goruntu piksel piksel
karsilastirilir ve ayni CNN modeline verilerek tahminlerin ayni olup olmadigina bakilir.

Gereksinim: playwright (pip install playwright, ardindan: playwright install chromium)
Calistirma (proje klasorunden): python testler/on_isleme_testi.py --model modeller/secilen_model_hafta4.pt
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ortak import CIKTI, HAM, mnist_formatina_getir, vuruslari_ciz           # noqa: E402
from model_yapisi import MODEL_SINIFLARI, tahmin_et                          # noqa: E402


def vuruslari_oku(n, tohum=0):
    k = os.path.join(HAM, "dokunmatik")
    glyph = pd.read_csv(os.path.join(k, "glyph.csv"))
    stroke = pd.read_csv(os.path.join(k, "stroke.csv"))
    touch = pd.read_csv(os.path.join(k, "touch.csv"))
    glyph = glyph[glyph.ZVALID == 1]
    touch = touch.sort_values(["ZSTROKE", "Z_FOK_STROKE"])
    nokta = {s: list(zip(g.ZX.values.tolist(), g.ZY.values.tolist())) for s, g in touch.groupby("ZSTROKE")}
    stroke = stroke.sort_values(["ZGLYPH", "Z_FOK_GLYPH"])
    vurus = {g_: [nokta[s] for s in g.Z_PK.values if s in nokta] for g_, g in stroke.groupby("ZGLYPH")}
    secim = glyph.sample(n=n, random_state=tohum)
    return [(vurus[r.Z_PK], int(r.ZCHARACTER)) for r in secim.itertuples() if vurus.get(r.Z_PK)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--model", default=os.path.join("modeller", "secilen_model_hafta4.pt"))
    ap.add_argument("--model-adi", default=None, help="MLP, LeNet-5 benzeri CNN veya Derin CNN")
    args = ap.parse_args()
    from playwright.sync_api import sync_playwright

    ornekler = vuruslari_oku(args.n)
    py = np.stack([mnist_formatina_getir(vuruslari_ciz(v)) for v, _ in ornekler])
    y = np.array([e for _, e in ornekler])
    html = "file://" + os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "cizim.html"))
    with sync_playwright() as p:
        tarayici = p.chromium.launch(executable_path=os.environ.get("CHROMIUM") or None)
        sayfa = tarayici.new_page()
        sayfa.goto(html)
        js = np.array(sayfa.evaluate(
            "v => v.map(c => Array.from(OnIsleme.cizimden28(c)))",
            [[[list(q) for q in s] for s in v] for v, _ in ornekler]), dtype=np.uint8).reshape(-1, 28, 28)
        sure = sayfa.evaluate("v => { const t = performance.now(); for (const c of v) OnIsleme.cizimden28(c); "
                              "return (performance.now() - t) / v.length; }",
                              [[[list(q) for q in s] for s in v] for v, _ in ornekler[:100]])
        tarayici.close()

    fark = np.abs(py.astype(int) - js.astype(int))
    ad = args.model_adi
    if ad is None:
        bilgi = os.path.splitext(args.model)[0] + ".txt"
        ad = open(bilgi, encoding="utf-8").readline().strip() if os.path.exists(bilgi) else "LeNet-5 benzeri CNN"
    m = MODEL_SINIFLARI[ad]()
    m.load_state_dict(torch.load(args.model, map_location="cpu"))
    p_py = tahmin_et(m, torch.from_numpy(py), "cpu")
    p_js = tahmin_et(m, torch.from_numpy(js), "cpu")
    sonuc = {
        "cizim_sayisi": len(y),
        "ortalama_mutlak_fark_gri_duzey": round(float(fark.mean()), 3),
        "en_buyuk_fark_gri_duzey": int(fark.max()),
        "ayni_piksel_orani_%": round(float((fark == 0).mean()) * 100, 2),
        "2_gri_duzeyden_az_farkli_piksel_%": round(float((fark <= 2).mean()) * 100, 2),
        "model": ad,
        "tahmin_ayni_%": round(float((p_py == p_js).mean()) * 100, 2),
        "python_dogruluk_%": round(float((p_py == y).mean()) * 100, 2),
        "js_dogruluk_%": round(float((p_js == y).mean()) * 100, 2),
        "js_on_isleme_ms_cizim_basina": round(float(sure), 2),
    }
    os.makedirs(CIKTI, exist_ok=True)
    with open(os.path.join(CIKTI, "on_isleme_testi.json"), "w", encoding="utf-8") as f:
        json.dump(sonuc, f, ensure_ascii=False, indent=2)
    print(json.dumps(sonuc, ensure_ascii=False, indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sec = np.argsort(-fark.reshape(len(y), -1).mean(1))[:1].tolist() + list(range(5))
    fig, axs = plt.subplots(3, 6, figsize=(6.4, 3.3))
    for j, i in enumerate(sec):
        for r, (g, baslik) in enumerate(((py[i], "Python"), (js[i], "JavaScript"), (fark[i], "Fark"))):
            a = axs[r, j]
            a.imshow(g, cmap="gray_r", vmin=0, vmax=255 if r < 2 else 64)
            a.set_xticks([]); a.set_yticks([])
            if j == 0:
                a.set_ylabel(baslik, fontsize=7.5)
        axs[0, j].set_title("en farklı" if j == 0 else f"örnek {j}", fontsize=7)
    fig.tight_layout(h_pad=0.3, w_pad=0.3)
    fig.savefig(os.path.join(CIKTI, "sekil_on_isleme_testi.png"), dpi=200); plt.close(fig)


if __name__ == "__main__":
    main()
