"""
HAFTA 6 - ADIM 2: Grad-CAM ile modelin cizimin hangi bolgesine bakarak karar verdigini gosterme (H6)

Grad-CAM (Selvaraju vd., 2017): son evrisim katmaninin cikisindaki her oznitelik haritasi, tahmin edilen sinifin
skorunun o haritaya gore turevinin ortalamasiyla agirliklandirilip toplanir. Negatif degerler atilir (ReLU) ve
sonuc 28x28'e buyutulur. Parlak bolgeler, skoru en cok artiran bolgelerdir.

Ne yapar?
  1) Uygulama modeli (LeNet-5 benzeri CNN) ve derin CNN icin Grad-CAM hesaplar
  2) Sekiller: her rakamdan dogru tahmin edilen ornekler ve toplanan verideki 9 hatalari
  3) Silme testi: Grad-CAM'in en onemli gordugu murekkep pikselleri silinince tahmin ne kadar bozuluyor?
     Ayni sayida rastgele murekkep pikseli silmekle karsilastirilir. Isi haritasi gercekten onemli bolgeyi
     gosteriyorsa birinci silme tahmini cok daha fazla bozmalidir.
  4) Uygulamadaki JavaScript Grad-CAM'inin testi icin ornek sonuclari kaydeder (testler/gradcam_ornekleri.npz)

Calistirma: python 10_gradcam.py
"""
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ortak import CIKTI, MODELLER, VERI, npz_yukle
from model_yapisi import MODEL_SINIFLARI

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})


def model_yukle(sinif, yol):
    m = MODEL_SINIFLARI[sinif]()
    m.load_state_dict(torch.load(yol, map_location="cpu")); m.eval()
    return m


def hedef_katman(model):
    """Son evrisim katmanindan sonraki ReLU (havuzlamadan once). LeNet: 16x10x10, derin CNN: 128x7x7."""
    son_relu = None
    for k in model.oznitelik:
        if isinstance(k, torch.nn.ReLU):
            son_relu = k
    return son_relu


def gradcam(model, x_u8, sinif=None, parti=512):
    """x_u8: (N, 28, 28) uint8. Donus: (cam 28x28 [0,1], tahmin, skorlar). sinif None ise tahmin edilen sinif."""
    katman = hedef_katman(model)
    camler, tahminler, skorlar = [], [], []
    for i in range(0, len(x_u8), parti):
        x = torch.from_numpy(x_u8[i:i + parti]).float().div(255).unsqueeze(1)
        kayit = {}

        def kanca(_m, _g, cikis):
            cikis.retain_grad(); kayit["a"] = cikis
        h = katman.register_forward_hook(kanca)
        skor = model(x); h.remove()
        c = skor.argmax(1) if sinif is None else torch.as_tensor(sinif[i:i + parti])
        model.zero_grad()
        skor.gather(1, c[:, None]).sum().backward()
        a, g = kayit["a"].detach(), kayit["a"].grad
        agirlik = g.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((agirlik * a).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=(28, 28), mode="bilinear", align_corners=False)[:, 0]
        en = cam.flatten(1).max(1)[0].clamp_min(1e-8).view(-1, 1, 1)
        camler.append((cam / en).numpy()); tahminler.append(c.numpy()); skorlar.append(skor.detach().numpy())
    return np.concatenate(camler), np.concatenate(tahminler), np.concatenate(skorlar)


def silme_testi(model, x_u8, cam, oran=0.2, tohum=0):
    """Her cizimde murekkep piksellerinin %oran'i silinir: (a) Grad-CAM'in en yuksek oldugu, (b) rastgele."""
    rng = np.random.default_rng(tohum)
    xa, xb = x_u8.copy(), x_u8.copy()
    for i in range(len(x_u8)):
        murekkep = np.flatnonzero(x_u8[i].ravel() > 30)
        k = max(1, int(round(len(murekkep) * oran)))
        sirali = murekkep[np.argsort(-cam[i].ravel()[murekkep])]
        xa[i].ravel()[sirali[:k]] = 0
        xb[i].ravel()[rng.choice(murekkep, k, replace=False)] = 0
    with torch.no_grad():
        f = lambda x: F.softmax(model(torch.from_numpy(x).float().div(255).unsqueeze(1)), 1).numpy()
        p0, pa, pb = f(x_u8), f(xa), f(xb)
    c = p0.argmax(1); n = np.arange(len(c))
    return {"ilk_dogruluk": p0.argmax(1), "gradcam_silme": pa, "rastgele_silme": pb, "sinif": c,
            "guven_ilk": p0[n, c], "guven_gradcam": pa[n, c], "guven_rastgele": pb[n, c]}


def bindir(ax, x, cam, baslik=None):
    ax.imshow(x, cmap="gray_r", vmin=0, vmax=255)
    ax.imshow(np.ma.masked_less(cam, 0.15), cmap="jet", alpha=0.5, vmin=0, vmax=1)
    ax.set_xticks([]); ax.set_yticks([])
    if baslik:
        ax.set_title(baslik, fontsize=7, pad=2)


def main():
    # uygulamadaki model: 9_rakam_degil.py calistirildiysa onun sectigi model, yoksa 4. haftada secilen model
    ad = "uygulama_modeli" if os.path.exists(os.path.join(MODELLER, "uygulama_modeli.pt")) else "secilen_model_hafta4"
    lenet = model_yukle(open(os.path.join(MODELLER, ad + ".txt"), encoding="utf-8").readline().strip(),
                        os.path.join(MODELLER, ad + ".pt"))
    derin_yol = os.path.join(MODELLER, "hafta4", "B_Derin_CNN_42.pt")
    derin = model_yukle("Derin CNN", derin_yol) if os.path.exists(derin_yol) else None

    d = npz_yukle("dokunmatik.npz")
    ayrim = pd.read_csv(os.path.join(CIKTI, "hafta4_dokunmatik_kisi_ayrimi.csv"), encoding="utf-8-sig")
    test = np.isin(d["kisi"], ayrim[ayrim.kume == "test"].kisi)
    xd, yd = d["x"][test], d["y"][test]
    t = np.load(os.path.join(VERI, "toplanan.npz"), allow_pickle=True)
    xt, yt = t["x"], t["y"]

    sonuc = {}
    # --- silme testi (dokunmatik test kumesinden 2.000 cizim)
    sec = np.random.default_rng(1).choice(len(xd), 2000, replace=False)
    for ad, m in (("LeNet-5 benzeri CNN", lenet), ("Derin CNN", derin)):
        if m is None:
            continue
        cam, p, _ = gradcam(m, xd[sec])
        s = silme_testi(m, xd[sec], cam)
        y = yd[sec]
        sonuc[ad] = {
            "cizim": len(sec), "silinen_murekkep_orani": 0.2,
            "dogruluk_ilk_%": round(float((s["ilk_dogruluk"] == y).mean()) * 100, 2),
            "dogruluk_gradcam_silme_%": round(float((s["gradcam_silme"].argmax(1) == y).mean()) * 100, 2),
            "dogruluk_rastgele_silme_%": round(float((s["rastgele_silme"].argmax(1) == y).mean()) * 100, 2),
            "guven_ilk": round(float(s["guven_ilk"].mean()), 3),
            "guven_gradcam_silme": round(float(s["guven_gradcam"].mean()), 3),
            "guven_rastgele_silme": round(float(s["guven_rastgele"].mean()), 3),
            "camin_murekkepte_kalan_orani": round(float(
                (cam * (xd[sec] > 30)).sum((1, 2)).sum() / cam.sum((1, 2)).sum()), 3),
        }

    # --- sekil 1: her rakamdan dogru tahmin edilen bir ornek
    cam_l, p_l, _ = gradcam(lenet, xd[:3000])
    secim = [int(np.flatnonzero((yd[:3000] == r) & (p_l == r))[0]) for r in range(10)]
    satir = 3 if derin is not None else 2
    fig, axs = plt.subplots(satir, 10, figsize=(6.4, 0.75 * satir + 0.35))
    if derin is not None:
        cam_d, _, _ = gradcam(derin, xd[secim])
    for j, i in enumerate(secim):
        axs[0, j].imshow(xd[i], cmap="gray_r"); axs[0, j].set_xticks([]); axs[0, j].set_yticks([])
        axs[0, j].set_title(str(yd[i]), fontsize=7, pad=2)
        bindir(axs[1, j], xd[i], cam_l[i])
        if derin is not None:
            bindir(axs[2, j], xd[i], cam_d[j])
    axs[0, 0].set_ylabel("Çizim", fontsize=7); axs[1, 0].set_ylabel("LeNet", fontsize=7)
    if derin is not None:
        axs[2, 0].set_ylabel("Derin CNN", fontsize=7)
    fig.tight_layout(h_pad=0.2, w_pad=0.2)
    fig.savefig(os.path.join(CIKTI, "sekil_hafta6_gradcam_rakamlar.png"), dpi=200); plt.close(fig)

    # --- sekil 2: toplanan verideki 9 hatalari (LeNet'in yanlis tahmini ve derin CNN'in dogru tahmini)
    cam_tl, p_tl, _ = gradcam(lenet, xt)
    hata9 = np.flatnonzero((yt == 9) & (p_tl != 9))[:8]
    if len(hata9):
        satir = 3 if derin is not None else 2
        fig, axs = plt.subplots(satir, len(hata9), figsize=(0.8 * len(hata9) + 0.4, 0.8 * satir + 0.3), squeeze=False)
        if derin is not None:
            cam_td, p_td, _ = gradcam(derin, xt[hata9])
        for j, i in enumerate(hata9):
            axs[0, j].imshow(xt[i], cmap="gray_r"); axs[0, j].set_xticks([]); axs[0, j].set_yticks([])
            bindir(axs[1, j], xt[i], cam_tl[i], f"LeNet: {p_tl[i]}")
            if derin is not None:
                bindir(axs[2, j], xt[i], cam_td[j], f"Derin: {p_td[j]}")
        axs[0, 0].set_ylabel("Çizim (9)", fontsize=7)
        fig.tight_layout(h_pad=0.4, w_pad=0.2)
        fig.savefig(os.path.join(CIKTI, "sekil_hafta6_gradcam_9_hatalari.png"), dpi=200); plt.close(fig)
    sonuc["toplanan_9_hatasi"] = int(len(np.flatnonzero((yt == 9) & (p_tl != 9))))

    # --- JavaScript testi icin ornekler (32 bit LeNet ile)
    os.makedirs("testler", exist_ok=True)
    ornek = np.concatenate([xd[:150], xt[::10]])
    cam_o, p_o, _ = gradcam(lenet, ornek)
    np.savez_compressed(os.path.join("testler", "gradcam_ornekleri.npz"), x=ornek, cam=cam_o.astype(np.float32), tahmin=p_o)

    with open(os.path.join(CIKTI, "hafta6_gradcam.json"), "w", encoding="utf-8") as f:
        json.dump(sonuc, f, ensure_ascii=False, indent=2)
    print(json.dumps(sonuc, ensure_ascii=False, indent=2))
    zip_yol = os.path.join(CIKTI, "hafta6_sonuclar.zip")       # 9_rakam_degil.py'nin olusturdugu zip'e eklenir
    if os.path.exists(zip_yol):
        import zipfile
        with zipfile.ZipFile(zip_yol, "a", zipfile.ZIP_DEFLATED) as z:
            mevcut = set(z.namelist())
            for f in ("hafta6_gradcam.json", "sekil_hafta6_gradcam_rakamlar.png", "sekil_hafta6_gradcam_9_hatalari.png"):
                if f not in mevcut:
                    z.write(os.path.join(CIKTI, f), f)
    print(f"Sekiller ve sonuclar '{CIKTI}' klasorune kaydedildi.")


if __name__ == "__main__":
    main()
