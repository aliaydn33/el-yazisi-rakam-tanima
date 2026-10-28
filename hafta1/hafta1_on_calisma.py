"""
Hafta 1 - On calisma
- MNIST veri setini indirir ve inceler (ornekler, sinif dagilimi)
- Tuvalden gelen cizimi MNIST formatina getiren on isleme adimlarini gosterir
- Iki basit referans (baseline) model egitir: Lojistik Regresyon ve MLP

Calistirma:  python hafta1_on_calisma.py
Gereksinimler: numpy, matplotlib, scikit-learn, pillow, scipy
"""
import gzip, os, time, pickle, urllib.request
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score

VERI_KLASORU = "data"
CIKTI_KLASORU = "cikti"
URL = "https://raw.githubusercontent.com/fgnt/mnist/master/"
DOSYALAR = ["train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
            "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})


def indir():
    os.makedirs(VERI_KLASORU, exist_ok=True)
    for f in DOSYALAR:
        yol = os.path.join(VERI_KLASORU, f)
        if not os.path.exists(yol):
            print("indiriliyor:", f)
            urllib.request.urlretrieve(URL + f, yol)


def goruntu_oku(yol):
    with gzip.open(yol, "rb") as f:
        return np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28, 28)


def etiket_oku(yol):
    with gzip.open(yol, "rb") as f:
        return np.frombuffer(f.read(), np.uint8, offset=8)


def mnist_formatina_getir(tuval):
    """Tuval goruntusunu (beyaz zemin, siyah murekkep) MNIST gibi 28x28'e cevirir.
    1) renkleri ters cevir  2) rakami cevreleyen kutuyu kirp
    3) en-boy oranini koruyarak 20x20'ye sigdir  4) agirlik merkezine gore 28x28'de ortala"""
    g = 255 - np.asarray(tuval.convert("L"), dtype=np.float32)
    satir = np.where(g.max(axis=1) > 30)[0]
    sutun = np.where(g.max(axis=0) > 30)[0]
    kirpik = g[satir[0]:satir[-1] + 1, sutun[0]:sutun[-1] + 1]
    h, w = kirpik.shape
    olcek = 20.0 / max(h, w)
    yeni = Image.fromarray(kirpik.astype(np.uint8)).resize(
        (max(1, round(w * olcek)), max(1, round(h * olcek))), Image.LANCZOS)
    kutu = np.zeros((28, 28), np.float32)
    yh, yw = yeni.size[1], yeni.size[0]
    kutu[(28 - yh) // 2:(28 - yh) // 2 + yh, (28 - yw) // 2:(28 - yw) // 2 + yw] = np.asarray(yeni)
    cy, cx = ndimage.center_of_mass(kutu)
    # kaydirma miktarini rakam 28x28 disina tasmayacak sekilde sinirla
    ust, sol = (28 - yh) // 2, (28 - yw) // 2
    dy = float(np.clip(14 - cy, -ust, 28 - (ust + yh)))
    dx = float(np.clip(14 - cx, -sol, 28 - (sol + yw)))
    kutu = ndimage.shift(kutu, (dy, dx), order=1, mode="constant")
    return g, kirpik, np.clip(kutu, 0, 255)


def kare_yap(a):
    """Sadece gosterim icin: dikdortgen goruntuyu kenarlarina bosluk ekleyerek kareye tamamlar."""
    h, w = a.shape; n = max(h, w)
    k = np.zeros((n, n), a.dtype)
    k[(n - h) // 2:(n - h) // 2 + h, (n - w) // 2:(n - w) // 2 + w] = a
    return k


def ornek_cizim():
    """Kullanicinin tuvalin sol ust kosesine kalin firca ile '7' cizdigini canlandirir."""
    img = Image.new("L", (280, 280), 255)
    d = ImageDraw.Draw(img)
    d.line([(40, 45), (150, 40), (95, 170)], fill=0, width=18, joint="curve")
    d.line([(70, 105), (135, 102)], fill=0, width=16)
    return img


def main():
    indir()
    Xtr = goruntu_oku(os.path.join(VERI_KLASORU, DOSYALAR[0]))
    ytr = etiket_oku(os.path.join(VERI_KLASORU, DOSYALAR[1]))
    Xte = goruntu_oku(os.path.join(VERI_KLASORU, DOSYALAR[2]))
    yte = etiket_oku(os.path.join(VERI_KLASORU, DOSYALAR[3]))
    os.makedirs(CIKTI_KLASORU, exist_ok=True)

    print("Egitim:", Xtr.shape, "Test:", Xte.shape)
    dagilim_tr = np.bincount(ytr, minlength=10)
    dagilim_te = np.bincount(yte, minlength=10)
    print("Egitim sinif dagilimi:", dagilim_tr.tolist())
    print("Test sinif dagilimi:", dagilim_te.tolist())
    print("Ortalama piksel degeri: %.2f" % Xtr.mean(), " sifir piksel orani: %.3f" % (Xtr == 0).mean())

    # --- Sekil: ornekler + sinif dagilimi
    rng = np.random.default_rng(7)
    fig = plt.figure(figsize=(7.6, 3.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1], wspace=0.28)
    sol = gs[0].subgridspec(10, 10, wspace=0.04, hspace=0.04)
    for r in range(10):
        idx = rng.choice(np.where(ytr == r)[0], 10, replace=False)
        for c in range(10):
            ax = fig.add_subplot(sol[r, c])
            ax.imshow(Xtr[idx[c]], cmap="gray_r")
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
    fig.text(0.29, -0.06, "(a) Her sınıftan rastgele 10 örnek", ha="center")
    ax = fig.add_subplot(gs[1])
    ax.bar(range(10), dagilim_tr, color="#4C72B0", width=0.7)
    for i, v in enumerate(dagilim_tr):
        ax.text(i, v + 60, str(v), ha="center", va="bottom", fontsize=7.5, rotation=90)
    ax.set_xticks(range(10)); ax.set_xlabel("Rakam (sınıf)"); ax.set_ylabel("Örnek sayısı")
    ax.set_ylim(0, 7900); ax.spines[["top", "right"]].set_visible(False)
    fig.text(0.735, -0.06, "(b) Eğitim kümesi sınıf dağılımı (n = 60.000)", ha="center")
    fig.savefig(os.path.join(CIKTI_KLASORU, "sekil2_veri_seti.png"), dpi=220, bbox_inches="tight")
    plt.close(fig)

    # --- Sekil: on isleme adimlari
    tuval = ornek_cizim()
    ters, kirpik, son = mnist_formatina_getir(tuval)
    kaba = 255 - np.asarray(tuval.resize((28, 28), Image.BILINEAR), dtype=np.float32)
    fig, axs = plt.subplots(1, 5, figsize=(7.6, 2.05))
    paneller = [(np.asarray(tuval), "gray", "(a) Tuval 280×280"),
                (kaba, "gray_r", "(b) Doğrudan 28×28\n(hatalı yaklaşım)"),
                (kare_yap(kirpik), "gray_r", "(c) Ters çevirme\nve kırpma"),
                (son, "gray_r", "(d) 20×20 ölçek +\nağırlık merkezi 28×28"),
                (Xtr[np.where(ytr == 7)[0][3]], "gray_r", "(e) MNIST'ten\ngerçek bir '7'")]
    for ax, (im, cm, t) in zip(axs, paneller):
        ax.imshow(im, cmap=cm, vmin=0, vmax=255)
        ax.set_title(t, fontsize=8.5); ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(os.path.join(CIKTI_KLASORU, "sekil3_on_isleme.png"), dpi=220, bbox_inches="tight")
    plt.close(fig)

    # --- Referans modeller
    Xtr_f = Xtr.reshape(len(Xtr), -1).astype(np.float32) / 255.0
    Xte_f = Xte.reshape(len(Xte), -1).astype(np.float32) / 255.0
    modeller = {
        "Lojistik Regresyon": LogisticRegression(max_iter=200, C=0.1),
        "MLP (784-128-10)": MLPClassifier(hidden_layer_sizes=(128,), max_iter=15,
                                           batch_size=128, random_state=0),
    }
    sonuclar = []
    for ad, m in modeller.items():
        t0 = time.time(); m.fit(Xtr_f, ytr); egitim = time.time() - t0
        t0 = time.time(); p = m.predict(Xte_f); tahmin_ms = (time.time() - t0) / len(Xte_f) * 1000
        acc = accuracy_score(yte, p); f1 = f1_score(yte, p, average="macro")
        boyut_kb = len(pickle.dumps(m)) / 1024
        # on islenmis ve kaba cizim icin tahmin
        p_kaba = m.predict((kaba.reshape(1, -1) / 255.0).astype(np.float32))[0]
        p_son = m.predict((son.reshape(1, -1) / 255.0).astype(np.float32))[0]
        sonuclar.append((ad, acc, f1, egitim, tahmin_ms, boyut_kb, p_kaba, p_son))
        print(f"{ad}: dogruluk={acc:.4f} makroF1={f1:.4f} egitim={egitim:.1f}s "
              f"tahmin={tahmin_ms:.4f}ms/ornek boyut={boyut_kb:.0f}KB  "
              f"cizim(kaba)->{p_kaba}  cizim(on islenmis)->{p_son}")


if __name__ == "__main__":
    main()
