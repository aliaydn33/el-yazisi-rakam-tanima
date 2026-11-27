"""
HAFTA 8: Kullanici testinin analizi (H7) ve yeni kisilerle model karari

Kullanici testi uygulamanin "?test" adresiyle yapilir (uygulama_sablon.html, "kullanici testi" bolumu). Her
katilimcinin sonucu Google Drive'daki rakam_verisi klasorune kullanici_testi_<no>_<zaman>.json olarak yazilir.

Calistirma:
  python 12_kullanici_testi.py --aday
        Testten ONCE bir kez calistirilir. Ince ayar adayini hazirlar: uygulama modeli, ekibin topladigi verideki
        20 kisinin modelin yanlis tahmin ettigi ya da reddettigi cizimleriyle 7. haftadaki ayarlarla ince ayar
        yapilir (tohum 42). 8 bit ONNX dosyasi onnx/ince_ayar_aday_int8.onnx olarak yazilir (bilgi.json'a
        eklenmedigi icin uygulamaya girmez). Aday, test verisi gelmeden
        hazirlandigi icin secimde test verisi kullanilmamis olur.
  python 12_kullanici_testi.py rakam_verisi.zip      (ya da klasor / .json dosyalari)
        Drive'dan indirilen klasorun icindeki kullanici_testi_*.json dosyalarini okur (veri toplama dosyalari
        atlanir) ve analiz eder.

Analiz (kurallar test verisi gelmeden yazildi):
  - SUS puani (Brooke, 1996): tek numarali maddeler (cevap - 1), cift numarali maddeler (5 - cevap), toplam x 2,5.
    H7 hedefi: en az 15 katilimci ve ortalama SUS >= 70. Ortalamanin %95 guven araligi (t dagilimi) de verilir.
  - Gorevler: rakam dogrulugu, sayi ve islem okuma (E1), rakam olmayan cizimlerin reddi (H5), "Acikla"
    dugmesinin bulunmasi (H6), fotograftan okuma (E2), duzeltme ozelliginin kullanimi (IG-09) ve gorev sureleri.
  - Cihazlar (IOG-07): isletim sistemi, tarayici ve cihaz turu. Hiz (IOG-01, H3): her cihazda olculen 100
    tekrarin medyani. Ilk acilis (IOG-02): sayfanin service worker olmadan (ilk kez) acildigi katilimcilarda.
  - Model karari: daha once veri toplamaya katilmamis katilimcilarin 20'ser rakam cizimiyle uygulama modeli
    (LeNet-5 benzeri CNN, 8 bit), derin CNN (8 bit) ve ince ayar adayi (8 bit) karsilastirilir. Aday, ancak
    dogrulugu uygulama modelinden yuksekse ve fark katilimci duzeyinde isaret degistirme testine gore anlamliysa
    (iki karsilastirma icin Holm duzeltmesi, alfa 0,05) uygulamaya alinir. Ikisi de saglarsa dogrulugu yuksek
    olan, esitse kucuk olan secilir. Derin CNN icin ayrica katilimcilarin telefonlarinda olculen model suresi
    H3 sinirini (50 ms) gecmemelidir.
"""
import argparse
import glob
import hashlib
import importlib.util
import io
import json
import os
import zipfile

import numpy as np
import pandas as pd
from scipy import stats

from ortak import CIKTI, MODELLER, VERI

ADAY = os.path.join(MODELLER, "ince_ayar_aday")
ADAY_ONNX = os.path.join("onnx", "ince_ayar_aday_int8.onnx")    # depoya eklenir, uygulamaya eklenmez (bilgi.json'da yok)
SUS_HEDEF, EN_AZ_KATILIMCI = 70.0, 15


def modul(ad):
    spec = importlib.util.spec_from_file_location(ad.replace(".py", "").lstrip("0123456789_"), os.path.join(os.path.dirname(os.path.abspath(__file__)), ad))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


# ------------------------------------------------------------------------------------------ ince ayar adayi
def aday_hazirla():
    import torch
    import torch.nn.functional as F
    from model_yapisi import kayan_nokta
    ia = modul("11_ince_ayar.py"); ox = modul("6_onnx_donusum.py")
    x_q, y_q, x_v, y_v, x_m, y_m = ia.veriler()
    t = np.load(os.path.join(VERI, "toplanan.npz"), allow_pickle=True)
    x_t, y_t = t["x"], t["y"].astype(np.int64)
    temel = ia.model_yukle()
    esik = json.load(open(os.path.join("onnx", "bilgi.json"), encoding="utf-8"))["secilen.onnx"]["rakam_degil"]["esik"]
    with torch.no_grad():
        p = F.softmax(temel(kayan_nokta(torch.from_numpy(x_t))), 1).numpy()
    secim = (p.argmax(1) != y_t) | (p.max(1) < esik)
    m = ia.ince_ayar(ia.model_yukle(), x_t[secim], y_t[secim], x_q, y_q, 42)
    torch.save(m.state_dict(), ADAY + ".pt")
    q = np.load(os.path.join(VERI, "qmnist.npz"))
    kal = q["x_train"][np.random.default_rng(0).choice(len(q["x_train"]), 512, replace=False)].astype(np.float32)[:, None] / 255.0
    ox.onnx_yaz(m.eval(), ADAY + ".onnx"); ox.int8_yaz(ADAY + ".onnx", ADAY_ONNX, kal)
    bilgi = {"egitimde_kullanilan_cizim": int(secim.sum()), "kisi": int(len(set(t["katilimci"]))), "tohum": 42,
             "dokunmatik_dogrulama_%": {"once": round(ia.dogruluk(temel, x_v, y_v), 2), "sonra": round(ia.dogruluk(m, x_v, y_v), 2)},
             "mnist_test_%": {"once": round(ia.dogruluk(temel, x_m, y_m), 2), "sonra": round(ia.dogruluk(m, x_m, y_m), 2)},
             "dosya_KB": round(os.path.getsize(ADAY_ONNX) / 1024, 1),
             "sha256_int8": hashlib.sha256(open(ADAY_ONNX, "rb").read()).hexdigest()}
    with open(os.path.join(CIKTI, "hafta8_ince_ayar_adayi.json"), "w", encoding="utf-8") as f:
        json.dump(bilgi, f, ensure_ascii=False, indent=2)
    print(json.dumps(bilgi, ensure_ascii=False, indent=2))


# ------------------------------------------------------------------------------------------ okuma
def kayitlari_oku(yollar):
    metinler = []
    for y in yollar:
        if y.endswith(".zip"):
            with zipfile.ZipFile(y) as z:
                metinler += [z.read(a).decode("utf-8") for a in z.namelist() if os.path.basename(a).startswith("kullanici_testi_") and a.endswith(".json")]
        elif os.path.isdir(y):
            metinler += [open(d, encoding="utf-8").read() for d in sorted(glob.glob(os.path.join(y, "**", "kullanici_testi_*.json"), recursive=True))]
        else:
            metinler.append(open(y, encoding="utf-8").read())
    kayitlar = {}
    for m in metinler:
        k = json.loads(m)
        if k.get("tur") != "kullanici_testi" or not k.get("sus") or None in k["sus"]:
            continue
        if k["katilimci_no"] not in kayitlar or k["bitis"] > kayitlar[k["katilimci_no"]]["bitis"]:
            kayitlar[k["katilimci_no"]] = k
    return [kayitlar[n] for n in sorted(kayitlar, key=lambda n: kayitlar[n]["baslama"])]


def sus_puani(c):
    return sum((c[i] - 1) if i % 2 == 0 else (5 - c[i]) for i in range(10)) * 2.5


def ga_t(x):
    x = np.asarray(x, float); n = len(x); s = x.std(ddof=1) if n > 1 else 0.0
    h = stats.t.ppf(0.975, n - 1) * s / np.sqrt(n) if n > 1 else float("nan")
    return round(x.mean(), 1), round(s, 1), round(x.mean() - h, 1), round(x.mean() + h, 1)


def clopper(k, n):
    if n == 0:
        return (float("nan"), float("nan"))
    a = stats.beta.ppf(0.025, k, n - k + 1) if k > 0 else 0.0
    b = stats.beta.ppf(0.975, k + 1, n - k) if k < n else 1.0
    return round(a * 100, 1), round(b * 100, 1)


def oran(k, n):
    return {"dogru": int(k), "toplam": int(n), "%": round(100 * k / n, 1) if n else None, "ga95": clopper(k, n)}


# ------------------------------------------------------------------------------------------ model karari
def model_karari(kayitlar, rng):
    import onnxruntime as ort
    ist = modul("13_istatistiksel_karsilastirma.py")
    yeni = [k for k in kayitlar if k.get("onceki_katilim") == "hayir"]
    x, y, kume = [], [], []
    for k in yeni:
        for o in k["ornekler"]:
            x.append(o["goruntu"]); y.append(int(o["hedef"])); kume.append(k["katilimci_no"])
    if not x:
        return None
    x = np.array(x, np.float32).reshape(-1, 1, 28, 28) / 255.0; y = np.array(y); kume = np.array(kume)
    modeller = {"LeNet-5 benzeri CNN (uygulama)": os.path.join("onnx", "secilen_int8.onnx"),
                "Derin CNN": os.path.join("onnx", "derin_int8.onnx"), "İnce ayarlı LeNet": ADAY_ONNX}
    dogru = {}
    for ad, yol in modeller.items():
        if not os.path.exists(yol):
            continue
        o = ort.InferenceSession(yol, providers=["CPUExecutionProvider"])
        dogru[ad] = o.run(None, {"giris": x})[0].argmax(1) == y
    uyg = "LeNet-5 benzeri CNN (uygulama)"
    kayit_ile_ayni = float(np.mean([str(t) == str(p) for t, p in zip(
        [o["tahmin"] for k in yeni for o in k["ornekler"]], np.argmax(ort.InferenceSession(modeller[uyg], providers=["CPUExecutionProvider"]).run(None, {"giris": x})[0], 1))]))
    satirlar = []
    for ad, d in dogru.items():
        a, n = ist.kume_toplamlari(d.astype(float), kume)
        bs = ist.bootstrap(a, None, n, rng)
        satir = {"model": ad, "dogruluk_%": round(a.sum() / n.sum() * 100, 2), "ga95": [round(np.percentile(bs, 2.5), 2), round(np.percentile(bs, 97.5), 2)]}
        if ad != uyg:
            satir.update(ist.karsilastir(d.astype(float), dogru[uyg].astype(float), kume, rng, d, dogru[uyg]))
        satirlar.append(satir)
    adaylar = [s for s in satirlar if s["model"] != uyg]
    for s, p in zip(adaylar, ist.holm([s["p_kisi_duzeyi"] for s in adaylar])):
        s["p_holm"] = p
    return {"katilimci": len(yeni), "cizim": int(len(y)), "uygulamadaki_tahminle_ayni_%": round(kayit_ile_ayni * 100, 2), "modeller": satirlar}


# ------------------------------------------------------------------------------------------ analiz
def analiz(kayitlar):
    rng = np.random.default_rng(2026)
    satir = []
    for k in kayitlar:
        o, s, rd = k["ornekler"], k["sayilar"], k["rakam_degil"]
        f, c, h = k.get("fotograf") or {}, k["cihaz"], k.get("hiz") or {}
        satir.append({
            "katilimci": k["katilimci_no"], "onceki_katilim": k.get("onceki_katilim"), "cihaz_turu": c["cihaz_turu"],
            "isletim_sistemi": c["isletim_sistemi"], "tarayici": c["tarayici"], "uygulama_ici": c.get("uygulama_ici"),
            "kurulu": c.get("kurulu"), "ekran_genisligi": c["ekran"][0], "giris": "+".join(k.get("giris_turleri") or []),
            "sure_dk": round(k["sure_sn"] / 60, 1), "sus": sus_puani(k["sus"]),
            "rakam_dogru": sum(str(x["tahmin"]) == x["hedef"] for x in o), "rakam_sayisi": len(o),
            "rakam_reddedilen": sum(bool(x["rakam_degil"]) for x in o),
            "yanlis_tahmin": sum(str(x["tahmin"]) != x["hedef"] for x in o),
            "duzeltilen": sum(x.get("duzeltme") is not None for x in o),
            "sayi_dogru": sum(bool(x["dogru"]) for x in s), "sayi_sayisi": len(s),
            "rakam_degil_red": sum(bool(x["rakam_degil"]) for x in rd), "rakam_degil_sayisi": len(rd),
            "acikla_bulundu": bool((k.get("acikla") or {}).get("bulundu")),
            "foto_yapildi": bool(f) and not f.get("atlandi"), "foto_dogru": bool(f.get("dogru")) if f and not f.get("atlandi") else None,
            "model_ms": (h.get("uygulama_modeli_ms") or {}).get("medyan"), "on_isleme_ms": (h.get("on_isleme_ms") or {}).get("medyan"),
            "derin_ms": (h.get("derin_ms") or {}).get("medyan"), "hesaplayan": h.get("hesaplayan"),
            "ilk_acilis": not (k.get("acilis") or {}).get("service_worker_ile", True),
            "acilis_ms": (k.get("acilis") or {}).get("uygulama_hazir_ms"), "ort_hazir_ms": (k.get("acilis") or {}).get("ort_hazir_ms"),
            "baglanti": (k.get("acilis") or {}).get("baglanti"),
            "rakam_gorev_sn": round(np.median([x["sure_ms"] for x in o]) / 1000, 1),
            "sayi_gorev_sn": round(np.median([x["sure_ms"] for x in s]) / 1000, 1) if s else None,
            "yorum": k.get("yorum", ""),
        })
    t = pd.DataFrame(satir)
    n = len(t)
    sus = ga_t(t.sus)
    maddeler = np.array([k["sus"] for k in kayitlar], float)
    katki = np.where(np.arange(10) % 2 == 0, maddeler - 1, 5 - maddeler)        # 0-4, yuksek = olumlu
    tel = t[t.cihaz_turu == "telefon"]
    ilk = t[t.ilk_acilis & t.acilis_ms.notna()]
    ozet = {
        "katilimci": n, "daha_once_katilmamis": int((t.onceki_katilim == "hayir").sum()),
        "sus": {"ortalama": sus[0], "ss": sus[1], "ga95": [sus[2], sus[3]], "medyan": float(t.sus.median()),
                "en_dusuk": float(t.sus.min()), "en_yuksek": float(t.sus.max()), "70_ve_ustu": int((t.sus >= 70).sum()),
                "tek_yonlu_t_p_70_ustu": round(float(stats.ttest_1samp(t.sus, SUS_HEDEF, alternative="greater").pvalue), 4) if n > 1 else None,
                "madde_katkisi_ortalama_0_4": [round(v, 2) for v in katki.mean(0)],
                "cronbach_alfa": round(float(10 / 9 * (1 - katki.var(0, ddof=1).sum() / katki.sum(1).var(ddof=1))), 2) if n > 2 else None,
                "H7_saglandi": bool(n >= EN_AZ_KATILIMCI and sus[0] >= SUS_HEDEF)},
        "gorevler": {
            "rakam": oran(t.rakam_dogru.sum(), t.rakam_sayisi.sum()),
            "rakam_yanlis_ret": oran(t.rakam_reddedilen.sum(), t.rakam_sayisi.sum()),
            "sayi_islem": oran(t.sayi_dogru.sum(), t.sayi_sayisi.sum()),
            "rakam_degil_red": oran(t.rakam_degil_red.sum(), t.rakam_degil_sayisi.sum()),
            "acikla_bulundu": oran(t.acikla_bulundu.sum(), n),
            "fotograf_yapan": int(t.foto_yapildi.sum()), "fotograf_dogru": oran(t.foto_dogru.fillna(False).sum(), t.foto_yapildi.sum()),
            "yanlis_tahmin": int(t.yanlis_tahmin.sum()), "duzeltilen": int(t.duzeltilen.sum()),
            "test_suresi_dk_medyan": float(t.sure_dk.median()),
            "rakam_basina_sn_medyan": float(t.rakam_gorev_sn.median()),
        },
        "cihazlar": {"cihaz_turu": t.cihaz_turu.value_counts().to_dict(), "isletim_sistemi": t.isletim_sistemi.value_counts().to_dict(),
                     "tarayici": t.tarayici.value_counts().to_dict(), "uygulama_ici_tarayici": int(t.uygulama_ici.fillna(False).sum()),
                     "giris": t.giris.value_counts().to_dict(), "en_dar_ekran": int(t.ekran_genisligi.min())},
        "hiz": {"telefon_sayisi": len(tel), "telefonda_model_ms_medyan_en_yavas": float(tel.model_ms.max()) if len(tel) else None,
                "telefonda_on_isleme_ms_en_yavas": float(tel.on_isleme_ms.max()) if len(tel) else None,
                "telefonda_toplam_ms_en_yavas": float((tel.model_ms + tel.on_isleme_ms).max()) if len(tel) else None,
                "telefonda_derin_ms_en_yavas": float(tel.derin_ms.max()) if len(tel) and tel.derin_ms.notna().any() else None,
                "yedek_hesaplayici_kullanan": int((t.hesaplayan == "js").sum()),
                "H3_saglandi": bool(len(tel) and (tel.model_ms + tel.on_isleme_ms).max() <= 50)},
        "ilk_acilis": {"olculen": len(ilk), "medyan_ms": float(ilk.acilis_ms.median()) if len(ilk) else None,
                       "en_yavas_ms": float(ilk.acilis_ms.max()) if len(ilk) else None,
                       "ort_hazir_medyan_ms": float(ilk.ort_hazir_ms.median()) if len(ilk) and ilk.ort_hazir_ms.notna().any() else None,
                       "3_sn_alti": int((ilk.acilis_ms <= 3000).sum())},
        "yorumlar": [y for y in t.yorum if y.strip()],
    }
    ozet["model_karari"] = model_karari(kayitlar, rng)
    karar = ozet["model_karari"]
    if karar:
        uygun = [m for m in karar["modeller"] if m.get("p_holm", 1) < 0.05 and m["fark_puan"] > 0]
        if any(m["model"] == "Derin CNN" for m in uygun) and ozet["hiz"]["telefonda_derin_ms_en_yavas"] and ozet["hiz"]["telefonda_derin_ms_en_yavas"] > 50:
            uygun = [m for m in uygun if m["model"] != "Derin CNN"]
        karar["secilen"] = max(uygun, key=lambda m: m["dogruluk_%"])["model"] if uygun else "LeNet-5 benzeri CNN (uygulama)"
    return t, ozet


def sekil(t, ozet, yol):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5})
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.2, 2.9), gridspec_kw={"width_ratios": [1.25, 1]})
    s = t.sort_values("sus").reset_index(drop=True)
    renk = {"telefon": "#2F4B7C", "tablet": "#55A868", "bilgisayar": "#DD8452"}
    a1.bar(range(len(s)), s.sus, color=[renk.get(c, "#8C8C8C") for c in s.cihaz_turu])
    a1.axhline(SUS_HEDEF, color="#C44E52", ls="--", lw=1, label="Hedef (70)")
    o = ozet["sus"]
    a1.axhline(o["ortalama"], color="#1C2430", lw=1, label=f"Ortalama ({o['ortalama']:.1f})".replace(".", ","))
    a1.axhspan(o["ga95"][0], o["ga95"][1], color="#1C2430", alpha=0.08, label="%95 güven aralığı")
    for c, r in renk.items():
        if (t.cihaz_turu == c).any():
            a1.bar([0], [0], color=r, label=c.capitalize())
    a1.set_xticks([]); a1.set_xlabel("Katılımcılar (SUS puanına göre sıralı)"); a1.set_ylabel("SUS puanı"); a1.set_ylim(0, 100)
    a1.legend(fontsize=7, loc="lower right", ncol=2)
    m = ozet["sus"]["madde_katkisi_ortalama_0_4"]
    a2.barh(range(10), m, color=["#2F4B7C" if i % 2 == 0 else "#8172B3" for i in range(10)])
    a2.set_yticks(range(10)); a2.set_yticklabels([f"{i + 1}. madde" for i in range(10)]); a2.invert_yaxis()
    a2.set_xlim(0, 4); a2.set_xlabel("Maddenin ortalama katkısı (0-4, yüksek = olumlu)")
    fig.tight_layout(); fig.savefig(yol, dpi=200); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kaynak", nargs="*", help="Drive'dan indirilen zip, klasor ya da kullanici_testi_*.json dosyalari")
    ap.add_argument("--aday", action="store_true", help="ince ayar adayini hazirla (testten once bir kez)")
    a = ap.parse_args()
    if a.aday:
        return aday_hazirla()
    if not a.kaynak:
        raise SystemExit("Kullanim: python 12_kullanici_testi.py rakam_verisi.zip   ya da   --aday")
    kayitlar = kayitlari_oku(a.kaynak)
    if not kayitlar:
        raise SystemExit("kullanici_testi_*.json bulunamadi.")
    t, ozet = analiz(kayitlar)
    t.drop(columns=["yorum"]).to_csv(os.path.join(CIKTI, "hafta8_katilimcilar.csv"), index=False, encoding="utf-8-sig")
    with open(os.path.join(CIKTI, "hafta8_kullanici_testi.json"), "w", encoding="utf-8") as f:
        json.dump(ozet, f, ensure_ascii=False, indent=2, default=lambda v: v.item() if hasattr(v, "item") else str(v))
    sekil(t, ozet, os.path.join(CIKTI, "sekil_hafta8_sus.png"))
    print(json.dumps(ozet, ensure_ascii=False, indent=2, default=lambda v: v.item() if hasattr(v, "item") else str(v)))


if __name__ == "__main__":
    main()
