"""
HAFTA 5 - ADIM 2: Uygulama sayfasini (index.html) olustur

uygulama_sablon.html dosyasina 'onnx' klasorundeki modelleri (6_onnx_donusum.py'nin urettigi) ekler ve
index.html dosyasini yazar. Modeller sayfanin icine base64 metni olarak gomulur. Boylece:
  - sayfa tek bir dosyadir, cift tiklayarak acilabilir ya da GitHub Pages gibi bir yerde yayinlanabilir
  - model dosyalarini ayrica indirmek gerekmez, internet olmadan da (yedek hesaplayiciyla) calisir

Calistirma: python 7_uygulama_olustur.py
            python 7_uygulama_olustur.py --sadece-secilen   (Derin CNN eklenmez, sayfa kuculur)
"""
import argparse
import base64
import json
import os

SABLON = "uygulama_sablon.html"
CIKIS = "index.html"
KLASOR = "onnx"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sadece-secilen", action="store_true", help="yalnizca 4. haftada secilen modeli ekle")
    args = ap.parse_args()

    bilgi_yolu = os.path.join(KLASOR, "bilgi.json")
    if not os.path.exists(bilgi_yolu):
        raise SystemExit("onnx/bilgi.json bulunamadi. Once 6_onnx_donusum.py calistirilmali.")
    bilgi = json.load(open(bilgi_yolu, encoding="utf-8"))

    modeller = {}
    for dosya, b in bilgi.items():
        if args.sadece_secilen and not b.get("secilen"):
            continue
        with open(os.path.join(KLASOR, dosya), "rb") as f:
            modeller[dosya] = {"bilgi": b, "veri": base64.b64encode(f.read()).decode("ascii")}

    sablon = open(SABLON, encoding="utf-8").read()
    if "/*MODELLER*/" not in sablon:
        raise SystemExit("Sablonda /*MODELLER*/ isareti bulunamadi.")
    sayfa = sablon.replace("/*MODELLER*/", json.dumps(modeller, ensure_ascii=False, separators=(",", ":")))
    with open(CIKIS, "w", encoding="utf-8", newline="\n") as f:
        f.write(sayfa)

    print(f"{CIKIS} olusturuldu ({os.path.getsize(CIKIS) / 1024:.0f} KB). Eklenen modeller:")
    for dosya, m in modeller.items():
        b = m["bilgi"]
        print(f"  {dosya:20s} {b['model']}, {b['sayi_bicimi']}, {b['dosya_KB']} KB")
    print("Sayfa cift tiklanarak tarayicida acilabilir.")


if __name__ == "__main__":
    main()
