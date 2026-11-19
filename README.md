# El Yazısı Rakamları Tanıyan Etkileşimli Uygulama

Fırat Üniversitesi Bilgisayar Mühendisliği, Bilgisayar Mühendisliği Tasarımı dersi projesi. **41.GRUP:** Eray Yolcu, Ali Aydın, Metehan Borlukçu.

Ekrana parmakla, kalemle ya da fareyle çizilen rakamları tarayıcıda gerçek zamanlı olarak tanıyan bir web uygulaması geliştiriyoruz. Projenin temel mühendislik problemi şu: Taranmış kâğıt görüntüleriyle (MNIST) eğitilen modeller, ekrana çizilen rakamlarda aynı başarıyı gösteremiyor. Literatürde buna veri dağılımı kayması deniyor.

**Uygulama:** https://aliaydn33.github.io/el-yazisi-rakam-tanima/ (çizdiğiniz rakamı tarayıcıda tanır, sunucuya bir şey gönderilmez)

## Durum (7. hafta sonu)

| Hafta | İş | Durum |
|---|---|---|
| 1 | Problem tanımı, literatür, hedefler, plan | Tamamlandı |
| 2 | Veri setleri (QMNIST, EMNIST-Digits, dokunmatik ekran verisi), ön işleme, ilk CNN | Tamamlandı |
| 3 | Veri artırma, veri toplama sayfası | Tamamlandı |
| 4 | Model karşılaştırması (5 tohum, istatistiksel testler), EMNIST deneyleri, çizim ekranı | Tamamlandı |
| 5 | Modelin tarayıcıya taşınması (ONNX), canlı tahmin, uygulamanın ilk sürümü, Drive'a kaydeden veri toplama | Tamamlandı |
| 6 | "Rakam değil" uyarısı, Grad-CAM | Tamamlandı |
| 7 | FastAPI sunucu, telefona kurulum (PWA), sayı ve işlem okuma, düzeltme kaydı, fotoğraftan okuma | Tamamlandı |
| 8 | Kullanıcı testi (SUS), istatistiksel karşılaştırma | Planlandı |
| 9 | Final, demo | Planlandı |

**Şu anki uygulama modeli:** LeNet-5 benzeri CNN (61.706 parametre, veri artırma yok, QMNIST ile eğitildi). Uygulamada 8 bit (int8) ONNX sürümü kullanılıyor: 77 KB, dokunmatik testte %97,47 (32 bit sürümde %97,48).

**Ekibin topladığı veri (20 katılımcı, 1.000 çizim, eğitimde kullanılmadı):** uygulama modeli %96,0 (%95 güven aralığı 94,6-97,1), derin CNN %97,6 (96,5-98,5). Uygulama modelinin hatalarının yarısı alt ucu sola kıvrılan 9'larda.

| Ölçüt (5 tohum ortalaması) | LeNet-5 benzeri CNN | Derin CNN | MLP |
|---|---|---|---|
| MNIST test | %98,97 ± 0,07 | %99,21 ± 0,09 | %97,54 ± 0,19 |
| Dokunmatik test (202 kişi, eğitimde ve seçimde kullanılmadı) | %97,24 ± 0,31 | %97,67 ± 0,43 | %93,98 ± 0,76 |
| Model dosyası | 245 KB | 956 KB | 400 KB |

## Klasör yapısı

```
1_veri_hazirla.py         2. hafta: veri setlerini indirir, hazırlar ve inceler
2_ilk_cnn.py              2. hafta: MLP ve ilk CNN
3_emnist_artirma.py       3. hafta: EMNIST ve veri artırma deneyleri (D1-D4)
4_toplanan_veri.py        Veri toplama sayfasından gelen dosyaları işler, modelleri bu veride dener
5_model_karsilastirma.py  4. hafta: veri artırma yöntemleri, 3 model, veri kaynağı (5 tohum)
6_onnx_donusum.py         5. hafta: modelleri ONNX'e ve 8 bite çevirir, PyTorch ile karşılaştırır
7_uygulama_olustur.py     5. hafta: ONNX modellerini uygulama sayfasına ekleyip index.html'i üretir
8_h2_olcumu.py            5. hafta: ekibin topladığı veride H2 ölçümü (güven aralığı, katılımcı bazında sonuçlar)
9_rakam_degil.py          6. hafta: "rakam değil" uyarısı (EMNIST harfleri ve Quick, Draw! ile)
10_gradcam.py             6. hafta: Grad-CAM ısı haritaları
sunucu.py                 7. hafta: FastAPI sunucusu (fotoğraftan okuma, düzeltme kaydı, uygulamanın yayınlanması)
fotograf_okuma.py         7. hafta: kâğıttaki rakamların fotoğraftan okunması (OpenCV)
model_yapisi.py           Modeller (MLP, LeNet-5 benzeri CNN, Derin CNN), eğitim ve veri artırma
ortak.py                  Ön işleme (MNIST biçimine getirme, vuruşları görüntüye çevirme)
index.html                Uygulama: çizim, canlı tahmin, 10 rakamın olasılıkları, hız testi
uygulama_sablon.html      index.html'in kaynağı (modeller eklenmeden önceki hali)
manifest.webmanifest, sw.js, simgeler/   Telefona kurulum ve internetsiz açılma (PWA)
onnx/                     Uygulamada kullanılan ONNX modelleri
veri_toplama.html         Veri toplama sayfası (aydınlatma metni, 50 rakam)
apps_script/              Veri toplama sayfasının Google Apps Script sürümü (çizimler Drive'a kaydedilir)
cizim.html                4. haftanın çizim ekranı (tahmin olmadan)
testler/                  Ön işleme ve uygulama testleri (tarayıcıdaki sonuçların Python ile karşılaştırılması)
toplanan_veri/            Toplanan anonim çizimler (JSON, katılımcı başına bir dosya)
cikti/                    Raporlarda kullanılan tablolar, şekiller ve özetler
hafta1/                   1. haftanın ön çalışma kodu
KURULUM*.md               Her haftanın kurulum ve çalıştırma notları
```

`veri/` (yaklaşık 3 GB) ve `modeller/` klasörleri depoya yüklenmez. `1_veri_hazirla.py` çalıştırılınca veriler yeniden oluşur, eğitim betikleri çalıştırılınca da modeller.

## Kurulum ve çalıştırma

Python 3.10 veya daha yeni bir sürüm gerekir. Ekran kartı gerekmez.

```
python -m pip install -r requirements.txt
python 1_veri_hazirla.py
python 2_ilk_cnn.py
python 3_emnist_artirma.py
python 5_model_karsilastirma.py
python 6_onnx_donusum.py
python 7_uygulama_olustur.py
python 4_toplanan_veri.py
python 8_h2_olcumu.py
```

Bütün deneyler sabit tohumla (42-46) çalışır. Aynı bilgisayarda aynı sonuçları verir.

Uygulama için `index.html` dosyasını tarayıcıda açmanız yeterli. Tahmin internetten yüklenen onnxruntime-web ile yapılır. İnternet yoksa sayfa aynı ONNX modelini kendi içindeki JavaScript koduyla çalıştırır. Veri toplama sayfasının Drive'a kaydeden sürümünün kurulumu `KURULUM_VERI_TOPLAMA.md` dosyasında anlatılıyor.

## Veri setleri

- **QMNIST:** C. Yadav ve L. Bottou, "Cold case: The lost MNIST digits," NeurIPS 2019.
- **EMNIST-Digits:** G. Cohen vd., "EMNIST: Extending MNIST to handwritten letters," IJCNN 2017.
- **Dokunmatik ekran rakamları:** P. J. Corr vd., IMVIP 2017, https://github.com/PhilipCorr/numeral-gesture-dataset. Lisans belirtilmediği için yalnızca akademik amaçla ve kaynak gösterilerek kullanılmaktadır.
- **Toplanan veri:** Ekip içinde, aydınlatma metni onaylanarak toplandı. Kişisel bilgi içermez, katılımcılar rastgele numarayla tanımlanır.

## Not

3. hafta raporundaki ilk deneyler farklı bir bilgisayarda çalıştırılmıştı. O koşunun çıktıları `cikti/hafta3_ilk_kosu/` klasöründe duruyor. Aynı deneylerin ekip bilgisayarındaki sonuçları ve aradaki farkın açıklaması 4. hafta raporunda yer alıyor.
