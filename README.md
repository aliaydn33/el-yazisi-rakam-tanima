# El Yazısı Rakamları Tanıyan Etkileşimli Uygulama

Fırat Üniversitesi Bilgisayar Mühendisliği, Bilgisayar Mühendisliği Tasarımı dersi projesi. **41.GRUP:** Eray Yolcu, Ali Aydın, Metehan Borlukçu.

Ekrana parmakla, kalemle ya da fareyle çizilen rakamları tarayıcıda gerçek zamanlı olarak tanıyan bir web uygulaması geliştiriyoruz. Projenin temel mühendislik problemi şu: Taranmış kâğıt görüntüleriyle (MNIST) eğitilen modeller, ekrana çizilen rakamlarda aynı başarıyı gösteremiyor. Literatürde buna veri dağılımı kayması deniyor.

## Durum (4. hafta sonu)

| Hafta | İş | Durum |
|---|---|---|
| 1 | Problem tanımı, literatür, hedefler, plan | Tamamlandı |
| 2 | Veri setleri (QMNIST, EMNIST-Digits, dokunmatik ekran verisi), ön işleme, ilk CNN | Tamamlandı |
| 3 | Veri artırma, veri toplama sayfası | Tamamlandı |
| 4 | Model karşılaştırması (5 tohum, istatistiksel testler), EMNIST deneyleri, çizim ekranı | Tamamlandı |
| 5 | Modelin tarayıcıya taşınması (ONNX) ve canlı tahmin | Planlandı |
| 6 | "Rakam değil" uyarısı, Grad-CAM | Planlandı |
| 7 | FastAPI sunucu, PWA | Planlandı |
| 8 | Kullanıcı testi (SUS), istatistiksel karşılaştırma | Planlandı |
| 9 | Final, demo | Planlandı |

**Şu anki uygulama modeli:** LeNet-5 benzeri CNN (61.706 parametre, veri artırma yok, QMNIST ile eğitildi).

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
model_yapisi.py           Modeller (MLP, LeNet-5 benzeri CNN, Derin CNN), eğitim ve veri artırma
ortak.py                  Ön işleme (MNIST biçimine getirme, vuruşları görüntüye çevirme)
veri_toplama.html         Veri toplama sayfası (aydınlatma metni, 50 rakam, JSON dışa aktarma)
cizim.html                Uygulamanın çizim ekranı (tuval, geri al, temizle, 28×28 önizleme)
testler/                  Çizim ekranındaki ön işlemenin Python ile tutarlılık testi
toplanan_veri/            Ekip içinde toplanan anonim çizimler (JSON)
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
```

Bütün deneyler sabit tohumla (42-46) çalışır. Aynı bilgisayarda aynı sonuçları verir.

Veri toplama için `veri_toplama.html`, çizim ekranı için `cizim.html` dosyasını tarayıcıda açmanız yeterli. İki sayfa da sunucu gerektirmez.

## Veri setleri

- **QMNIST:** C. Yadav ve L. Bottou, "Cold case: The lost MNIST digits," NeurIPS 2019.
- **EMNIST-Digits:** G. Cohen vd., "EMNIST: Extending MNIST to handwritten letters," IJCNN 2017.
- **Dokunmatik ekran rakamları:** P. J. Corr vd., IMVIP 2017, https://github.com/PhilipCorr/numeral-gesture-dataset. Lisans belirtilmediği için yalnızca akademik amaçla ve kaynak gösterilerek kullanılmaktadır.
- **Toplanan veri:** Ekip içinde, aydınlatma metni onaylanarak toplandı. Kişisel bilgi içermez, katılımcılar rastgele numarayla tanımlanır.

## Not

3. hafta raporundaki deneyler bulut ortamındaki bir bilgisayarda çalıştırılmıştı. O koşunun çıktıları `cikti/hafta3_ilk_kosu_bulut/` klasöründe duruyor. Aynı deneylerin ekip bilgisayarındaki sonuçları ve aradaki farkın açıklaması 4. hafta raporunda yer alıyor.
