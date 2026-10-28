# Hafta 4: Çalıştırma

Bu haftanın dosyaları:

| Dosya | Ne yapar | Süre (yaklaşık) |
|---|---|---|
| `5_model_karsilastirma.py` | Veri artırma yöntemlerini tek tek dener, MLP / LeNet-5 benzeri CNN / Derin CNN'i karşılaştırır, veri kaynaklarını (QMNIST, EMNIST, ikisi birlikte) karşılaştırır. Her deney 5 tohumla yapılır. | 35-45 dk |
| `cizim.html` | Uygulamanın çizim ekranı: tuval, Geri al, Temizle ve modelin göreceği 28×28 görüntünün canlı önizlemesi | - |
| `model_yapisi.py` | Modeller ve eğitim kodu (güncellendi, Derin CNN eklendi) | - |
| `4_toplanan_veri.py` | Toplanan veriyi işler (güncellendi, 4. haftada seçilen modeli de dener) | 10 sn |
| `testler/on_isleme_testi.py` | Çizim ekranındaki ön işlemenin Python'dakiyle aynı sonucu verdiğini test eder (Playwright gerekir, isteğe bağlı) | - |

## 1. Dosyaları yerleştirme

Zip'in içindekileri yine `hafta2_proje` klasörüne kopyalayın. Windows "aynı adlı dosya var" derse **"Hedefteki dosyaları değiştir"** seçin. `model_yapisi.py` ve `4_toplanan_veri.py` dosyalarının yeni sürümleri eskisinin yerine geçer.

## 2. Asıl deneyi başlatma

Bilgisayarı **şarja takın** ve uyku moduna geçmeyecek şekilde açık bırakın. Sonra terminalde şunu çalıştırın:

```
python 5_model_karsilastirma.py
```

Her deneyin sonucu bitince ekrana bir satır olarak yazılır. Toplam yaklaşık 70 eğitim yapılır.

Betik yarıda kesilirse aynı komutu tekrar çalıştırın. Biten deneyleri atlayıp kaldığı yerden devam eder.

## 3. Deney sürerken

- **Veri toplama:** Eray ve Metehan `veri_toplama.html` dosyasını açıp 50'şer rakam çizer. İnen `.json` dosyaları `toplanan_veri` klasörüne konur.
- **Çizim ekranı:** `cizim.html` dosyasına çift tıklayın. Tuvale bir rakam çizin, sağda modelin göreceği 28×28 görüntü belirir. Geri al ve Temizle düğmelerini deneyin.

## 4. Deney bittikten sonra

```
python 4_toplanan_veri.py
```

Bu komut toplanan verileri işler ve 4. haftada seçilen model dahil bütün modelleri bu veride dener.

## 5. Rapor için gönderilecekler

`cikti` klasöründen:
- `ozet_hafta4.txt`
- `ozet_toplanan.txt`
- `sekil_hafta4_artirma.png`
- `sekil_hafta4_modeller.png`
- `sekil_hafta4_veri_kaynagi.png`
- `sekil_hafta4_rakam_bazinda.png`
- `sekil_hafta4_karmasiklik.png`
- `sekil_toplanan_ornekler.png`

`modeller` klasöründen (5. haftada modeli tarayıcıya taşımak için gerekli):
- `secilen_model_hafta4.pt`
- `secilen_model_hafta4.txt`
