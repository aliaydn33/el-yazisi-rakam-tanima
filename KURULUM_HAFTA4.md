# Hafta 4: Çalıştırma

| Dosya | Ne yapar | Süre (yaklaşık) |
|---|---|---|
| `5_model_karsilastirma.py` | Veri artırma yöntemlerini tek tek dener, MLP / LeNet-5 benzeri CNN / Derin CNN'i karşılaştırır, veri kaynaklarını (QMNIST, EMNIST, ikisi birlikte) karşılaştırır. Her deney 5 tohumla yapılır. | 45-50 dk |
| `cizim.html` | Uygulamanın çizim ekranı: tuval, Geri al, Temizle ve modelin göreceği 28×28 görüntünün canlı önizlemesi | - |
| `model_yapisi.py` | Modeller ve eğitim kodu (Derin CNN bu hafta eklendi) | - |
| `4_toplanan_veri.py` | Toplanan veriyi işler (4. haftada seçilen modeli de dener) | 10 sn |
| `testler/on_isleme_testi.py` | Çizim ekranındaki ön işlemenin Python'dakiyle aynı sonucu verdiğini test eder (Playwright gerekir) | 1 dk |

## 1. Deneyler

Uzun sürdüğü için bilgisayar şarja takılı ve uyku moduna geçmeyecek şekilde çalıştırılmalıdır.

```
python 5_model_karsilastirma.py
```

Her eğitimin sonucu bitince ekrana bir satır olarak yazılır ve `cikti/hafta4_kosular.csv` dosyasına eklenir. Betik yarıda kesilirse aynı komut tekrar çalıştırılır, biten eğitimler atlanır.

Diğer seçenekler:
- `python 5_model_karsilastirma.py --hizli`: küçük veriyle 2-3 dakikalık deneme (çıktılar `cikti/hizli_deneme` klasörüne yazılır)
- `python 5_model_karsilastirma.py --bolum B`: yalnızca bir bölümü çalıştırır (A, B veya C)

## 2. Çizim ekranı

`cizim.html` dosyası tarayıcıda açılır. Tuvale çizilen rakamın modelin göreceği 28×28 hali sağda canlı olarak görünür.

Ön işleme testi için (isteğe bağlı):

```
python -m pip install playwright
python -m playwright install chromium
python testler/on_isleme_testi.py
```

## 3. Toplanan veri

Deneyler bittikten sonra:

```
python 4_toplanan_veri.py
```

## Çıktılar

`cikti` klasöründe:
- `ozet_hafta4.txt`: bütün tablolar, istatistiksel testler ve seçim kararları
- `hafta4_kosular.csv`: her eğitimin ayrı sonucu
- `hafta4_A_artirma.csv`, `hafta4_B_modeller.csv`, `hafta4_C_veri_kaynagi.csv`, `hafta4_rakam_bazinda.csv`
- `hafta4_dokunmatik_kisi_ayrimi.csv`: dokunmatik verideki her kişinin doğrulama mı test mi olduğu
- `sekil_hafta4_artirma.png`, `sekil_hafta4_modeller.png`, `sekil_hafta4_veri_kaynagi.png`, `sekil_hafta4_rakam_bazinda.png`, `sekil_hafta4_karmasiklik.png`

`modeller` klasöründe (depoya alınmaz):
- `secilen_model_hafta4.pt`, `secilen_model_hafta4.txt`: uygulama için seçilen model (5. haftada ONNX'e çevrilecek)
