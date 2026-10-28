# Hafta 3: Çalıştırma

| Dosya | Ne yapar | Süre (yaklaşık) |
|---|---|---|
| `3_emnist_artirma.py` | CNN'i 4 farklı durumda eğitir (QMNIST / EMNIST, veri artırma yok / var) ve dokunmatik ekran verisinde karşılaştırır | 10 dk |
| `veri_toplama.html` | Veri toplama sayfası. Tarayıcıda açılır, her katılımcı 50 rakam çizer | kişi başı 5 dk |
| `4_toplanan_veri.py` | Toplanan çizimleri okur, 28×28'e çevirir, modelleri bu veride dener | 10 sn |
| `model_yapisi.py` | Model ve veri artırma kodu (diğer betikler kullanır, tek başına çalıştırılmaz) | - |

Önce 2. haftanın betikleri çalıştırılmış olmalıdır (`veri` klasörü).

## 1. Eğitim

```
python 3_emnist_artirma.py
```

Her şeyin çalıştığını hızlıca görmek için `python 3_emnist_artirma.py --epoch 1` kullanılabilir. Asıl sonuçlar parametresiz (10 epoch) çalıştırmayla alınır.

## 2. Veri toplama

1. `veri_toplama.html` dosyası tarayıcıda açılır.
2. Katılımcı aydınlatma metnini onaylayıp oturumu başlatır. Sayfa 50 rakam ister, her biri çizilip **Kaydet**'e basılır.
3. Sonunda **Dosyayı indir** ile `rakamlar_K-xxxx_....json` dosyası indirilir.
4. Bu dosya `toplanan_veri` klasörüne konur.

Fare, dokunmatik ekran ve kalem desteklenir. Sayfa hangisinin kullanıldığını kendisi kaydeder.

## 3. Toplanan veriyi işleme

```
python 4_toplanan_veri.py
```

## Çıktılar

`cikti` klasöründe oluşanlar:
- `ozet_hafta3.txt`, `sonuclar_hafta3.csv`: dört deneyin sonuçları ve McNemar testi
- `ozet_toplanan.txt`: toplanan verinin özeti ve modellerin bu verideki sonuçları
- `sekil_hafta3_karsilastirma.png`, `sekil_rakam_bazinda_hata.png`, `sekil_artirma_ornekleri.png`, `sekil_hatali_ornekler.png`, `sekil_egitim_egrisi_hafta3.png`, `sekil_toplanan_ornekler.png`
- `rakam_bazinda_hata_hafta3.csv`, `karisan_rakamlar_hafta3.csv`

`veri` ve `modeller` klasörleri depoya alınmaz.
