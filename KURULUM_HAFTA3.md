# Hafta 3: Çalıştırma

Bu haftanın dosyaları:

| Dosya | Ne yapar | Süre (yaklaşık) |
|---|---|---|
| `3_emnist_artirma.py` | CNN'i 4 farklı durumda eğitir (QMNIST / EMNIST, veri artırma yok / var) ve dokunmatik ekran verisinde karşılaştırır | 10 dk |
| `veri_toplama.html` | Veri toplama sayfası. Tarayıcıda açılır, her katılımcı 50 rakam çizer | kişi başı 5 dk |
| `4_toplanan_veri.py` | Toplanan çizimleri okur, 28×28'e çevirir, modelleri bu veride dener | 10 sn |
| `model_yapisi.py` | Model ve veri artırma kodu (diğer betikler kullanır, tek başına çalıştırılmaz) | - |

## 1. Dosyaları yerleştirme

Zip'in içindeki dosyaları **2. haftanın proje klasörüne** kopyalayın. Bu, içinde `veri`, `modeller` ve `cikti` klasörleri olan klasördür. Yeni kurulum gerekmez, kütüphaneler aynı.

## 2. Eğitimi başlatma

Klasörde terminali açın ve şu komutu çalıştırın:

```
python 3_emnist_artirma.py
```

Önce her şeyin çalıştığını 1 dakikada görmek isterseniz `python 3_emnist_artirma.py --epoch 1` çalıştırabilirsiniz. Asıl sonuç için komutu **parametresiz** çalıştırın (10 epoch).

## 3. Eğitim sürerken veri toplama

1. `veri_toplama.html` dosyasına çift tıklayın, tarayıcıda açılır.
2. Aydınlatma metnini onaylayıp oturumu başlatın. Sayfa 50 rakam ister, her birini çizip **Kaydet**'e basın.
3. Sonunda **Dosyayı indir**'e basın. `rakamlar_K-xxxx_....json` adlı dosya İndirilenler klasörüne iner.
4. Bu dosyayı proje klasöründeki `toplanan_veri` klasörüne taşıyın. Klasör yoksa oluşturun.
5. Ekipteki herkes aynı şekilde bir oturum yapsın. Başka bilgisayardan çizen arkadaşlar `veri_toplama.html` dosyasını kendi bilgisayarında açabilir ve indirdiği `.json` dosyasını gönderebilir.

Fare, dokunmatik ekran ve kalem hepsi olur. Sayfa hangisinin kullanıldığını kendisi kaydeder.

## 4. Toplanan veriyi işleme

Eğitim bittikten ve `.json` dosyaları `toplanan_veri` klasörüne konduktan sonra şu komutu çalıştırın:

```
python 4_toplanan_veri.py
```

## 5. Rapor için gönderilecekler (`cikti` klasöründen)

- `ozet_hafta3.txt`
- `ozet_toplanan.txt`
- `sekil_hafta3_karsilastirma.png`
- `sekil_rakam_bazinda_hata.png`
- `sekil_artirma_ornekleri.png`
- `sekil_hatali_ornekler.png`
- `sekil_egitim_egrisi_hafta3.png`
- `sekil_toplanan_ornekler.png`
- `rakam_bazinda_hata_hafta3.csv`

GitHub'a bu haftanın kod dosyalarını, `toplanan_veri` klasörünü ve `cikti` klasörünü yükleyin. `veri` ve `modeller` klasörleri yine yüklenmez.
