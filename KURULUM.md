# Hafta 2: Kurulum ve Çalıştırma

Bu hafta iki betik çalıştırılır:

| Dosya | Ne yapar | Süre (yaklaşık) |
|---|---|---|
| `1_veri_hazirla.py` | QMNIST, EMNIST-Digits ve dokunmatik ekran veri setlerini indirir, hazırlar ve inceler | 10-20 dk (çoğu indirme) |
| `2_ilk_cnn.py` | MLP ve ilk CNN'i eğitir, iki test kümesinde ölçer | 1-10 dk (bilgisayara göre) |

Ekran kartı (GPU) gerekmez, normal işlemcide çalışır.

## 1. Klasörün yeri

Proje klasörünü **OneDrive dışında** bir yerde tutun (örneğin `C:\Users\<kullanıcı>\Proje`). EMNIST indirildikten sonra klasörde yaklaşık 3 GB veri oluşur. Klasör OneDrive içindeyse bunların hepsi buluta yüklenmeye çalışılır.

## 2. Terminali açma

Klasörün içinde boş bir yere sağ tıklayın:
- Windows 11: **"Terminalde aç"**
- Windows 10: Shift tuşu basılıyken sağ tıklayıp **"PowerShell penceresini burada açın"**

Python'u kontrol edin:

```
python --version
```

`Python 3.10` veya daha yeni bir sürüm yazmalı. Hata verirse `py --version` deneyin. O çalışırsa aşağıdaki komutlarda `python` yerine `py` yazın.

## 3. Kütüphaneleri kurma (bir kez yapılır)

```
python -m pip install -r requirements.txt
```

PyTorch büyük olduğu için birkaç dakika sürebilir.

## 4. Çalıştırma

```
python 1_veri_hazirla.py
python 2_ilk_cnn.py
```

İlk komutta ilk yazının görünmesi 1-2 dakika sürebilir, çünkü PyTorch ilk açılışta yavaş yüklenir.

## Karşılaşılan sorunlar

- **"DLL load failed ... Uygulama Denetimi ilkesi bu dosyayı engelledi"**: Windows 11'in Akıllı Uygulama Denetimi scikit-learn'ün bir dosyasını engelliyor. Windows Güvenliği → Uygulama ve tarayıcı denetimi → Akıllı Uygulama Denetimi ayarları bölümünden kapatılabilir. Kapatıldıktan sonra Windows sürümüne göre tekrar açmak için Windows'u sıfırlamak gerekebilir.
- **EMNIST inmezse**: https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip dosyasını tarayıcıdan indirip `veri\ham\EMNIST\raw` klasörüne koyun ve komutu tekrar çalıştırın.

## Çıktılar

`cikti` klasöründe oluşanlar:
- `veri_ozeti.txt` / `.csv`: veri setlerinin örnek sayısı, kişi sayısı, sınıf dengesi, mürekkep oranı
- `ozet_hafta2.txt` / `sonuclar_hafta2.csv`: model sonuçları
- `sekil_*.png`: rapordaki şekiller

`veri` ve `modeller` klasörleri büyük olduğu için `.gitignore` ile depoya alınmaz.
