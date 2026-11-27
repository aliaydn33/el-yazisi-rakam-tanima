# 8. hafta

Bu hafta uygulamanın kullanıcı testi (SUS anketi) yapıldı ve modeller istatistiksel olarak karşılaştırıldı.

## Kullanıcı testi

Test uygulamanın kendisiyle yapılır. Katılımcılara gönderilen bağlantı:

```
https://aliaydn33.github.io/el-yazisi-rakam-tanima/?test=<kimlik>
```

`<kimlik>`, veri toplamada kullanılan Apps Script web uygulaması adresinde `/s/` ile `/exec` arasındaki kısımdır (`KURULUM_VERI_TOPLAMA.md`). Sonuçlar Drive'daki `rakam_verisi` klasörüne `kullanici_testi_*.json` olarak yazılır, Apps Script'te değişiklik gerekmez. Bağlantı yalnızca `?test` ile açılırsa sonuç Drive'a gönderilmez, dosya olarak indirilir (ekip içinde denemek için).

```
python 12_kullanici_testi.py --aday
python 12_kullanici_testi.py rakam_verisi.zip
```

Birincisi testten önce bir kez çalıştırılır ve ince ayar adayını hazırlar (`onnx/ince_ayar_aday_int8.onnx`). İkincisi Drive'dan zip olarak indirilen klasörü okur. Klasördeki veri toplama dosyaları atlanır.

## İstatistiksel karşılaştırma

```
python 13_istatistiksel_karsilastirma.py
```

Modeller 4. haftadaki ayarlarla yeniden eğitilir (ilk çalıştırmada yaklaşık 20 dakika), sonra kişi düzeyinde karşılaştırılır.

## Test

```
python testler/hafta8_testi.py
```

Kullanıcı testi ekranını telefon ve bilgisayar boyutunda baştan sona dener (Drive'a bir şey gönderilmez), farklı ekran genişliklerinde düğme boyutlarını ve renk kontrastını ölçer.
