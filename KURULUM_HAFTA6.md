# 6. hafta

Bu hafta uygulamaya "rakam değil" uyarısı ve Grad-CAM ısı haritası eklendi.

## Çalıştırma

```
python -m pip install onnx onnxruntime
python 4_toplanan_veri.py
python 9_rakam_degil.py
python 10_gradcam.py
python 7_uygulama_olustur.py
```

`9_rakam_degil.py` ilk çalıştırmada Quick, Draw! veri setinden 20 kategorinin çizimlerini indirir (yaklaşık 40 MB). EMNIST harflerini `veri/ham/EMNIST` klasöründen okur, klasör yoksa EMNIST'i indirir (yaklaşık 550 MB). Ardından modeli iki şekilde 5'er kez eğitir, bu yüzden toplam süre 10-15 dakikayı bulabilir.

Sonuçlar `cikti/ozet_hafta6_rakam_degil.txt` dosyasına, raporda kullanılan dosyaların hepsi `cikti/hafta6_sonuclar.zip` dosyasına yazılır. `7_uygulama_olustur.py` uygulamayı yeni modelle günceller.
