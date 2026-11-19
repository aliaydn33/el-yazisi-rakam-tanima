# 7. hafta

Bu hafta uygulamanın sunucu tarafı (FastAPI) yazıldı ve uygulama telefona kurulabilir hale getirildi (PWA). Ayrıca sayı ve işlem okuma, yanlış tahminin düzeltilmesi ve fotoğraftan okuma eklendi.

## Çalıştırma

```
python -m pip install fastapi uvicorn python-multipart opencv-python-headless
python 7_uygulama_olustur.py
python sunucu.py
```

Uygulama http://localhost:8000 adresinde açılır, API'nin açıklaması http://localhost:8000/docs adresindedir. Aynı ağdaki bir telefondan denemek için sunucu `python sunucu.py --host 0.0.0.0` ile başlatılır ve ekrana yazılan adres telefonda açılır. Kullanıcıların düzeltmeleri `geri_bildirim/` klasörüne yazılır, bu klasör depoya yüklenmez.

Telefona kurulum ve internetsiz açılma https gerektirir. Bu yüzden GitHub Pages'teki adreste çalışır. Sunucu orada olmadığı için fotoğraftan okuma bölümü görünmez, düzeltmeler de telefonda saklanır.

## Test

```
python -m pip install playwright
python -m playwright install chromium
python testler/hafta7_testi.py
```

Test sunucuyu kendisi başlatır, 200 fotoğraf ve 630 sayı/işlem örneği hazırlar. Sonuçlar `cikti/hafta7_testi.json` dosyasına yazılır.
