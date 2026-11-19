# 7. hafta

Bu hafta uygulamanın sunucu tarafı (FastAPI) yazıldı ve uygulama telefona kurulabilir hale getirildi (PWA). Ayrıca sayı ve işlem okuma, fotoğraftan okuma, yanlış tahminin düzeltilmesi ve düzeltmelerle ince ayar eklendi.

## Çalıştırma

```
python -m pip install fastapi uvicorn python-multipart opencv-python-headless
python 7_uygulama_olustur.py
python sunucu.py
```

Uygulama http://localhost:8000 adresinde açılır, API'nin açıklaması http://localhost:8000/docs adresindedir. Aynı ağdaki bir telefondan denemek için sunucu `python sunucu.py --host 0.0.0.0` ile başlatılır ve ekrana yazılan adres telefonda açılır. Kullanıcıların düzeltmeleri `geri_bildirim/` klasörüne yazılır, bu klasör depoya yüklenmez.

Telefona kurulum ve internetsiz açılma https gerektirir. Bu yüzden GitHub Pages'teki adreste çalışır. Fotoğraftan okuma telefonda yapılır, sunucu gerekmez. Sunucu olmadığında düzeltmeler telefonda saklanır, "Düzeltmeleri indir" ile dosya olarak alınabilir.

## İnce ayar

```
python 11_ince_ayar.py --deney
python 11_ince_ayar.py --kayitlar geri_bildirim duzeltmeler_*.json
```

Birincisi yöntemi ekibin topladığı veriyle dener (yaklaşık 1 dakika), ikincisi gerçek düzeltme kayıtlarıyla `modeller/ince_ayar_modeli.pt` dosyasını üretir. Uygulamadaki model kendiliğinden değişmez.

## Test

```
python -m pip install playwright
python -m playwright install chromium
python testler/hafta7_testi.py
```

Test sunucuyu kendisi başlatır, 200 fotoğraf ve 630 sayı/işlem örneği hazırlar. Sonuçlar `cikti/hafta7_testi.json` dosyasına yazılır.
