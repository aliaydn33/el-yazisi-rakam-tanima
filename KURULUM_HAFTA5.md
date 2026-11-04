# 5. hafta: modelin tarayıcıya taşınması ve uygulamanın ilk sürümü

Bu hafta 4. haftada seçilen LeNet-5 benzeri CNN, ONNX biçimine dönüştürüldü ve çizim ekranına bağlandı. Uygulama (`index.html`) çizim yapılırken tahmini canlı olarak gösteriyor. Tahmin kullanıcının tarayıcısında yapılıyor, sunucu gerekmiyor.

## Yeni dosyalar

```
6_onnx_donusum.py        Modelleri ONNX'e çevirir, 8 bit (int8) sürümünü üretir, PyTorch ile karşılaştırır
7_uygulama_olustur.py    uygulama_sablon.html'e onnx/ klasöründeki modelleri ekleyip index.html'i üretir
8_h2_olcumu.py           Toplanan veride H2 ölçümü (doğruluk, güven aralığı, katılımcı ve rakam bazında hatalar)
uygulama_sablon.html     Uygulamanın kaynağı (çizim, ön işleme, tahmin, hız testi)
index.html               Uygulamanın kendisi (modeller içine gömülü, tek dosya)
onnx/                    ONNX modelleri ve uygulamanın gösterdiği model bilgileri
testler/uygulama_testi.py  Tarayıcıdaki tahminlerin Python ile karşılaştırılması, hız ve internetsiz çalışma testi
apps_script/             Veri toplama sayfasının Google Apps Script sürümü (Drive'a otomatik kayıt)
.nojekyll                GitHub Pages'in dosyaları olduğu gibi yayınlaması için
```

## Çalıştırma

```
python -m pip install onnx onnxruntime
python 6_onnx_donusum.py
python 7_uygulama_olustur.py
python 4_toplanan_veri.py
python 8_h2_olcumu.py
```

`6_onnx_donusum.py`, `modeller/secilen_model_hafta4.pt` dosyasını kullanır. `modeller/hafta4/B_Derin_CNN_42.pt` varsa Derin CNN'i de dönüştürür. Sonuçlar `cikti/ozet_hafta5_onnx.txt` dosyasına yazılır. `7_uygulama_olustur.py` ise `index.html` dosyasını yeniden oluşturur. Bu dosya çift tıklanarak açılabilir. `4_toplanan_veri.py`, `toplanan_veri` klasöründeki çizimleri hazırlar. `8_h2_olcumu.py` bunlarla H2 ölçümünü yapar, sonuçları `cikti/ozet_hafta5_h2.txt` dosyasına yazar.

İsteğe bağlı tarayıcı testi (Chromium gerekir):

```
python -m pip install playwright
python -m playwright install chromium
python testler/uygulama_testi.py
```

## Uygulamanın yayınlanması (GitHub Pages, bir kez)

1. Dosyalar GitHub'a gönderildikten sonra depo sayfasında **Settings → Pages**'e girin.
2. **Build and deployment** altında Source: **Deploy from a branch**, Branch: **main**, klasör: **/ (root)** seçip **Save**'e basın.
3. Bir iki dakika sonra uygulama şu adreste açılır: https://aliaydn33.github.io/el-yazisi-rakam-tanima/

Veri toplama sayfası da aynı adresten açılabilir (`.../veri_toplama.html`), ama oradan yapılan çizimler Drive'a kaydedilmez. Katılımcılara Apps Script linki gönderilmeye devam edilmelidir (`KURULUM_VERI_TOPLAMA.md`).

## Telefonda hız ölçümü (H3)

1. Uygulamayı telefonda açın ve "onnxruntime-web hazır" yazısının çıkmasını bekleyin (ilk açılışta birkaç saniye sürer).
2. Bir rakam çizin ya da tuvali boş bırakın, **Model ve hız ölçümü** bölümündeki **Hız testi** düğmesine basın.
3. **Sonucu kopyala** ile çıkan metni ekibe gönderin. Metinde ölçüm sonucu, model ve tarayıcı bilgisi bulunur.
