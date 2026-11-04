# Veri toplama sayfasının yayınlanması (Google Apps Script)

Sayfa Google Apps Script ile yayınlandığında tek bir link her cihazda ve her tarayıcıda açılır, uygulama içi tarayıcılar dahil. Çizimler indirme gerekmeden, her 5 çizimde bir ve oturum sonunda otomatik olarak Google Drive'daki `rakam_verisi` klasörüne kaydedilir. Sayfa yarıda kapanırsa o ana kadar gönderilen çizimler kaybolmaz. Sayfa yeniden açılınca kaldığı yerden devam eder.

## Kurulum (bir kez, yaklaşık 5 dakika)

1. https://script.google.com adresine gidip **Yeni proje** oluşturun. Sol üstteki "Adsız proje" yazısına tıklayıp adı `Rakam Toplama` yapın.
2. Açık olan `Kod.gs` dosyasının içini silin ve `apps_script/Kod.gs` dosyasının içeriğini yapıştırın.
3. Soldaki **Dosyalar** başlığının yanındaki **+** işaretinden **HTML**'i seçin ve adını `Sayfa` yazın. İçindeki hazır kodu silin, `apps_script/Sayfa.html` dosyasının içeriğini yapıştırın. **Ctrl + S** ile kaydedin.
4. Sağ üstten **Dağıt → Yeni dağıtım**'a basın. "Tür seçin" yanındaki dişli simgesinden **Web uygulaması**'nı seçin.
   - Şu kullanıcı olarak yürüt: **Ben**
   - Erişimi olanlar: **Herkes**
   - **Dağıt**'a basın.
5. **Erişimi yetkilendir** deyip Google hesabınızı seçin. "Google bu uygulamayı doğrulamadı" ekranı çıkarsa **Gelişmiş → Rakam Toplama (güvenli değil) sayfasına git → İzin ver** deyin. Bu uyarı, kendi yazdığınız her Apps Script projesinde çıkar.
6. Çıkan **Web uygulaması URL'sini** kopyalayın (`.../exec` ile biter). Katılımcılara gönderilecek link budur.

Sayfanın üstünde Google'ın "Bu uygulama bir Google Apps Script kullanıcısı tarafından oluşturuldu" yazısı görünür. Bu normaldir.

## Sayfa güncellenirse

`Sayfa.html` değiştirildikten sonra **Dağıt → Dağıtımları yönet**'e gidin, kalem simgesine basın ve "Sürüm" kısmında **Yeni sürüm**'ü seçip **Dağıt**'a basın. Link değişmez.

## Toplanan verilerin alınması

1. Google Drive'da `rakam_verisi` klasörüne sağ tıklayıp **İndir**'e basın. Klasör zip olarak iner.
2. Zip'teki `.json` dosyalarını proje klasöründeki `toplanan_veri` klasörüne koyun.
3. `python 4_toplanan_veri.py` komutunu çalıştırın.

Aynı oturum hem Drive'dan hem dosya olarak alınırsa betik aynı çizimleri iki kez saymaz.

## İnternetsiz ya da sunucusuz kullanım

`veri_toplama.html` doğrudan bir tarayıcıda açıldığında sunucuya göndermez. Bu durumda oturum sonunda dosya olarak paylaşma ve indirme düğmeleri çıkar.
