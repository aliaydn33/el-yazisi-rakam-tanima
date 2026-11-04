/**
 * Rakam toplama sunucusu (Google Apps Script)
 *
 * - doGet : veri toplama sayfasini (Sayfa.html) acar. Web uygulamasinin linki herkese gonderilir.
 * - kaydet: sayfadan gelen oturumu Google Drive'daki "rakam_verisi" klasorune JSON dosyasi olarak yazar.
 *           Ayni oturum her 5 cizimde bir yeniden gonderildigi icin dosya her seferinde guncellenir.
 * - doPost: sayfa baska bir adresten (or. GitHub Pages) acildiginda ayni kaydi POST ile yapar.
 *
 * Kurulum: Dagit > Yeni dagitim > Web uygulamasi, "Su kullanici olarak calistir: Ben",
 *          "Erisimi olan kullanicilar: Herkes".
 */
var KLASOR_ADI = "rakam_verisi";
var EN_BUYUK_BOYUT = 3 * 1024 * 1024;   // 3 MB (50 cizimlik bir oturum yaklasik 100 KB)

function doGet() {
  return HtmlService.createHtmlOutputFromFile("Sayfa")
    .setTitle("Rakam Toplama Oturumu")
    .addMetaTag("viewport", "width=device-width, initial-scale=1, viewport-fit=cover");
}

function kaydet(metin) {
  if (typeof metin !== "string" || metin.length > EN_BUYUK_BOYUT) throw new Error("Gecersiz veri");
  var veri = JSON.parse(metin);
  if (!veri.katilimci_no || !veri.ornekler || !veri.ornekler.length) throw new Error("Eksik veri");

  var ad = String(veri.dosya_adi || ("rakamlar_" + veri.katilimci_no + ".json")).replace(/[^A-Za-z0-9_.-]/g, "");
  if (!/\.json$/.test(ad)) ad += ".json";

  var kilit = LockService.getScriptLock();
  kilit.waitLock(20000);
  try {
    var klasor = klasorAl_();
    var dosyalar = klasor.getFilesByName(ad);
    if (dosyalar.hasNext()) dosyalar.next().setContent(metin);
    else klasor.createFile(ad, metin, MimeType.PLAIN_TEXT);
  } finally {
    kilit.releaseLock();
  }
  return { durum: "tamam", dosya: ad, cizim: veri.ornekler.length };
}

function doPost(e) {
  var sonuc;
  try {
    sonuc = kaydet(e.postData.contents);
  } catch (hata) {
    sonuc = { durum: "hata", mesaj: String(hata) };
  }
  return ContentService.createTextOutput(JSON.stringify(sonuc)).setMimeType(ContentService.MimeType.JSON);
}

function klasorAl_() {
  var it = DriveApp.getFoldersByName(KLASOR_ADI);
  return it.hasNext() ? it.next() : DriveApp.createFolder(KLASOR_ADI);
}
