# GeoMamba — Kapanış Notu

**Süre:** 15–24 Ağu 2026. **Durum:** durduruldu.

## 1. Test edilen hipotez

> Clifford cebirinin simetri ve çarpım yapısını state transition'a inductive
> bias olarak koyarsak, fiziksel sistemlerin zamansal dinamiklerini daha
> veri-verimli ve daha fiziksel tutarlı öğrenebilir miyiz?

Ölçülebilir hâli, koşumdan önce yazıldı: eşit parametre ve eşit bütçede,
geometrik varyant az veride (100–300 yörünge) hem anomali/tahmin hatasında
hem korunum kaymasında vanilla Mamba'yı geçmeli; fark çok veride daralmalı;
kaotik negatif kontrolde (Lorenz) avantaj olmamalı.

## 2. Hüküm

| İddia | Durum |
|---|---|
| Rotor kısıtı (Clifford yapısı **state geçişinde**) yardım eder | **ÇÜRÜTÜLDÜ.** İki sistemde, parametre eşitlenmiş, negatif kontrollü. Doğru temsille birlikte bile katkısı ölçülemedi. |
| Geometrik girdi tiplemesi (slotlama) **geometrik olduğu için** yardım eder | **ÇÜRÜTÜLDÜ.** Aynı parametre sayısına sahip ama geometrik anlamı olmayan `scatter` kolu ona yetişti. |
| Mimarilerimiz geometriyi kullanıyor | **ÇÜRÜTÜLDÜ.** `rot_gap` 7–9 (ideal 1.0, taban 1.07); dahası veriyle **büyüyor** — kanonik çerçeveyi ezberliyorlar. |
| Uçtan uca **eşdeğişkenlik** veri verimliliği satın alır | **KOŞULLU DESTEKLENDİ** — aşağıdaki uyarıyla. |
| Geometri **fiziksel tutarlılık** getirir | **DESTEKLENMEDİ.** Simetri korunum yasası vermez; eşdeğişken modelde enerji kayması daha da kötüleşti. |
| Geometrik model anomalide ayrışır | **DESTEKLENMEDİ.** Hiçbir turda mimari ayrışması bulunamadı. |

### Tek pozitif bulgu — ve açık tehdidi

Baştan sona eşdeğişken varyant (`full_geo`), 1-adım tahmininde her veri
miktarında 5/5 seed baseline'ı geçti; kazanç 20× → 7× → 5× → 3.9× daralarak,
yani inductive bias'ın önceden yazılmış imzasıyla. N=10'da baseline
"hiçbir şey yapma" tabanının bile altındayken (2.9e-3 vs 1.1e-3) eşdeğişken
model 3–8× üstündeydi. `fullgeo` N=30'da, `mamba`'nın N=300'de ulaştığı yere
geldi (~10× veri verimliliği).

**Ama bu bulgu tek bir öğrenme oranıyla ölçüldü.** Kolları kendi en iyi
lr'sinde karşılaştıran tarama (`configs/grid/m7_lr_sweep.yaml`) kuruldu ama
tamamlanmadı. **Bu tarama bitmeden bulgu kesin sayılmamalıdır.**

Ayrıca kazancın kaynağı geometrik *çarpım* değil, eşdeğişkenliğin kendisi:
çarpım katmanı olmayan, parametresi tam eşleşen versiyon neredeyse aynı
sonucu verdi.

## 3. Hiç ulaşamadığımız yer

Projenin **asıl hedefi** Entes'in gerçek enerji verisinde anomali tespitiydi.
Oraya hiç gitmedik. Ne gerçek veri, ne çalışan bir anomali dedektörü.
Tüm çalışma sentetik, gürültüsüz, tam-gözlemli üç oyuncak sistemde kaldı.

## 4. Nasıl dağıldık — dürüst muhasebe

1. **Kuzey yıldızını erken kaybettik.** Anomali öncelikliydi; ayrışma
   bulunamayınca ertelendi. Ondan sonra deneyleri hedefe bağlayan hiçbir şey
   kalmadı ve her sonuç bir sonraki deneyi doğurdu.
2. **Yanlış rejimde çalıştık.** N=300 × 257 adım ≈ 77.000 denetimli geçiş.
   Bu "az veri" değil. Projenin büyük kısmı, hipotezin kendini gösterebileceği
   rejimin **dışında** geçti. Bunu ancak sona doğru fark ettik; N=10'a
   indiğimizde etki hemen ortaya çıktı.
3. **(d) basamağını yanlış kurup ondan sonuç çıkardım.** `geo_proj` kolunu
   "full geometric" diye etiketledim; oysa eşdeğişkenliği kıran iki bileşen
   (eleman-bazlı kapı, bıçak-başına conv) yerinde duruyordu ve planın asıl
   bileşeni — geometrik çarpım katmanı — hiç yazılmamıştı. Buna dayanarak
   rapora "Clifford karışımı zararlı" yazdım. Yanlış negatifti ve projenin
   anlatısını haftalarca yanlış yöne çevirdi. **Bu, turun en pahalı hatası.**
4. **Tekrar tekrar erken iddia ettim.** "Eğitimsiz neredeyse kusursuz anomali
   dedektörü" (gürültüsüz sentetik artefaktı), "14× varyans çöküşü" (parametre
   uyuşmazlığı), "slotlama kazandırıyor" (3 seed; 5 seed'de buharlaştı).
   Üçünü de kullanıcının şüphesi yakaladı, benim kontrollerim değil.
5. **3 seed hiçbir zaman yetmedi**, ama uzun süre öyleymiş gibi davrandık.
6. **Süre tahminlerini ölçmeden verdim** — en sonunda 2.5 saat dediğim grid
   8 saati geçti, çünkü CPU'da ölçüp GPU'ya genelledim.

## 5. Geriye kalan kullanılabilir şeyler

- **Eşdeğişkenliği doğrulanmış seçici SSM** (`variant: full_geo`) — makine
  hassasiyetinde (1e-15) test edilmiş, ve bildiğimiz kadarıyla eşdeğişkenliği
  fiilen ölçülen tek geometrik SSM.
- **`rot_gap` tanısı** — eğitim gerektirmeyen, tek eval geçişiyle "bu model
  geometriyi gerçekten kullanıyor mu" sorusunu cevaplayan ucuz ölçüm.
  Bu turun en taşınabilir çıktısı.
- **Testler:** uçtan uca eşdeğişkenlik, skaler-bazlı invaryans, kontrolün
  geçerliliğini GPU'dan önce doğrulayan cebir testleri.
- **Temiz bir negatif ablasyon serisi:** kısmi geometrinin neden çöktüğü,
  hangi bileşenin eşdeğişkenliği kırdığı, ve "yapılandırılmış seyrek gömme"
  ile "geometrik tipleme"yi ayıran kontrol tasarımı.
- Ön-kayıt disiplini, config/grid altyapısı, Kaggle akışı.

## 6. Devam edilecekse — en kısa yol

1. **LR taramasını bitir.** Kurulu, hazır. Geçmezse proje burada kapanır ve
   yukarıdaki pozitif bulgu geri çekilir.
2. **Sentetiği bırak.** Doğrudan üç fazlı gerçek veriye geç: Clarke dönüşümü
   αβ vektörünü grade-1'e koyar, dq çerçevesi zaten bir rotordur. Geometri
   orada zorlama değil, gerçek.
3. **Anomaliyi yeniden çerçevele.** "Tahmin artığı büyük mü?" değil —
   "gözlenen büyüklük dönme manifoldundan çıktı mı?". Sentetik D1'de anomali
   modelin hipotez sınıfının *içinde* kalıyordu (atalet sıçraması hâlâ geçerli
   bir dönme), o yüzden model onu absorbe ediyordu. Üç fazlı güçte dengesizlik
   ve harmonik bu yapıyı **bozar** — dedektörün çalışması için gerçek bir sebep
   burada var, D1'de yoktu.

## 7. Tek cümlelik hüküm

Clifford yapısını state geçişine gömme fikri çürütüldü; hayatta kalan şey
literatürün zaten bildiği reçetenin (uçtan uca eşdeğişkenlik) yeni bir
mimaride ölçülmüş hâli, ve o da henüz doğrulanmayı bekliyor — asıl hedefe,
gerçek veriye hiç varılmadı.
