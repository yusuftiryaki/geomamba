# GeoMamba — Kapanış Notu

**Süre:** 15–26 Ağu 2026. **Durum:** kapatıldı.

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
| Uçtan uca **eşdeğişkenlik** veri verimliliği satın alır | **DESTEKLENDİ** — lr taramasıyla doğrulandı (§2.1). |
| Geometri **fiziksel tutarlılık** getirir | **DESTEKLENMEDİ.** Simetri korunum yasası vermez; eşdeğişken modelde enerji kayması daha da kötüleşti. |
| Geometrik model anomalide ayrışır | **DESTEKLENMEDİ.** Hiçbir turda mimari ayrışması bulunamadı. |

### Tek pozitif bulgu — ve açık tehdidi

Baştan sona eşdeğişken varyant (`full_geo`), 1-adım tahmininde her veri
miktarında 5/5 seed baseline'ı geçti; kazanç 20× → 7× → 5× → 3.9× daralarak,
yani inductive bias'ın önceden yazılmış imzasıyla. N=10'da baseline
"hiçbir şey yapma" tabanının bile altındayken (2.9e-3 vs 1.1e-3) eşdeğişken
model 3–8× üstündeydi. `fullgeo` N=30'da, `mamba`'nın N=300'de ulaştığı yere
geldi (~10× veri verimliliği).

### 2.1 LR taraması — tehdit kapandı, üstelik ters yönde

Bulgu tek bir öğrenme oranıyla (3e-4) ölçülmüştü; eşdeğişken modellerin o
ayarı baseline'dan çok sevmiş olma ihtimali tek ciddi tehditti. 72 koşuluk
tarama tamamlandı (`configs/grid/m7_lr_sweep.yaml`). lr seçimi **validation**
ile yapıldı, test setine bakılmadan.

**Baseline'ın en iyi lr'si N=30'da 3e-4 çıktı — yani tam da kullandığımız
varsayılan. Eşdeğişken modeller ise 3e-3 istiyor, 10× daha büyük.** Yani
orijinal gridde eşdeğişken modeller kayırılmamış, **cezalandırılmıştı**.

Her kol kendi en iyi lr'sinde, test medyanı:

| N | mamba | fullgeonb | fullgeo | kazanç |
|---|---|---|---|---|
| 30 | 4.11e-4 @3e-4 | 2.36e-5 @3e-3 | 2.39e-5 @3e-3 | **17.4× / 17.2×** |
| 300 | 3.55e-5 @3e-3 | 5.92e-6 @3e-3 | 6.34e-6 @3e-3 | **6.0× / 5.6×** |

Önceki (herkes 3e-4'te): N=30 7.2×, N=300 3.9×. **Adil karşılaştırmada etki
büyüdü** ve N ile daralma imzası korundu (17× → 6×).

`rot_gap` değişmedi: eşdeğişken kollar 0.99–1.04, baseline 3.5–6.8.

**Kalan sınırlar:** 3e-3 tarama ızgarasının üst ucu; gerçek optimum daha
yukarıda olabilir, dolayısıyla oranlar alt sınır sayılmalı. N=30'da baseline'ın
optimumu iç noktada (3e-4), eşdeğişken kollarınki uçta — bu da 17×'i alt sınır
yapar. Ayrıca `fullgeo`'nun bir seed'i 3e-3'te ıraksadı (test ≈ persistence):
yüksek lr'de kararsızlık var, medyan bunu tolere ediyor, ortalama etmezdi.

Kazancın kaynağı geometrik *çarpım* değil, eşdeğişkenliğin kendisi: çarpım
katmanı olmayan, parametresi tam eşleşen `fullgeonb` her iki N'de de en iyi
sonucu verdi. Cebirin çarpım yapısı — projenin adındaki şey — hiçbir yerde
kendini ödetmedi.

Ve tahmin dışına çıkmıyor: en iyi lr'de bile rollout h100 ve enerji kaymasında
baseline önde (N=300: h100 0.45 vs 1.01, enerji 1.66 vs 4.64).

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

1. ~~LR taramasını bitir.~~ **Yapıldı, geçti** (§2.1).
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

Clifford **çarpımını state geçişine gömme** fikri — projenin adındaki iddia —
çürütüldü; hayatta kalan şey literatürün zaten bildiği reçetenin (uçtan uca
eşdeğişkenlik) yeni bir mimaride ölçülmüş ve doğrulanmış hâli: az veride
17×, çok veride 6× daha iyi tek-adım tahmini, ama uzun rollout ve korunumda
kazanç yok — ve asıl hedefe, gerçek veriye hiç varılmadı.

## 8. Son söz

Proje, sorduğu soruya dürüst bir cevap üretti ve cevap büyük ölçüde "hayır"dı.
Elde kalan tek pozitif, başlangıçtaki iddia değil onun komşusu; ve o bile
tahmin görevinde, sentetik veride, tek sistemde geçerli. Buna karşılık geriye
tekrar kullanılabilir bir mimari, bir tanı ölçütü (`rot_gap`) ve bir dizi
temiz negatif sonuç kaldı. On günde bir hipotezi bu kadar net gömebilmek
kötü bir sonuç değil; asıl maliyet, hedeften sapıp gerçek veriye hiç
gitmemiş olmamız.
