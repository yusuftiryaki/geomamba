# (d) Tam geometrik — SONUÇ (17 Ağu 2026)

100 koşu: 5 kol × N ∈ {10,30,100,300} × 5 seed, D1 rijit cisim.
Eşdeğişkenlik eğitimden önce doğrulanmıştı (1e-15); burada eğitilmiş
modellerde de ölçüldü.

## 1. Eğitim hedefinde (1-adım MSE) büyük ve tutarlı kazanç

Medyan, 5 seed. Persistence tabanı **1.11e-3**.

| N | mamba | mambaslot | rotorslot | **fullgeonb** | **fullgeo** |
|---|---|---|---|---|---|
| 10 | 2.9e-3 | 5.2e-3 | 3.6e-2 | **1.5e-4** | 4.0e-4 |
| 30 | 3.7e-4 | 9.3e-4 | 3.5e-3 | 8.3e-5 | **5.2e-5** |
| 100 | 1.2e-4 | 1.5e-4 | 3.1e-4 | 3.1e-5 | **2.4e-5** |
| 300 | 7.3e-5 | 6.5e-5 | 1.4e-4 | 2.7e-5 | **1.9e-5** |

Eşleştirilmiş seed: `fullgeo < mamba` ve `fullgeonb < mamba` **her N'de 5/5**.

**Kazanç oranı N ile daralıyor: 20× → 7× → 5× → 3.9×.** Bu, orijinal planda
önceden yazılmış inductive-bias imzasının ta kendisi: bias az veride çok
kazandırır, veri arttıkça fark kapanır.

**N=10'da baseline'lar persistence'ı bile geçemiyor** (mamba 2.9e-3 vs
persistence 1.1e-3 — 2.6× daha kötü; rotorslot 32× daha kötü), eşdeğişken
modeller ise 3–8× geçiyor. Yani kritik rejimde baseline hiçbir şey
öğrenemezken eşdeğişken model öğreniyor.

Veri verimliliği: `fullgeo` N=30'da (5.2e-5), `mamba`'nın N=300'de aldığı
sonucu (7.3e-5) geçiyor — **~10× daha az veri**.

## 2. Eşdeğişkenlik eğitilmiş modelde de duruyor

`rot_gap` ortalaması (ideal 1.0, normalizasyon tabanı 1.07):

| N | mamba | mambaslot | rotorslot | fullgeonb | fullgeo |
|---|---|---|---|---|---|
| 10 | 2.21 | 2.40 | 2.03 | **1.01** | **1.08** |
| 30 | 3.28 | 2.85 | 2.87 | **1.01** | **1.02** |
| 100 | 5.39 | 5.72 | 4.61 | **1.00** | **1.02** |
| 300 | 7.38 | 8.45 | 8.96 | **0.96** | **0.99** |

Ayrıca dikkat: baseline'ların rot_gap'i **veriyle birlikte büyüyor** (2.2 → 7.4).
Veri arttıkça kanonik çerçeveyi daha çok ezberliyorlar. Eşdeğişken modellerde
N ne olursa olsun 1.0.

## 3. Kazanç eşdeğişkenlikten, geometrik çarpımdan değil

`fullgeonb` = aynı eşdeğişken boru hattı, **geometrik çarpım katmanı yok**
(ve parametre eşleşmesi neredeyse tam: 128.5k vs mamba 128.2k).
1-adım MSE'de `fullgeo`'ya çok yakın, N=10'da daha iyi.
`fullgeo < fullgeonb` düellosu: 1/5, 4/5, 3/5, 4/5.

Yani veri verimliliğini satın alan şey **eşdeğişkenliğin kendisi**; geometrik
çarpım N≥30'da küçük bir ek katkı yapıyor, ana etken değil.

## 4. Kazanmadığımız yer — dürüstlük

**Rollout h100 (medyan):** eşdeğişken modeller daha iyi DEĞİL.

| N | mamba | mambaslot | rotorslot | fullgeonb | fullgeo |
|---|---|---|---|---|---|
| 10 | **1.05** | 1.48 | 1.97 | 3.04 | 2.34 |
| 30 | 1.12 | 1.20 | 1.24 | 2.94 | 1.22 |
| 100 | 0.71 | 0.77 | **0.65** | 1.01 | 1.03 |
| 300 | 0.72 | **0.48** | 0.50 | 0.69 | 0.73 |

**Enerji kayması (medyan):** eşdeğişken modeller her N'de daha KÖTÜ
(N=300: 6.19 / 28.6 vs mamba 1.56).

Bu çelişki değil, beklenen sonuç ve `tests/test_invariance.py` bunu zaten
belgeliyor: **grup simetrisi korunum yasası vermez.** Mimaride enerjiyi
koruyan hiçbir yapı yok; eşdeğişkenlik yerel vektör alanını öğrenmeyi
kolaylaştırır, 100 adımlık otoregresif hata birikimini kontrol etmez.

N=10/30'da rollout'un baseline'da daha "iyi" görünmesinin muhtemel sebebi de
şu: baseline persistence'tan kötü, yani neredeyse hiç hareket etmiyor —
durağan bir model ıraksamaz da. Eşdeğişken model dinamiği gerçekten takip
ediyor ve hata ettiğinde aktif hata ediyor.

## 5. Geçerliliğe yönelik tehditler

1. **Kol başına lr taraması yapılmadı.** Tek hiperparametre ayarı; eşdeğişken
   modeller varsayılan lr'yi daha çok seviyor olabilir. En ciddi tehdit bu.
2. `fullgeo` +%5.5 parametre fazlası taşıyor — ama `fullgeonb` +%0.3 ve etki
   onda da tam olarak var, dolayısıyla açıklama bu değil.
3. Hesap avantajı yok: `full_geo` CPU'da baseline'dan **hızlı** (0.61 vs 0.89 s/adım).
4. Tek sistem (D1). D2/D4'te tekrarlanmadı.

## 6. Hüküm

Projenin çekirdek hipotezi — *geometrik yapıyı inductive bias olarak koymak
fiziksel dinamikleri daha veri-verimli öğretir* — **tahmin hedefinde
DESTEKLENDİ**, ve önceden yazılmış imzayla (fark az veride büyük, çok veride
daralıyor). Ama bu ancak mimari **uçtan uca eşdeğişken** olduğunda gerçekleşti;
kısmi geometri (rotor kısıtı + serbest kapı) her turda başarısız oldu.

İddianın ikinci ayağı — *daha fiziksel tutarlı* — **DESTEKLENMEDİ.**
Korunum simetriden gelmiyor; ayrıca inşa edilmesi gerekiyor.

## 7. Sıradaki

1. **lr taraması** — kol başına, tehdit #1'i kapatmak için. Bunsuz sonuç
   yayınlanamaz.
2. **D2 parçacıkta tekrar** — etki sisteme özgü mü, genel mi.
3. **Korunum yapısı** — rollout ve enerji için ayrı bir bileşen
   (simplektik/korunumlu integratör ya da invaryant-cezalı eğitim).
4. Öğrenilen tipleme + öğrenilen simetri gücü (α): model kısıtı kendisi
   seçiyor mu?
