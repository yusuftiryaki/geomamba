# (d) Tam Geometrik — SONUÇ (24 Ağu 2026)

100 koşu: 5 kol × N ∈ {10,30,100,300} × 5 seed, D1 rijit cisim, parametre
eşit (+%5.5 içinde). Eşdeğişkenlik eğitimden önce 1e-15'te doğrulanmıştı.

## 1. Eşdeğişkenlik: inşaat tuttu

`rot_gap` (döndürülmüş test / normal test, ideal 1.0), ortalama:

| kol | N=10 | N=30 | N=100 | N=300 |
|---|---|---|---|---|
| **fullgeo** | **1.08** | **1.02** | **1.02** | **0.99** |
| **fullgeonb** | **1.01** | **1.01** | **1.00** | **0.96** |
| mamba | 2.21 | 3.28 | 5.39 | 7.38 |
| mambaslot | 2.40 | 2.85 | 5.72 | 8.45 |
| rotorslot | 2.03 | 2.87 | 4.61 | 8.96 |

İki gözlem:
1. Eşdeğişken kollar her N'de tam olarak 1.0 — tasarım gerçekten çalışıyor.
2. **Eşdeğişken olmayan modeller veri arttıkça daha çerçeve-bağımlı hâle
   geliyor** (2.2 → 7.4). Daha çok veri, kanonik çerçeveyi daha sıkı
   ezberlemek demek. Bu, eşdeğişkenliğin mühendislik değerini gösteriyor:
   eğitimde görülmemiş bir montaj açısında baseline 8 kat bozuluyor,
   eşdeğişken model hiç bozulmuyor.

## 2. 1-adım tahmini: (d) ezici üstünlükte

Medyan 1-adım MSE:

| kol | N=10 | N=30 | N=100 | N=300 |
|---|---|---|---|---|
| **fullgeo** | 4.0e-4 | **5.2e-5** | **2.4e-5** | **1.9e-5** |
| fullgeonb | **1.5e-4** | 8.3e-5 | 3.1e-5 | 2.7e-5 |
| mamba | 2.9e-3 | 3.7e-4 | 1.2e-4 | 7.3e-5 |
| mambaslot | 5.2e-3 | 9.3e-4 | 1.5e-4 | 6.5e-5 |
| rotorslot | 3.6e-2 | 3.5e-3 | 3.1e-4 | 1.4e-4 |

Eşleştirilmiş seed düellosu: **fullgeo, her N'de 5/5 seed'de** mamba,
mambaslot ve rotorslot'u yeniyor. 20/20. Bu bugüne kadarki en temiz sinyal.

Veri verimliliği (medyan, mamba'nın N=300'üne yetişme):
- fullgeo N=30'da yetişiyor → **10× veri verimliliği**
- fullgeonb N=100'de → 3×
- rotorslot hiçbir N'de yetişemiyor

**Hipotezin öngördüğü şey birebir bu.** Ama devamını okuyun.

## 3. Uzun rollout ve korunum: (d) KAZANMIYOR

Eşleştirilmiş seed düellosu, fullgeo'nun mamba'yı yendiği seed sayısı:

| metrik | N=10 | N=30 | N=100 | N=300 |
|---|---|---|---|---|
| 1-adım | 5/5 | 5/5 | 5/5 | 5/5 |
| **rollout h100** | 2/5 | 1/5 | **0/5** | 3/5 |
| **enerji kayması** | 2/5 | 1/5 | **0/5** | 1/5 |

h100 medyanları N=300'de: mambaslot 0.479, rotorslot 0.501, mamba 0.718,
fullgeo 0.732. Enerji medyanı: mamba 1.57, fullgeo 6.19.

## 4. Bunu nasıl okumalı — kritik kısım

Dünkü ölçüm olmasa bu raporun başlığı "HİPOTEZ DOĞRULANDI, 20/20 seed,
10× veri verimliliği" olurdu. Ama 1-adım metriğinin ne ölçtüğünü biliyoruz:

| yöntem | 1-adım MSE |
|---|---|
| 4. mertebe sonlu fark (4 satır numpy) | 1.0e-9 |
| **fullgeo (en iyi modelimiz)** | **1.9e-5** |
| mamba | 7.3e-5 |

Yani 1-adım metriği **"ne kadar iyi bir yerel düzleştiricisin"** sorusunu
soruyor, "fiziği biliyor musun" sorusunu değil. Dört satırlık bir stencil
bizim en iyi modelimizden 20.000 kat iyi. Fiziği gerektiren metrik uzun
rollout (stencil orada h100'de 8.9–114000'e patlıyor, modeller 0.7 veriyor).

Dolayısıyla dürüst manşet:

> **Gerçekten eşdeğişken bir geometrik SSM kurduk, makine hassasiyetinde
> doğruladık, ve her seed'de 10× veri verimliliği kazandı — ama yalnızca
> fiziği bilmeyi gerektirmeyen metrikte. Fiziği gerektiren metrikte
> kazanmıyor, korunumda ise açıkça kaybediyor.**

Ek olarak: eğitim kaybı zaten 1-adım MSE. Yani fullgeo **eğitildiği hedefi
daha iyi optimize ediyor, gerçek göreve daha iyi genellemiyor.** Kısıtlı
mimari daha iyi koşullanmış bir optimizasyon problemi veriyor — bu gerçek
bir kazanç ama fizik kazancı değil, kondisyon kazancı. Literatürün
(RotRNN, Mamba-3, DeltaProduct, Versor) bulduğu şeyle aynı kategoride.

## 5. Geometrik çarpım ablasyonu: neredeyse hiçbir şey

fullgeo (çarpım var) vs fullgeonb (yok), fullgeo'nun kazandığı seed sayısı:

| metrik | N=10 | N=30 | N=100 | N=300 |
|---|---|---|---|---|
| 1-adım | 1/5 | 4/5 | 3/5 | 4/5 |
| h100 | 3/5 | 3/5 | 2/5 | 3/5 |
| enerji | 2/5 | 4/5 | 3/5 | **5/5** |

Cebrin "asıl kası" dediğim geometrik çarpım, N≥30'da 1-adımda ve enerjide
küçük ama tutarlı bir kazanç veriyor, rollout'ta hiçbir şey vermiyor.
**Kazandıran şey çarpım değil, eşdeğişken yapının kendisi.** En egzotik
bileşen çıkarılabilir.

## 6. Eşdeğişkenlik ≠ korunum — artık ampirik kanıt var

`tests/test_invariance.py` bu ayrımı bir test olarak yazmıştı. Şimdi
verisi de var: **en eşdeğişken modelimiz en kötü enerji korunumuna sahip**
(fullgeo enerji medyanı N=300'de 6.19, mamba 1.57). Simetriden korunum
çıkmıyor; onun için ayrı yapı (varyasyonel/simplektik) gerekiyor.

## 7. Hayatta kalan iddialar

| iddia | durum |
|---|---|
| Rotor kısıtı (c) yardım ediyor | ÇÜRÜTÜLDÜ |
| Slotlama geometrik olduğu için yardım ediyor | ÇÜRÜTÜLDÜ (scatter kontrolü) |
| Tam eşdeğişken mimari **tahmin doğruluğunda** yardım ediyor | **DESTEKLENDİ** — 20/20 seed, 10× veri verimliliği |
| ...ama bu kazanç **fiziksel dinamikten** geliyor | **DESTEKLENMEDİ** — rollout 6/20, enerji 4/20 |
| Geometrik çarpım gerekli | zayıf/gereksiz |
| Eşdeğişkenlik korunum getirir | ÇÜRÜTÜLDÜ (ampirik) |
| **Eşdeğişkenlik görülmemiş yönelime genelleme getirir** | **DESTEKLENDİ** — rot_gap 1.0 vs 7–9 |

## 8. Sonraki adım

Asıl açık soru artık şu: 1-adım kaybı trivial düzleştirmeyle domine
edildiği için, mimari farkları gerçek dinamiğe hiç ulaşamıyor olabilir.
Somut düzeltme: **artık tahmini** — `Δ_t = stencil(geçmiş) + model(geçmiş)`.
Trivial kısım bedavaya çözülür, gradyanın tamamı gerçek dinamiğe gider.
Bu yapılmadan "geometrik önsel fiziği öğrenmeye yardım etmiyor" sonucu
kesinleşmiş sayılmaz — çünkü model fiziği öğrenmeye hiç zorlanmadı.
