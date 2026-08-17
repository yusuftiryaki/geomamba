# Slotlama kontrolleri — SONUÇ (17 Ağu 2026)

Ön-kayıt: `report/preregistration_slot_controls.md` (koşumdan önce yazıldı).
50 koşu tamam: 5 gömme kolu × 2 mimari × 5 seed, D1 rijit cisim, N=300.

## Rakamlar (rollout h100, ortalama ± std, parantezde medyan)

| kol | mamba | rotor |
|---|---|---|
| free | 0.805 ±0.283 (0.718) | 0.742 ±0.323 (0.659) |
| **slot2** | **0.571 ±0.189 (0.479)** | 0.683 ±0.526 (0.501) |
| slot1 | 0.555 ±0.139 (0.543) | **0.535 ±0.234 (0.464)** |
| **scatter** | **0.658 ±0.129 (0.694)** | **0.557 ±0.265 (0.460)** |
| frozen | 1.176 ±1.01 (0.655) | 0.622 ±0.234 (0.515) |

Eşleştirilmiş seed düelloları (5 seed):

| karşılaştırma | mamba | rotor |
|---|---|---|
| slot2 < free | 4/5 | 2/5 |
| **slot2 < scatter** | **3/5** | **3/5** |
| slot2 < frozen | 4/5 | 3/5 |
| slot2 < slot1 | 2/5 | 3/5 |

## Hüküm: KURAL 1 ATEŞLEDİ — kazanç geometrik değil

**`scatter`, `slot2`'ye yetişti.** Ön-kayıtta yazılan kural buydu:

> 1. `scatter` ≈ `slot2` → kazanç geometrik DEĞİL (açıklama K/S).
>    Slotlama bulgusu "yapılandırılmış seyrek gömme" olarak yeniden yazılır,
>    geometri iddiası düşer.

`scatter`, `slot2` ile **birebir aynı parametre sayısına ve seyrekliğe** sahip
ama seçtiği bıçaklar birlikte dönmüyor — yani geometrik bir nesne değil.
Buna rağmen mamba'da fark seed gürültüsünün içinde (3/5, %13), rotorda ise
**scatter daha iyi** (0.557 vs 0.683). Geometrik tipleme, geometrik olmayan
aynı-maliyetli bir yerleştirmeden ayırt edilemiyor.

Dolayısıyla slotlama kazancının kaynağı **grade yapısı değil**, gömmedeki
serbestlik derecesinin azalması (24·C → 4·C parametre).

**Yan hüküm — teorinin doğru çıkan tek parçası:** `slot1 ≈ slot2` öngörüsü
tuttu (mamba 0.555 vs 0.571; rotor 0.535 vs 0.683, ikisi de seed gürültüsü
içinde). Cl(3,0)'da grade-1 ve grade-2'nin aynı SO(3) temsilini taşıdığı,
dolayısıyla hangisine slotladığımızın fark etmemesi gerektiği doğrulandı.
Ama bu, mimarinin geometriyi *kullandığını* değil, iki seçeneğin denk
olduğunu gösteriyor.

## KURAL 6 DE ATEŞLEDİ — hiçbir varyant eşdeğişken değil

`rot_gap` (döndürülmüş test MSE / normal test MSE, ideal 1.0):

| kol | mamba | rotor |
|---|---|---|
| free | 7.38 | 8.48 |
| slot2 | **8.45** | **8.96** |
| slot1 | 7.70 | 9.44 |
| scatter | 7.64 | 8.40 |
| frozen | 6.78 | 7.33 |

Normalizasyon tabanı (persistence): **1.07**. Yani taban ihmal edilebilir ve
gözlenen 7–9 kat tamamen modelden geliyor.

Ön-kayıt: *"Tüm kollarda rot_gap ≫ 1 ve birbirine yakınsa, hiçbir
varyantımız anlamlı ölçüde eşdeğişken değildir; bu durumda projenin geometri
iddiası ölçüm düzeyinde de desteklenmiyor demektir."*

Tam olarak bu oldu. Üstelik **slotlama modeli daha da az dönme-dayanıklı
yapıyor** (her iki mimaride de en yüksek rot_gap slot2'de) — geometrik
hikâyenin tersi.

## Üçüncü bulgu: 3 seed bizi yanılttı

Slotlama raporundaki "h100 0.477 vs 0.767" tablosu 3 seed'liydi. 5 seed'le:
mamba slot2 0.571 ±0.189 vs free 0.805 ±0.283 — işaret aynı ama 4/5 seed
(tek yönlü işaret testi p≈0.19, anlamlı değil) ve std'ler örtüşüyor. Rotorda
etki tamamen kayboluyor (2/5). Enerji kaymasında rotor+slot bir seed'de
176.7 veriyor — kararsız.

Yani "slotlama kazandırıyor" iddiası bile, kontrolleri bir kenara bıraksak
dahi, 5 seed'de zayıf.

## Bunun ardından geriye ne kaldı

| iddia | durum |
|---|---|
| Rotor state kısıtı yardım ediyor | ÇÜRÜTÜLDÜ (önceki gridler + burada 2/5) |
| Slotlama **geometrik** olduğu için yardım ediyor | **ÇÜRÜTÜLDÜ** — scatter yetişiyor |
| Slotlama (yapı fark etmeksizin) yardım ediyor | zayıf, mamba'da 4/5, rotorda yok |
| Mimarilerimiz geometriyi kullanıyor | **ÇÜRÜTÜLDÜ** — rot_gap 7–9, taban 1.07 |
| grade-1 ≡ grade-2 (dualite) | doğrulandı |
| **Gerçek eşdeğişken mimari (d) yardım ediyor** | **TEST EDİLMEDİ** |

Son satır artık tek açık soru — ve bu grid onu daha da önemli hale getirdi:
rot_gap ölçümü, eşdeğişkenliğin bugüne kadar hiçbir kolumuzda **var olmadığını**
gösterdi. `variant: full_geo` eşdeğişkenliği 1e-15'te doğrulanmış tek
mimarimiz; `rot_gap ≈ 1.0` vermesi gereken tek kol o. Eğer o da kazandırmazsa,
geometrik inductive bias iddiası bu problem sınıfı için kapanmıştır.
