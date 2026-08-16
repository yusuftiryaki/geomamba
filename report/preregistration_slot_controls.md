# ÖN-KAYIT — Slotlama kontrolleri: kazanç geometrik mi?

**Yazılma tarihi: 16 Ağu 2026, koşumdan ÖNCE.** Karar kuralları aşağıda
sabittir; sonuçlara bakıp değiştirilmeyecek.

## Cevaplanan soru

Slotlama (girdiyi kendi grade'inin bıçaklarına yerleştirmek) D1 ve D2'de
tekrarlanabilir bir kazanç veriyor. Üç rakip açıklama var:

- **G (geometrik):** kazanç, bileşenlerin *birlikte dönen bir nesne* olarak
  tiplenmesinden geliyor.
- **K (kısıtlılık):** kazanç, gömmede öğrenilecek parametrenin 6× azalmasından
  geliyor; yapının kendisi önemsiz. (Serbest gömme 24·C, slot 4·C parametre.)
- **S (sabitlik):** kazanç, gömmenin öğrenilmek zorunda olmamasından geliyor.

Bu grid üçünü ayırır. Ayrılmazsa slotlama bulgusu "geometrik" olarak
sunulamaz.

## Kurulum

- Sistem: D1 rijit cisim, **N=300** (etkinin 3/3 seed tutarlı olduğu rejim).
- **5 seed** (0–4); mevcut 3 seed ayrımı taşımıyor.
- Bütçe: 10k adım, varsayılan lr — mevcut D1 koşularıyla birebir aynı.
- 5 gömme kolu × 2 mimari (mamba, rotor) × 5 seed = **50 koşu** (~2 sa, T4×2).

### Gömme kolları

| kod | ne yapar | trainable gömme param |
|---|---|---|
| `free` | serbest `Linear` (mevcut baseline) | 24·C |
| `slot2` | ω → grade-2 bıçakları (mevcut kazanan) | 4·C |
| `slot1` | ω → grade-1 bıçakları | 4·C |
| `scatter` | 3 bileşen → skaler + tek vektör bıçağı + pseudoskaler | 4·C |
| `frozen` | serbest `Linear`, init'te dondurulmuş (gömme + kafa) | 0 |

`scatter`, `slot`'la **aynı seyreklik ve aynı parametre sayısına** sahiptir;
tek farkı seçilen bıçakların birlikte dönmemesidir — skaler ve pseudoskaler
rotor sandviçi altında değişmez, tek vektör bıçağı ise tek başına döner.
Yani K ve S açıklamalarının sağladığı her avantajı sağlar, G'ninkini sağlamaz.
(Not: `scatter` üç bileşene üç *ayrı* grade skaleri verdiği için ifade gücü
`slot`'tan bir tık **fazladır**; bu, kontrolün lehine değil aleyhine çalışır.)

`slot1` bir "değişmemeli" öngörüsüdür: Cl(3,0)'da grade-1 ve grade-2 aynı
SO(3) temsilini taşır (dualite), rotor sandviçi ikisine de aynı dönmeyi
uygular. Teori doğruysa `slot1 ≈ slot2` olmalıdır.

### Ek ölçüm: eşdeğişkenlik boşluğu (rot_gap) — ek eğitim gerektirmez

D1'de atalet `np.sort` ile köşegen ve sıralı üretilir, yani eğitim dağılımının
kanonik bir çerçevesi vardır. Test yörüngelerine yörünge-başına rastgele bir
R ∈ SO(3) uygulanırsa ortaya çıkan dizi, I' = R I Rᵀ ataletli **geçerli** bir
rijit cisim yörüngesidir. Tam eşdeğişken bir model için hata **değişmez**.

    rot_gap = MSE(döndürülmüş test) / MSE(test)      ideal = 1.0

Her koşu için `test_mse` ve `rollout_mse_h100` üzerinden raporlanır. Bu,
"modelde ne kadar geometri var" sorusunun doğrudan ve nicel ölçümüdür.
Fiziksel kayma metrikleri döndürülmüş sette raporlanmaz (döndürülmüş atalet
köşegen olmadığı için mevcut 3-vektör arayüzüne sığmaz).

D2'de bu ölçüm anlamsızdır: B izotropik üretildiğinden döndürülmüş test seti
zaten dağılım içidir.

## Karar kuralları (önceden sabit)

Ölçüt: rollout h100, 5 seed ortalaması. "Anlamlı fark" = seed std'lerinin
birleşiminden büyük fark (eşleştirilmiş seed'lerde ≥4/5 aynı yön).

1. **`scatter` ≈ `slot2`** → kazanç geometrik DEĞİL (açıklama K/S).
   Slotlama bulgusu "yapılandırılmış seyrek gömme" olarak yeniden yazılır,
   geometri iddiası düşer.
2. **`frozen` ≈ `slot2`** → kazanç gömmeyi öğrenmemekten (açıklama S).
   Aynı sonuç: geometri iddiası düşer.
3. **`slot2` ≈ `slot1` ≫ `scatter` ve ≫ `frozen`** → kazanç grade yapısından.
   **G destekleniyor**; slotlama geometrik bir bulgu olarak sunulabilir.
4. **`slot1` ≉ `slot2`** → dualite argümanı yanlış; model grade'i değil bıçak
   konumunu kullanıyor. Teorik çerçeve gözden geçirilir, iddia yayınlanmaz.
5. **rot_gap:** `rotor+slot`'un rot_gap'i `mamba+slot`'unkinden anlamlı düşükse,
   rotor kısıtı doğrulukta değil **eşdeğişkenlikte** kendini gösteriyor
   demektir. Bu, rotoru "daha iyi tahminci" olarak kurtarmaz ama
   "daha geometrik model" iddiasını ayakta tutar ve ayrı bir bulgudur.
6. Tüm kollarda rot_gap ≫ 1 ve birbirine yakınsa, hiçbir varyantımız anlamlı
   ölçüde eşdeğişken değildir; bu durumda projenin geometri iddiası
   ölçüm düzeyinde de desteklenmiyor demektir.

## Bu grid'in cevaplamadıkları

- N=100 rejimi (etki orada tutarsızdı) — dahil edilmedi, bilinçli.
- Gürültülü/kısmi gözlem — ayrı ve henüz test edilmemiş rejim.
- Transfer — ayrı grid, adil bütçeyle tekrar edilecek.

## Uygulama notları

1. `GeoEmbed`/`GeoHead`: `grade` yerine açık bıçak indeksi listesi
   (`slot_blades`) kabul etsin; `slot_grade` bunun kısayolu olarak kalsın.
2. `ModelConfig.freeze_interface: bool` — gömme ve forecast kafasını dondurur.
3. `scripts/evaluate.py --rot-test` — test setine yörünge-başına rastgele
   SO(3) uygular, `*_rot` metriklerini ekler.
4. Birim testi (GPU'dan önce, tezgâhta): `slot2` gömmesinin eşdeğişken,
   `scatter` gömmesinin eşdeğişken **olmadığı** doğrulanmalı. Bu test
   geçmezse kontrol geçersizdir.
5. `tests/test_configs.py` yeni grid'i otomatik kapsar.
