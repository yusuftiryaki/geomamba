# Karar Grid'i Analizi — D1 Rijit Cisim (16 Ağu 2026, Kaggle T4)

24 koşu: 3 varyant × N ∈ {100,300,1000,3000}; N=100/300 hücrelerinde 3 seed.
Tam bütçe (d_model 64, 10k adım). Ham veri: `experiments/kaggle/grid_results_rb.csv`.

## 3-seed hücreleri (ortalama ± std)

**N=300 (en güvenilir hücre):**

| metrik | mamba (a) | mv (b) | rotor (c) |
|---|---|---|---|
| 1-adım test MSE | **8.0e-5** ±2.2e-5 | 1.3e-4 ±4.4e-5 | 9.2e-5 ±**9.7e-6** |
| rollout h50 | 0.375 ±0.32 | 0.302 ±0.092 | **0.210 ±0.023** |
| rollout h100 | 1.11 ±1.0 | 0.781 ±0.12 | **0.672 ±0.052** |
| enerji kayması | 3.26 ±2.4 | **2.89** ±0.72 | 3.98 ±0.83 |
| momentum kayması | 0.843 ±0.56 | **0.719** ±0.097 | 0.892 ±0.21 |
| anomali AUROC (2y) | **0.573** ±0.011 | 0.549 ±0.026 | 0.545 ±0.012 |

**N=100:** rotor h50 0.318±0.087 vs mamba 0.358±0.16; enerji rotor 4.43±1.1 vs
mamba 3.27±**3.3**; anomali hepsi ~0.58-0.60 (ayrışmıyor).
**N=1000/3000 (tek seed, temkinli):** mamba h100 ve enerji önde.

## Önceden kayıtlı kritere göre hüküm

Kriter: geo, N∈{100,300}'de HEM doğrulukta HEM konservasyonda (a)'yı geçmeli;
fark N=3000'de daralmalı; D4 Lorenz'de avantaj olmamalı (kontrol grid'i beklemede).

- **Doğruluk (rollout): KISMEN SAĞLANDI.** Rotor küçük N'de h50/h100
  ortalamalarında önde ve fark N≥1000'de kayboluyor (bias imzası). Ancak
  üstünlük büyük ölçüde mamba'nın felaket-seed'lerinden geliyor (seed-bazlı
  ikili kıyasta rotor 2/3 kazanıyor); 1-adım MSE'de mamba hâlâ önde.
- **Konservasyon: SAĞLANMADI.** Rotorun enerji kayması hiçbir hücrede
  mamba'dan iyi değil (aralıklar örtüşüyor, ortalama aleyhte). En güçlü
  yapısal iddiamız bu grid'de doğrulanmadı.
- **Resmî sonuç: hipotez, kayıtlı haliyle bu grid'de DESTEKLENMEDİ.**

## Sağlam ikincil bulgular

1. **Varyans çöküşü (asıl manşet):** Rotor kısıtı seed varyansını dramatik
   düşürüyor: h50 std 0.023 vs mamba 0.32 (14×); h100 en-kötü-seed rotor 0.71
   vs mamba 2.26. Vanilla ara sıra felaket üretiyor; rotor hiç üretmiyor.
   "Ortalama üstünlük" değil, **güvenilirlik** kazancı — endüstriyel değer
   açısından ciddi ve rapor edilebilir bir bulgu.
2. **Rotor enerji kayması sistematik bir tabanda (~3.5-4.5), N'den ve
   seed'den bağımsız.** Gürültü değil — teşhis edilebilir tek kaynak imzası
   (şüpheliler: λ sönüm kalibrasyonu, serbest embedding'in invaryantı yanlış
   yuvalara dağıtması, Δ ölçeği). Mamba'nınki 0.5-7.0 arasında savruluyor.
3. **Sondadaki mv-anomali sinyali seed'lerde TEKRARLAMADI** (N=100:
   0.598±0.032, mamba 0.580±0.023 ile örtüşük). Düşürüldü.
4. **Ablasyon okuması:** rotor > mv (rollout tutarlılığında) → kazancın
   kaynağı yapı değil kısıt; mv 1-adımda en kötü ve erken checkpoint'lerde
   tepe yapıyor (ckpt 4000-9000; aşırı-uyum eğilimi).

## Bilinen kısıtlar / düzeltilecekler

- Parametre eşitsizliği: 92k / 129k / 166k (plan ±%5 derdi) — geo aleyhine
  değil lehine bir kayırma olabilirdi, yine de eşitlenecek.
- Anomali tespiti tüm varyantlarda zayıf (AUROC ≤ 0.6, gecikme ~ tavan) —
  skorlama yöntemi (CUSUM benzeri birikimli skor) ele alınmalı.
- D4 Lorenz negatif kontrolü ve D2 parçacık genellemesi: 2. hesapta koşuyor.

## Sonraki adımlar (öncelik sırasıyla)

1. Kontrol grid'i sonuçları → kriterin kalan maddeleri.
2. **Teşhis turu:** rotor state normu + öğrenilen λ/θ dağılımları loglanarak
   enerji tabanının kaynağı bulunacak; λ-init taraması ucuz (Kaggle).
3. **Slotlamalı GeoEmbed ablasyonu** (ω → grade-2): temsil hipotezinin testi.
4. Parametre eşitleme + karar hücrelerinin yeniden koşusu.
5. Sonuçlara göre (d) full-geometric ve/veya "güvenilirlik" ekseninin
   (varyans bulgusu) ana iddiaya terfisi.
