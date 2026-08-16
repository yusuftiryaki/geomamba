# Teşhis Grid'i — Enerji Tabanı ve Adil Baseline (16 Ağu 2026)

11 koşu, D1 rijit cisim, N=300. Ham veri:
`experiments/kaggle/grid_results_diagnosis.csv`. Karşılaştırma referansları
karar grid'inden (`grid_results_rb.csv`).

| konfig | enerji kayması | rollout h50 | rollout h100 |
|---|---|---|---|
| rotor λ=-12 (fiilen korunumlu) | 6.47 ±0.77 | 0.39 ±0.14 | 1.33 ±0.52 |
| rotor λ=-6 | 6.97 ±0.91 | 0.38 ±0.12 | 1.33 ±0.48 |
| **rotor λ=-3 (varsayılan)** | **3.98 ±0.83** | **0.21 ±0.02** | **0.67 ±0.05** |
| rotor λ=-1 (güçlü sönüm) | 3.87 ±2.52 | 0.21 ±0.10 | 0.71 ±0.32 |
| rotor dt_max=0.05 | 3.96 | 0.30 | 0.92 |
| rotor dt_max=0.2 | 2.68 | 0.25 | 0.61 |
| **mamba72 (eşit param)** | 3.19 ±1.93 | 0.25 ±0.04 | 0.72 ±0.06 |
| mamba64 (karar grid) | 3.25 ±2.39 | 0.38 ±0.32 | 1.11 ±1.00 |

## Bulgular

**1. Enerji tabanının kaynağı sönüm DEĞİL — tam tersi.** Sönümü fiilen
kapatmak (λ=-12, adım başına decay ~6e-6) kaymayı azaltmadı, **artırdı**
(3.98 → 6.47) ve rollout'u da bozdu (h100 0.67 → 1.33). λ=-6 aynı.
Yani "sönüm faktörü modeli yanlış yönlendiriyor" hipotezi çürüdü.
Yorum: sönüm burada fizik olarak "enerji kaybı" değil, **rollout stabilizatörü**
işlevi görüyor — hafif kasılma, hataların üstel birikimini bastırıyor.
Tamamen korunumlu geçiş, hata biriktiğinde onu da sonsuza taşıyor.

**2. Rotasyon adımı ölçeği (dt_max) hafif ama doğru yönde etkili.**
dt_max=0.2 tek seed'de en iyi enerji (2.68) ve h100 (0.61) — daha büyük
dönme adımına izin vermek yardımcı. Tek seed, teyit gerek; ama λ'dan daha
umut verici bir ayar ekseni.

**3. Parametre eşitlenince baseline'ın avantajı ERİDİ (kritik bulgu).**
mamba72 (125.6k ≈ rotor 129.4k) vs rotor λ=-3:
- enerji: 3.19 ±1.93 vs 3.98 ±0.83 → **aralıklar tamamen örtüşük, fark yok**
- h50: 0.25 ±0.04 vs **0.21 ±0.02**, h100: 0.72 ±0.06 vs **0.67 ±0.05** →
  rotor hafif önde, std'ler yine rotorda daha küçük
Karar grid'inde mamba64'ün enerjide "kazanması" (3.25) ile mamba72 (3.19)
aynı yerde; yani baseline'ı büyütmek enerjiyi düzeltmiyor — ama rotorun
rollout üstünlüğü eşit parametrede de duruyor.

**4. Varyans bulgusu güçlendi.** mamba72 h100 std 0.06 — mamba64'ün 1.00'ından
çok daha iyi, yani karar grid'indeki "felaket seed" kısmen küçük-model
etkisiymiş. Ama enerji kaymasında mamba72 std'si hâlâ 1.93 vs rotor 0.83;
h50'de 0.04 vs 0.02. Rotor tutarlılık avantajını eşit parametrede de koruyor,
ancak 14× değil ~2× seviyesinde. Dürüst düzeltme: önceki "14× varyans
çöküşü" rakamı parametre-eşitsiz kıyastan geliyordu.

## Güncellenmiş okuma

- Rotorun enerji kayması "sistematik bir kusur" değil; **eşit parametrede
  baseline ile aynı seviyede**. Yani karar grid'inin "konservasyon bacağı
  başarısız" hükmü, adil kıyasta "fark yok"a dönüşüyor — hipotez lehine
  değil ama aleyhine de değil.
- Rotorun tutarlı (küçük varyanslı) uzun-ufuk rollout üstünlüğü, eşit
  parametrede de görülen en dayanıklı sinyal.
- λ ekseni kapandı (varsayılan -3 iyi); dt_max ekseni açık.

## Sonraki adımlar

1. dt_max ∈ {0.2, 0.4} × 3 seed teyidi (ucuz).
2. Skorlama v2 grid'i (m3_scoring) — anomali tarafı hâlâ ölçülmedi.
3. Kontrol grid'i (Lorenz negatif kontrol + parçacık).
4. Slotlamalı GeoEmbed: temsil hipotezinin testi (hâlâ yapılmadı).
5. Tüm ileri kıyaslar mamba72 (eşit param) baseline'ıyla yapılmalı.
