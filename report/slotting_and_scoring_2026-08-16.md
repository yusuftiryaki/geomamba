# Slotlama + Skorlama v2 — Dönüm Noktası (16 Ağu 2026)

İki grid: temsil ablasyonu (m3_slotting) ve anomali skorlama v2 (m3_scoring).
Hepsi D1 rijit cisim, N=300, 3 seed, parametre eşit (±%3).

## 1. Slotlama: temsil hipotezi DOĞRULANDI

| varyant | test MSE | rollout h50 | rollout h100 | enerji kayması |
|---|---|---|---|---|
| **slot** (ω → grade-2, 128.9k) | 1.2e-4 | **0.171 ±0.047** | **0.477 ±0.120** | **2.63 ±1.78** |
| rotor (serbest embed, 129.4k) | 9.2e-5 | 0.210 ±0.023 | 0.672 ±0.052 | 3.98 ±0.83 |
| mamba76 (eşit param, 128.1k) | 7.8e-5 | 0.243 ±0.093 | 0.767 ±0.335 | 4.12 ±4.79 |
| fullgeo (+CliffordLinear, 132.6k) | 7.6e-4 | 0.301 ±0.049 | 0.888 ±0.189 | 4.88 ±1.26 |

**İlk kez bir geo varyant hem doğrulukta hem korunumda eşit-parametreli
baseline'ı geçti.** Slotlama, serbest embedding'li rotora göre h100'de %29,
baseline'a göre %38 iyileşme; enerji kaymasında da en iyi ortalama.

Bu, kontrol grid'inin teşhisini doğruluyor: kısıt gizli state'te, metrik
gözlem uzayındaydı; ikisi arasındaki bağı serbest embedding koparıyordu.
Girdiyi doğru grade yuvasına koymak bağı kuruyor.

**Ama fullgeo (Clifford-linear kanal karışımı) BOZDU:** 1-adım MSE 10× kötü
(7.6e-4 vs 1.2e-4), rollout da kötü. Grade-koruyucu karışım, aynı parametre
bütçesinde çok daha az ifade gücü veriyor (grade başına tek matris) ve
görünüşe göre SiLU gate ile de uyumsuz. Yani "daha çok geometri her zaman
daha iyi" değil — **doğru yere geometri.**

Kısıt: enerji kaymasında std'ler büyük (slot ±1.78), 3 seed yetersiz.
Doğrulama için seed sayısı artırılmalı ve N=100 hücresi eklenmeli.

## 2. Skorlama v2: model-siz geometrik izleyici modelleri EZDİ

D1 anomali tespiti (atalet sıçraması), aynı test seti:

| yöntem | AUROC | olay-F1 |
|---|---|---|
| **geometrik invaryant izleyici (model-siz)** | **0.996** | **0.806** (P 0.69, R 0.96) |
| CUSUM (model residual'ı üzerinde) | 0.58-0.71 | ≤0.16 |
| iki-yönlü residual | 0.51-0.58 | — |
| rejim-dondurmalı rollout | 0.48-0.58 | ~0 |
| ham residual (v1) | 0.44-0.49 | — |

Şüpheci kontrol — naif değişim dedektörleri aynı veride:

| naif yöntem | AUROC | olay-F1 |
|---|---|---|
| \|ω\| sapması | 0.616 | 0.140 |
| bileşen z-skoru | 0.660 | 0.140 |
| kayan varyans değişimi | 0.651 | 0.195 |
| \|ω\| üzerinde CUSUM | 0.661 | 0.123 |

Fark geometrik yapıdan geliyor, trivial bir sıçrama tespitinden değil.
Kalibrasyon penceresi (0-63. adım) anomali başlangıçlarından (≥64) temiz —
sızıntı yok.

**İzleyicinin sınırları (dürüstlük):**

| sistem | AUROC | neden |
|---|---|---|
| D1 rijit cisim (atalet sıçraması) | 0.996 | anomali izlenen invaryantı bozuyor |
| D2 parçacık (alan ters dönmesi) | 0.514 | **anomali \|v\|'yi bozmuyor** — Boris dinamiğinde alan ters dönse de hız normu korunur; izleyici doğru olarak bir şey bulmuyor |
| D4 Lorenz (ρ değişimi) | 0.768 | korunum yok; kuadratik formlar yine de kayıyor, ama yanlış alarm yüksek (P 0.31) |

Yani izleyici evrensel bir dedektör değil: **izlenen invaryantı bozan
anomalileri** yakalıyor. Bu bir kusur değil, tanım — ve kullanım için net
bir reçete veriyor: "hangi büyüklük korunmalı?" sorusunu cevaplayabildiğin
her sistemde eğitimsiz, neredeyse mükemmel bir dedektör kurulabiliyor.

## 3. Tezin güncellenmiş hali

Veriler üç ayrı iddiayı ayrıştırdı:

1. **Temsil (geometrik tipleme):** DESTEKLENDİ — slotlama hem doğruluğu hem
   korunumu iyileştirdi, eşit parametrede baseline'ı geçti.
2. **Geçiş kısıtı (rotor):** TEK BAŞINA YETERSİZ — serbest embedding'le
   çalışmıyor (kontrol grid'i, D2'de net olumsuz), doğru temsille birlikte
   çalışıyor. Kısıtın işe yaraması temsile bağımlı.
3. **Çarpım/karışım geometrisi (Clifford-linear):** ŞU HALİYLE ZARARLI —
   ifade gücünü çok kısıyor.
4. **Bonus, planlanmamış:** geometrik invaryant izleme, öğrenilmiş anomali
   skorlamasından bir kategori daha iyi (0.996 vs 0.55-0.71) ve eğitim
   gerektirmiyor.

## 4. Sonraki adımlar

1. **Slotlama teyidi:** N ∈ {100, 300, 1000} × 5 seed × {slot, rotor,
   mamba76}; slotlamayı vanilla mamba'ya da ekle (slot ⊗ rotor ayrıştırması:
   kazanç slotlamadan mı, rotor+slotlama etkileşiminden mi?).
2. **D2 parçacıkta slotlama** (orada slot_grade=1, hız bir vektör) — kontrol
   grid'inin olumsuz sonucu slotlamayla dönüyor mu?
3. fullgeo'yu onar veya düşür (gate'i grade-bazlı norm-gate yap).
4. İnvaryant izleyiciyi ürünleştir: Entes verisinde karşılığı üç fazlı
   sistemde korunan büyüklükler (dengeli sistemde |V_αβ|, güç formları).
