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

**~~Ama fullgeo (Clifford-linear kanal karışımı) BOZDU~~ — BU SONUÇ GEÇERSİZ.**

Orijinal metin: *"1-adım MSE 10× kötü (7.6e-4 vs 1.2e-4)… daha çok geometri
her zaman daha iyi değil."* İki hata var:

1. **Rakam yanlış:** 10× değil **6.3×** (7.6e-4 / 1.2e-4). h100'de 1.9×.
2. **Daha ciddisi, o kol "full geometric" değildi.** `geo_proj` yalnızca
   giriş/çıkış projeksiyonlarını Clifford-lineer yapıyor. Eleman-bazlı
   `u * silu(z)` kapısı, bıçak-başına ayrı filtreleyen depthwise conv ve
   düzleştirilmiş bıçaklar üzerinde çalışan SSM projeksiyonları **olduğu gibi
   kalıyordu** — ve planın (d) basamağındaki asıl bileşen, geometrik-çarpım
   bilineer katmanı, hiç yazılmamıştı.

Ölçtük (`tests/test_equivariance.py`, girdiyi döndür–çıktı dönüyor mu):

| model | eşdeğişkenlik hatası (göreli) |
|---|---|
| gerçek (d) `full_geo` | **3e-16 … 1e-15** |
| eski "fullgeo" (`geo_proj`) | 0.93 |
| rotor+slot (c) | 0.28 |
| vanilla mamba | 1.08 |

Eski kol eşdeğişkenlik bakımından vanilla Mamba'dan neredeyse ayırt edilemez.
Yani ölçtüğümüz şey "geometrinin bedeli" değil, **kısıtlı lineer katmanların
serbest bir kapıyla kavga etmesinin** bedeliydi — iki dünyanın da kötüsü.
Bu satır, "Clifford karışımı zararlı" iddiasının dayanağı olarak
**kullanılamaz**; gerçek (d) ayrıca test edilmelidir.

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

### DÜZELTME (aynı gün, şüpheci inceleme sonrası)

İlk yazımdaki "eğitimsiz, neredeyse kusursuz dedektör" çerçevesi **fazla
iddialıydı**. Üç düzeltme:

**1. "Eğitimsiz" değil, "kalibrasyonlu".** İzleyici normali her trajektorinin
ilk %25'inden (64 adım) öğreniyor. Eğitim veri seti ve gradyan gerektirmiyor,
ama anomalisiz bir referans pencere şart. Normali bilmeden anormali bilmiyor.

**2. 0.996 rakamı sentetik verinin artefaktı.** Skor dağılımına bakınca
görüldü: normal dönemde ~1.9, anomali sonrası ~10^6. Sebep, ground-truth'ta
korunan büyüklüğün integratör hassasiyetinde (1e-10) korunması — normal bandı
absürt dar, her sapma milyonlarca sigma. Gerçekçi gürültüde çöküyor:

| gözlem gürültüsü | AUROC | olay-F1 | precision | recall |
|---|---|---|---|---|
| %0 | 0.996 | 0.806 | 0.69 | 0.96 |
| %0.5 | 0.865 | 0.706 | 0.73 | 0.68 |
| %1 | 0.816 | 0.592 | 0.76 | 0.53 |
| %5 | 0.671 | 0.175 | 0.43 | 0.15 |

**3. Precision'ı da vermeliydim:** gürültüsüzde bile 0.69 — alarmların ~%31'i
yanlış. "Neredeyse kusursuz" yanlış tarifti.

**Geriye kalan gerçek bulgu (daha mütevazı ama sağlam):** gürültü altında bile
geometrik izleyici naif dedektörlerin hepsini geçiyor —

| gürültü | geometrik | en iyi naif |
|---|---|---|
| %0 | 0.996 / F1 0.81 | 0.661 / F1 0.20 |
| %1 | 0.816 / F1 0.59 | 0.645 / F1 0.30 |
| %5 | 0.671 / F1 0.18 | 0.625 / F1 0.17 |

Yani %1 gürültüye kadar anlamlı üstünlük var, %5'te fark kapanıyor. Doğru
ifade: **"korunan büyüklüğü doğrudan izlemek, tahmin-artığı izlemekten iyidir
— ölçüm gürültüsü invaryantın korunma sıkılığını aşmadığı sürece."**
Bu bir mühendislik reçetesi; devrim değil.

**İzleyicinin sınırları (dürüstlük):**

| sistem | AUROC | neden |
|---|---|---|
| D1 rijit cisim (atalet sıçraması) | 0.996 | anomali izlenen invaryantı bozuyor |
| D2 parçacık (alan ters dönmesi) | 0.514 | **anomali \|v\|'yi bozmuyor** — Boris dinamiğinde alan ters dönse de hız normu korunur; izleyici doğru olarak bir şey bulmuyor |
| D4 Lorenz (ρ değişimi) | 0.768 | korunum yok; kuadratik formlar yine de kayıyor, ama yanlış alarm yüksek (P 0.31) |

Yani izleyici evrensel bir dedektör değil: **izlenen invaryantı bozan**
anomalileri yakalıyor, başkasını değil. Reçete şu: "hangi büyüklük korunmalı?"
sorusunu cevaplayabildiğin ve ölçüm gürültüsü o korunumun sıkılığını aşmadığı
sistemlerde, öğrenilmiş bir tahminciye göre daha iyi bir dedektör kurulabilir.

## 3. Tezin güncellenmiş hali

Veriler üç ayrı iddiayı ayrıştırdı:

1. **Temsil (geometrik tipleme):** DESTEKLENDİ — slotlama hem doğruluğu hem
   korunumu iyileştirdi, eşit parametrede baseline'ı geçti.
2. **Geçiş kısıtı (rotor):** TEK BAŞINA YETERSİZ — serbest embedding'le
   çalışmıyor (kontrol grid'i, D2'de net olumsuz), doğru temsille birlikte
   çalışıyor. Kısıtın işe yaraması temsile bağımlı.
3. **Çarpım/karışım geometrisi (Clifford-linear):** ~~ŞU HALİYLE ZARARLI~~ —
   **HÜKÜM GERİ ÇEKİLDİ.** Test edilen kol eşdeğişken değildi (hata 0.93,
   vanilla mamba 1.08) ve geometrik çarpım katmanını hiç içermiyordu.
   Gerçek (d) — `variant: full_geo`, eşdeğişkenliği 1e-15'te doğrulanmış —
   henüz eğitilmedi. Bu iddia hakkında elimizde veri YOK.
4. **Bonus, planlanmamış (düzeltilmiş haliyle):** korunan büyüklüğü doğrudan
   izlemek, tahmin-artığı izlemekten iyi — ama sadece ölçüm gürültüsü düşükken
   (%1'e kadar anlamlı üstünlük, %5'te fark kapanıyor) ve sadece invaryantı
   bozan anomaliler için. Sentetik verideki 0.996 rakamı, korunumun makine
   hassasiyetinde olmasının artefaktı; gerçekçi beklenti 0.8 bandı.

## 4. Sonraki adımlar

1. **Slotlama teyidi:** N ∈ {100, 300, 1000} × 5 seed × {slot, rotor,
   mamba76}; slotlamayı vanilla mamba'ya da ekle (slot ⊗ rotor ayrıştırması:
   kazanç slotlamadan mı, rotor+slotlama etkileşiminden mi?).
2. **D2 parçacıkta slotlama** (orada slot_grade=1, hız bir vektör) — kontrol
   grid'inin olumsuz sonucu slotlamayla dönüyor mu?
3. fullgeo'yu onar veya düşür (gate'i grade-bazlı norm-gate yap).
4. İnvaryant izleyici için gerçek testler: (a) gerçek ölçüm gürültüsü ve
   sensör kaymasıyla, (b) invaryantı bozmayan anomalilerle, (c) öğrenilmiş
   dedektörlere gerçekten adil koşulda (aynı kalibrasyon penceresi + aynı
   yapısal bilgi verilerek). Bunlar geçilmeden Entes'e taşınmamalı.
