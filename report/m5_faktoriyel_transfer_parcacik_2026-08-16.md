# M5 — Faktoriyel + Transfer + Parçacık-Slotlama (16 Ağu 2026)

Üç grid aynı gün koştu. Birlikte okunduğunda çekirdek iddia hakkında net bir
hüküm, bir de geçersiz sayılması gereken bir deney çıkıyor.

Tüm koşular parametre eşlenmiş (125.1k–129.6k, ±%2 ortalama etrafında):

| config | param | d_model | d_state | varyant | slot |
|---|---|---|---|---|---|
| rb_mamba | 128.2k | 76 | 8 | mamba | hayır |
| rb_mamba_slot | **125.1k** | 72 | 16 | mamba | evet |
| rb_rotor | 129.6k | 64 | 8 | rotor | hayır |
| rb_rotor_slot | 128.9k | 64 | 8 | rotor | evet |

En iyi hücre (mamba+slot) **en küçük** model — parametre confound'u bu sefer
sonucun aleyhine çalışıyor, lehine değil.

---

## 1. Faktoriyel (D1 rijit cisim): kazanç slotlamadan, rotordan değil

2×2×{N=100,300}×3 seed. Rollout MSE h100:

**N=300**

| hücre | h100 | enerji kayması | 1-adım MSE |
|---|---|---|---|
| **mamba + SLOT** | **0.442 ±0.040** | **1.98 ±1.55** | 7.6e-5 |
| rotor + SLOT | 0.477 ±0.120 | 2.63 ±1.78 | 1.20e-4 |
| rotor / serbest | 0.515 ±0.126 | 16.2 ±24.6 | 1.07e-4 |
| mamba / serbest | 0.767 ±0.335 | 4.12 ±4.79 | 7.8e-5 |

**N=100**

| hücre | h100 | enerji kayması |
|---|---|---|
| mamba / serbest | **0.680 ±0.069** | 2.71 |
| rotor + SLOT | 0.694 ±0.120 | 5.23 |
| mamba + SLOT | 0.802 ±0.145 | 3.86 |
| rotor / serbest | 0.860 ±0.103 | 6.28 |

Seed-bazlı işaret tutarlılığı (3 seed):

| karşılaştırma | N=100 | N=300 |
|---|---|---|
| slotlama vanilla mamba'ya | **0/3 kazanç** (bozuyor) | **3/3 kazanç** |
| slotlama rotora | **3/3 kazanç** | 2/3 kazanç |
| rotor, slotlamanın üstüne | 3/3 kazanç | 1/3 (yani yok) |

**Hüküm:** N=300'de kazancın tamamı slotlamaya atfedilebiliyor; rotor kısıtı
slotlamanın üstüne ölçülebilir bir şey eklemiyor (0.442 vs 0.477, seed
bazında 1/3). N=100'de tablo tersine dönüyor ve en iyi hücre hâlâ düz mamba.
İki N zıt etkileşim işareti veriyor — 3 seed bu ayrımı taşımıyor.

Bu, `slotting_and_scoring` raporundaki "geo varyant baseline'ı geçti"
ifadesini **daraltıyor**: geçiren şey geometrik state kısıtı değil, geometrik
girdi tiplemesi. Projenin çekirdeğindeki rotor sandviçi, doğru temsille
birlikte bile kendini ödetmiyor.

### Rakip açıklama (henüz elenmedi)

Slotlama = sabit, yönü koruyan, kimliğe yakın bir gömme. Az veride kazancı
"geometri"den değil, **öğrenilecek daha az şey olmasından** geliyor olabilir.
Ayırt edici kontroller:

1. **Yanlış grade:** D1'de ω'yı grade-2 yerine grade-1'e slotla. Aynı kazanç
   geliyorsa iş geometride değil. Sadece rotorlu modelde bozuluyorsa
   "grade, geçiş grade-farkındaysa önemli" demektir.
2. **Donmuş rastgele gömme:** aynı ranklı, sabit, öğrenilmeyen lineer gömme.
   Slotlamayla eşitse kazanç yapıdan değil, serbestlik derecesi azalmasından.

Bu iki kontrol geçilmeden slotlama bulgusu "geometrik" diye sunulamaz.

---

## 2. Transfer (D1 → D2): **GEÇERSİZ, yeniden koşulmalı**

Tasarım: D1'de ön-eğit, D2'de çok az veriyle (N=25/100) ince ayar; kontrol =
aynı bütçeyle sıfırdan. İnce ayar bütçesi: 3000 adım, lr=1e-4.

| N=100 | 1-adım MSE | h100 | \|v\| kayması |
|---|---|---|---|
| mamba / sıfırdan | 6.4e-4 | 1.79 | 0.70 |
| **mamba / transfer** | **1.4e-4 (−79%, 3/3)** | **1.27 (−29%, 3/3)** | **0.46 (−34%, 3/3)** |
| rotor+slot / sıfırdan | 1.41e-3 | 16.3 | 13.3 |
| rotor+slot / transfer | 1.33e-3 (−6%) | 16.9 (+4%) | 15.2 (+14%) |

İlk okumada sonuç şu görünüyor: baseline temiz transfer ediyor, geo model
etmiyor. **Ama bu okuma geçersiz.** Aynı `lp_rotor_slot` mimarisi, aynı D2
verisinde, tam bütçeyle (10k adım, varsayılan lr) koşan parçacık grid'inde
N=100'de h100 ≈ **1.3** alıyor. Burada 16.9 alıyor. Yani geo model bu
gridde **yakınsamamış**; 3000 adım / lr=1e-4 onun için yeterli değil.

Yakınsamamış iki koşu arasında "başlangıç noktası yardım etti mi?" sorusu
ölçülemez. Dolayısıyla:

- **Baseline hakkında geçerli bulgu:** vanilla Mamba, rijit cisimden yüklü
  parçacığa (farklı denklem, farklı gözlenebilir) **transfer ediyor** —
  N=100'de dört metrikte de 3/3 seed. Bu başlı başına dikkate değer ve
  "sadece geometrik model transfer edebilir" çerçevesini zayıflatıyor.
- **Geo model hakkında bulgu yok.** Deney tekrar edilmeli: her iki mimari için
  de tam bütçe (10k adım, varsayılan lr) ya da mimari başına ayrı lr taraması.

Yan bulgu: geo model belirgin biçimde **daha yavaş optimize oluyor**. Bu
kendi başına bir veri-verimliliği aleyhtarı sinyal — adım bütçesi sabitken
kısıtın bedeli var.

---

## 3. Parçacık + slotlama (D2): olumsuz sonuç kısmen düzeldi

Kontrol grid'i D2'de rotoru serbest gömmeyle koşmuş ve her metrikte
kaybetmişti. Slotlamayla (slot_grade=1, hız bir vektör), N=300 medyanları:

| hücre | h50 | h100 | \|v\| kayması | anom AUROC |
|---|---|---|---|---|
| mamba / serbest | 0.402 | 1.029 | 0.472 | 0.685 |
| mamba / SLOT | 0.281 | 0.817 | 0.605 | 0.706 |
| rotor / serbest | 0.436 | 1.259 | 1.032 | 0.689 |
| **rotor / SLOT** | **0.321** | **0.809** | **0.500** | **0.725** |

Slotlama rotorun |v| korunumunu iki kattan fazla düzeltiyor (1.03 → 0.50) —
kontrol grid'inin teşhisi ("kısıt gizli state'te, metrik gözlem uzayında;
serbest gömme bağı koparıyor") bu sistemde de doğrulanıyor. Ama en iyi h100
hücresi yine slotlamalı iki mimari arasında berabere (0.809 vs 0.817): rotor
yine ayırt edici değil.

N=100'de ortalamalar patlayan tek seed'lerle bozuluyor (mamba+slot s1 h100
5.64; rotor+slot s1 12.5) — bu N'de hiçbir şey iddia edilemez.

**Anomali (parkta ama not düşülmeli):** slotlama AUROC'u her iki mimaride de
tutarlı biçimde yükseltiyor; rotor+slot N=300'de 3/3 seed'de rotor/serbest'i
geçiyor (0.725 vs 0.689) ve tüm gridin en iyisi. İlk kez bir geo varyant
anomalide ayrışıyor. Küçük ama temiz; anomali turuna girdi.

---

## 4. Birleşik hüküm

| iddia | durum |
|---|---|
| Geometrik girdi tiplemesi (slotlama) yardım ediyor | **DESTEKLENDİ** — D1 ve D2'de, N=300'de, hem doğrulukta hem korunumda; ama "geometri mi yoksa basit gömme mi" kontrolü yapılmadı |
| Rotor state kısıtı, doğru temsille birlikte yardım ediyor | **DESTEKLENMEDİ** — slotlamanın üstüne ölçülebilir katkısı yok (D1 1/3 seed, D2 berabere) |
| Rotor tek başına (serbest gömme) yardım ediyor | **ÇÜRÜTÜLDÜ** (önceki gridler) |
| Clifford-lineer kanal karışımı yardım ediyor | **ÇÜRÜTÜLDÜ** (fullgeo, 10× kötü) |
| Geometri sistemler arası transfer ediyor | **ÖLÇÜLEMEDİ** — deney bütçe hatasıyla geçersiz; baseline'ın transfer ettiği doğrulandı |
| Geo varyant anomalide ayrışıyor | zayıf ama ilk pozitif işaret (D2, slotlamalı, 3/3 seed) |

Projenin başlangıçtaki çekirdek hipotezi — *Clifford çarpım/simetri yapısını
**state transition'a** koymak* — bu noktada desteklenmiyor. Hayatta kalan ve
tekrarlanan bulgu, iddianın **temsil ayağı**: gözlenebilir büyüklüğü doğru
geometrik tipe yerleştirmek işe yarıyor, geçişi kısıtlamak yaramıyor.

## 5. Sonraki adımlar (öncelik sırasıyla)

1. **Yanlış-grade + donmuş-rastgele-gömme kontrolleri.** Ucuz ve belirleyici:
   slotlama kazancı geometrik mi, yoksa sadece "daha az öğrenilecek parametre"
   mi? Bu cevaplanmadan M5 raporu yazılamaz.
2. **Transfer'i adil bütçeyle tekrarla** (10k adım, varsayılan lr, iki mimari).
3. **Seed sayısını 3 → 6 çıkar** N=300 hücrelerinde; mevcut std'ler ayrımı
   taşımıyor.
4. Anomali turu (planı `anomaly_round_plan.md`) — D2 slotlamalı sinyali
   başlangıç noktası olarak kullan.
