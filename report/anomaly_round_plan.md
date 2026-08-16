# Anomali Turu — Ertelenmiş, Hedefli Çalışma Planı

**Karar (16 Ağu 2026):** Anomali, tahmin modelinin yan ürünü olarak ele
alınmaktan çıkarılıp ayrı bir tura bırakıldı. Mevcut odak: tahmin + mimari
(slotlama kazancı, genelleme, transfer).

## Neden mevcut kurulum zayıf kaldı (teşhis)

Anomalimiz bir **parametre değişimi** (atalet sıçraması). Model eğitimde
binlerce farklı parametre gördüğü için, arıza sonrası dinamik onun gözünde
tamamen dağılım-içi. Mamba'nın selectivity'si zaten "bağlamdan parametreyi
çıkar, uyum sağla" için tasarlanmış: model bu işte ne kadar iyiyse tahmin
artığı rejim değişimini o kadar iyi **gizliyor**.

Yanlış soru: "bu adımı tahmin edebiliyor musun?"
Doğru soru: **"hangi sisteme baktığına dair kanaatin değişti mi?"**

Ölçülen (D1 rijit cisim / D2 parçacık, AUROC):
- öğrenilmiş tahmin artığı: 0.55 / 0.67
- invaryant izleyici: 0.82 (%1 gürültü) / 0.51
- naif dedektörler: 0.62-0.66 / 0.50-0.70
İki yöntemin kör noktaları tümleyen — birleştirme adayı.

## Ana fikir: iç-durum (rejim kanaati) izleme

Rotor modelinde her adımda çıkarsanan `b_t` (dönme ekseni + hızı) ve `Δ_t`
(adım ölçeği) var. Bunlar modelin "şu an hangi rejimdeyim" tahmini ve
**fiziksel anlamı olan** büyüklükler — vanilla Mamba'nın opak state'inden
farkı bu. Rejim değişince tahmin artığı normale dönse bile bu iç
parametrelerin sıçraması beklenir.

Proje hikâyesine katkısı: "geometri modeli yorumlanabilir kılar, bu da
rejim değişimi tespitini mümkün kılar" — kısıtın bilinen ikinci faydası.

Uygulama: `GeoS6.forward` içinde `b_t`, `Δ_t`, `decay` tensörlerini
opsiyonel olarak dışarı ver (hook veya `return_internals=True`), kanal
üzerinden özetle (ör. eksen yönünün açısal değişimi, |b| değişimi), üzerine
CUSUM/değişim-noktası uygula.

## Turun kapsamı

1. **İç-durum izleyici** (yukarıdaki) — ana yeni yöntem.
2. **Ensemble:** invaryant izleyici + iç-durum + artık skoru; kör noktalar
   tümleyen olduğu için birleşim asıl adaydır.
3. **Tespit için doğrudan eğitim:** tek-sınıf/karşıtsal hedef, ya da rejim
   kimliğini tahmin eden yardımcı kafa (parametre-kestirim başlığı).
4. **Gerçekçi arıza tipleri:** mevcut anomaliler ani sıçrama; gerçek arızalar
   kademeli bozulma. Yavaş sürüklenen parametre üreteci eklenmeli (RUL ile
   de örtüşür).
5. **Adil ve tam değerlendirme:** olay-düzeyi F1 + precision ve recall birlikte,
   gürültü taraması (%0-5), invaryantı bozan VE bozmayan anomaliler ayrı ayrı,
   tüm yöntemlere aynı kalibrasyon penceresi.
6. **Gerçek veri:** Entes ölçümlerinden küçük bir kesit; sentetik sonuçlar
   gerçek arızalara genellemeden ürün iddiası kurulmayacak.

## Uyarılar (bu turda öğrenildi)

- Sentetik veride korunan büyüklükler makine hassasiyetinde korunur; bu
  invaryant tabanlı yöntemlere yapay avantaj verir. Değerlendirme mutlaka
  gürültülü yapılmalı.
- AUROC tek başına yanıltıcı: |v| CUSUM D2'de 0.699 AUROC alıp olay-F1'de
  0.000 verdi (sıralama iyi, eşik tutmuyor). Olay-düzeyi metrik zorunlu.
- Recall'ı precision'sız raporlamak yanıltıcıdır (bkz. skorlama raporundaki
  düzeltme).
