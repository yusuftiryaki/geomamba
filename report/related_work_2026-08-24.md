# Literatür taraması — öğrenilen simetri yönü (24 Ağu 2026)

M0 taraması (15 Ağu) "sertçe gömülü rotor + selective SSM" boşluğunu aramıştı.
Bu tarama yeni yönü arıyor: **geometri makinesi verilmiş ama simetriyi veriden
öğrenen** mimariler. Not: aşağıdakiler özet/abstract düzeyinde okundu, tam
metin doğrulaması yapılmadı.

## Sonuç: bu kutu boş değil

### 1. Latent Mixture of Symmetries (arXiv 2510.03578) — EN YAKIN, fikrin aynısı
Latent uzayda **Lie grubu aksiyonları üzerinde mixture-of-experts kapılaması**:
dönme, öteleme, ölçekleme ve bileşimleri arasından hangisinin geçerli olduğunu
veriden öğreniyor. Yapısız ODE güncellemesini "yapılandırılmış, simetri-koruyan"
bir güncellemeyle değiştiriyor. Çoklu zaman ölçeğinde hiyerarşik.

- **Bizim önerdiğimiz "öğrenen versor" kolunun Lie-teorisi versiyonu.**
- Raporladığı kazanç: %15–97, ve **özellikle veri kıtlığında** (%90 veri düşürme).
- Test ettiği sistemler: ODE'ler (spiral, glikolitik, av-avcı) **ve gerçek dünya
  zaman serileri: elektrik tüketimi, güneş enerjisi, güç sistemleri, EKG** —
  yani bizim Entes uygulama alanımız.
- Farkları: Clifford değil Lie teorisi; selective SSM değil; encoder/decoder ile
  latent uzayda çalışıyor.

**Dikkat çeken gerilim:** Latent MoS de kısıtı **latent uzayda** uyguluyor ve
serbest bir encoder/decoder kullanıyor — bizim "kısıt gizli state'te, metrik
gözlem uzayında, bağ kopuyor" teşhisimizin aynısı. Ama onlar kazanç
raporluyor. Ya kapılama mekanizması farkı yaratıyor, ya da bizim teşhisimiz
eksikti. Tam metin okunmalı.

### 2. Adaptive Symmetry Discovery for Dynamical System Identification (2608.08091, ICML 2026)
Teorik: bilinen simetriye sahip sistemlerin tanımlanması için **belirgin biçimde
daha kısa yörüngeler** yettiğini kanıtlıyor, ve tek bir yörüngeden simetri
grubunu öğrenen bir yöntemin aynı optimal yörünge uzunluğuna ulaştığını
gösteriyor. Yani "simetri veri verimliliği getirir" iddiasının teorik ispatı
sistem tanımlama bağlamında yapılmış. Sinir ağı mimarisi önermiyor.

### 3. RECM — Recurrent Equivariant Constraint Modulation (2602.02853, Şub 2026)
Katman başına simetri gevşetmesini veriden öğreniyor; her katman kendi
"simetri açığı" ile sınırlı bir gevşeme düzeyine yakınsıyor. İsmindeki
"recurrent" dizi modeli anlamında değil. Moleküler konformer üretimi.

### 4. EQ-VMamba — Rotation Equivariant Mamba (2603.09138, Mar 2026)
**Uçtan uca dönme-eşdeğişken Mamba**, eşdeğişkenlik hatasının teorik analiziyle.
Sadece görü görevleri (sınıflandırma, segmentasyon, süper çözünürlük).
Eşit performans için ~%50 daha az parametre.

Yani "eşdeğişken SSM" de artık yeni değil; fiziksel zaman serisine
uygulanmamış olması kalan tek boşluk, ve sandığımızdan dar.

### 5. Diğerleri
- Relaxed group convolution ile simetri kırılması keşfi, fiziksel sistemler (2310.02299)
- REMUL: multitask ile gevşek eşdeğişkenlik (2410.17878)
- Eşdeğişken model eğitiminde kısıt gevşetme (2408.13242)
- GA-LSTNet: geometrik cebirle çok boyutlu zaman serisi tahmini (Clifford RNN)

## Bizim konumumuz — güncellenmiş

**Kaybettiğimiz iddialar:**
- "Öğrenilen simetri + geometri makinesi" fikri yeni değil; Latent MoS yapmış.
- "Eşdeğişken SSM" yeni değil; EQ-VMamba yapmış (görüde).
- "Simetri veri verimliliği getirir" teorik olarak da, deneysel olarak da
  başkalarınca desteklenmiş — bizim negatif sonucumuza rağmen.

**Hayatta kalan katkılar (mütevazı ama gerçek):**
1. **`rot_gap` teşhis aleti.** Eğitilmiş bir modelin gerçekte ne kadar
   eşdeğişken olduğunu ölçen basit, ucuz, mimariden bağımsız bir ölçüm
   (döndürülmüş test seti / normal test seti, persistence tabanıyla
   normalize edilmiş). Taramada bunun standart bir pratik olduğuna dair bir
   şey bulamadım; EQ-VMamba eşdeğişkenlik hatasını teorik analiz ediyor,
   ölçmüyor. Bu aletle "geometrik" diye anılan kendi kollarımızın hiç
   eşdeğişken olmadığını gösterdik (rot_gap 7–9, taban 1.07).
2. **`scatter` kontrolü.** Aynı parametre sayısı ve seyreklikte ama geometrik
   olmayan yerleştirme — "kazanç geometriden mi yoksa kısıtlılıktan mı"
   sorusunu ayıran ucuz kontrol. Slotlama kazancımızın geometrik olmadığını
   bununla gösterdik.
3. **Negatif sonuç:** sertçe gömülü SO(3) rotor kısıtı, parametre eşitlenmiş
   selective SSM'de rijit cisim ve yüklü parçacık dinamiklerinde veri
   verimliliği getirmiyor; negatif kontrol (Lorenz) beklendiği gibi davranıyor.

## Sonraki adım için not

Latent MoS'un tam metni okunmalı: (a) latent-uzay kısıtının bizde neden
çalışmadığını açıklıyor olabilir, (b) kapılama mekanizması "öğrenen versor"
kolumuzun doğrudan tasarım şablonu olabilir, (c) güç sistemi zaman serilerinde
raporladığı kazanç Entes fazına doğrudan ilgili.
