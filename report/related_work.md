# İlgili Çalışmalar Taraması (15 Ağu 2026)

Web taraması ~13 sorgu + en yakın iki adayın tam metin kontrolü. Sonuç:
**birebir fikir alınmamış, ama iki yandan yakın kuşatılmış durumda.**

## En yakın 5 çalışma (sıralı)

1. **Versor** ([arXiv:2602.10195](https://arxiv.org/abs/2602.10195), Şub 2026) — EN YAKIN.
   Konformal GA Cl(4,1)'de spinor/multivector hidden state; girdi-bağımlı yerel
   rotor ΔR_t (Cayley haritası) ile Ψ_{t+1} = Normalize(ΔR_t Ψ_t), lineer zaman.
   Mamba'yı baseline olarak yenmiş. **Farklarımız (tam metinden doğrulandı):**
   (a) tek-taraflı rotor çarpımı, sandviç R h R̃ değil; (b) Mamba-tarzı
   selectivity / girdi-bağımlı Δ yok; (c) sönüm yok (norm birebir korunuyor);
   (d) Cl(4,1) vs bizim Cl(3,0); (e) forecast/anomaly/RUL benchmark'ı yok.
2. **Mamba-3** ([arXiv:2603.15569](https://arxiv.org/abs/2603.15569), ICLR 2026) —
   selective SSM'e kompleks-değerli (2B rotasyonel) state update + decay ekliyor =
   bizim mekanizmanın 2B komütatif özel hâli. GA dili yok, multivector yok,
   fiziksel tutarlılık iddiası yok, LLM odaklı. (Sadece abstract erişilebildi —
   related-work yazarken tam metin doğrulanmalı.)
3. **DeltaProduct / RWKV-7** ([arXiv:2502.10297](https://arxiv.org/abs/2502.10297),
   [arXiv:2503.14456](https://arxiv.org/abs/2503.14456)) — girdi-bağımlı
   Householder (ortogonalimsi) geçişler; ekspresivite motivasyonlu, geometri/fizik yok.
4. **RotRNN** ([arXiv:2407.07239](https://arxiv.org/abs/2407.07239)) — rotasyon
   matrisli SSM geçişi, ama selective değil.
5. **GCAN** (Ruhe et al., ICML 2023) — versor sandviç çarpımı dinamik sistem
   primitivi olarak var, ama feedforward; recurrence/selectivity yok.

Ayrıca: Clifford Neural Layers (ICLR 2023), GATr (NeurIPS 2023), LinOSS
(ICLR 2025, fiziksel yapı koruyan discretization argümanıyla kıyaslanacağız),
Hamiltonian SSM'ler (2507.10792, 2601.06212). Mamba+RUL/anomali literatürü
bol ama tamamı geometrisiz.

## Hayatta kalan yenilik deltası

- **Sandviç etkisi R h R̃** tam multivector üzerinde (grade-kovaryant adjoint
  aksiyon) — Versor'un tek-taraflı spinor kompozisyonundan gerçek temsili fark;
  ablasyonla gösterilmeli.
- **Çarpımsal sönüm ayrışımı exp(-Δλ)·Rotor** — "korunumlu kısım × yitirimli
  kısım" mimari olarak ayrılmış; hiçbir yakın çalışmada yok. En temiz yapısal
  iddiamız, fiziksel tutarlılık hikayesini doğrudan taşıyor.
- **Selectivity = değişken-adımlı Spin(3) üstel-harita integratörü** çerçevesi —
  "Δ değişken adım" tek başına yeni değil (Mamba'nın kendisinde var), Spin(3)
  üzerinde girdi-bağımlı bivector üreteçle formülasyonu yeni.
- **Cl(3,0) seçimi** — 3B fiziksel sensör verisi için (vektör + açısal momentum
  bivector'ı aynı state'te) kimse kullanmamış.
- **Uygulama alanı tamamen boş**: geometrik/Clifford dizi modeliyle anomaly,
  RUL, endüstriyel fiziksel zaman serisi veri-verimliliği — hiç kimse dokunmamış.

## Riskler

- Versor grubu 6 aylık; selectivity/sönüm eklemeleri muhtemel — hız önemli.
- Mamba-3 sonrası "rotasyon selective SSM'e iyi gelir" artık genel kanı;
  katkımız "rotasyonel selective SSM" değil, **nonkomütatif Spin(3)/multivector
  genellemesi + fizik uygulaması** olarak konumlanmalı.
- Reviewer özeti "Versor'un state'i + Mamba-3'ün selectivity'si" olabilir;
  iki yarıya da ilklik iddia edilemez, ikisi de cite edilip kıyaslanmalı.
