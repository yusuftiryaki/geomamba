# GeoMamba — Full Geometric Mamba

ArGe projesi: Clifford (geometrik) cebrinin simetri ve çarpım yapısını
Mamba-tarzı bir SSM'in state transition'ına inductive bias olarak koyarak,
fiziksel sistemlerin zamansal dinamiklerini daha veri-verimli ve fiziksel
tutarlı öğrenmek.

Çekirdek fikir: hidden state `H_t ∈ R^{C×8}` Cl(3,0) multivector'ları;
transition rotor sandviçi `h_t = exp(-Δλ)·R_t h_{t-1} R̃_t + Δ·B_t⊙E(x_t)`
— Mamba'nın girdi-bağımlı Δ'sı değişken-adımlı Lie-grubu integratörünün adımı
olur. Görevler: anomaly (birincil), forecast, RUL.

## Kullanım

Her şey konteynerde koşar; host'a hiçbir şey kurulmaz.

```bash
make build   # imajı kur (aarch64, CPU torch)
make test    # testleri konteynerde koştur
make shell   # konteynerde kabuk
```

VS Code ile: "Reopen in Container" (`.devcontainer/`).

Plan ve deney tasarımı: bkz. proje planı (ablasyon merdiveni a→d,
veri-verimliliği protokolü, D1–D5 sentetik fizik veri setleri).
