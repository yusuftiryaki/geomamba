# Kaggle'da GPU ile koşturma

Kod saf PyTorch — CUDA'lı `mamba-ssm` gerekmez; trainer GPU'yu otomatik seçer.

## Adımlar

1. Bu repoyu zip'leyip Kaggle'a **Dataset** olarak yükleyin (veya GitHub'a itip
   notebook'ta `git clone` yapın).
2. Yeni Notebook → Settings → **Accelerator: GPU (T4)**, Internet: On.
3. Tek hücre:

```bash
%%bash
cp -r /kaggle/input/geomamba/geomamba /kaggle/working/geomamba  # dataset adına göre uyarlayın
cd /kaggle/working/geomamba
export PYTHONPATH=$PWD/src
python scripts/run_grid.py --grid configs/grid/m3_rigid_body.yaml
```

4. Çıktılar: `experiments/grid_results.csv` (+ her koşunun `eval.json`'ı).
   Notebook çıktısını indirip bu makinede `experiments/` altına koyun;
   analiz/grafik scriptleri aynen çalışır.

Notlar:
- Grid ilk hücreleri (N=100/300 — karar hücreleri) önce koşar; oturum yarıda
  kesilirse `experiments/grid_results.json` kısmi sonuçları tutar.
- Kaggle haftalık ~30 GPU-saat verir; bu grid (24 koşu × ~3-5 dk) ~2-3 saat.
- Veri notebook içinde script'le üretilir (seed'li, reprodüktif) — yükleme yok.
