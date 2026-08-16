# Kaggle'da GPU ile koşturma

Repo: `github.com/yusuftiryaki/geomamba` (private). Kod saf PyTorch —
CUDA'lı `mamba-ssm` gerekmez; trainer GPU'yu otomatik seçer.

## Bir kerelik hazırlık

1. GitHub'da token oluşturun: Settings → Developer settings →
   Fine-grained personal access token → sadece `geomamba` reposu,
   **Contents: Read-only** yetkisi.
2. Kaggle notebook'ta: Add-ons → Secrets → `GH_TOKEN` adıyla ekleyin.
3. Notebook Settings → **Accelerator: GPU (T4)**, Internet: **On**.

## Çalıştırma (iki hücre)

```python
from kaggle_secrets import UserSecretsClient
token = UserSecretsClient().get_secret("GH_TOKEN")
!git clone https://{token}@github.com/yusuftiryaki/geomamba.git /kaggle/working/geomamba
```

```bash
%%bash
cd /kaggle/working/geomamba
export PYTHONPATH=$PWD/src
python scripts/run_grid.py --grid configs/grid/m3_rigid_body.yaml
```

Çıktılar: `experiments/grid_results.csv` (+ her koşunun `eval.json`'ı).
Save Version ile çıktıyı kaydedin; `grid_results.csv`'yi bu VM'e getirin
(`experiments/` altına), analiz scriptleri aynen çalışır.

Notlar:
- Grid, karar hücrelerini (N=100/300) önce koşar; oturum yarıda kesilirse
  `experiments/grid_results.json` kısmi sonuçları tutar.
- Kaggle haftalık ~30 GPU-saat verir; bu grid (24 koşu) ~2-3 saat sürer.
- Veri notebook içinde seed'li script'le üretilir — dosya yüklemek gerekmez.
