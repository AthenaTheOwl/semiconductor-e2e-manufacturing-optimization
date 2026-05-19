# Deploy — Streamlit Community Cloud

The app is in deployable shape. To put it on a public URL:

1. Sign in at https://share.streamlit.io with the GitHub account that owns `AthenaTheOwl/semiconductor-wafer-robust-optimization` (one-time browser auth).
2. Click **New app** → **From existing repo**.
3. Fill in:
   - **Repository**: `AthenaTheOwl/semiconductor-wafer-robust-optimization`
   - **Branch**: `main`
   - **Main file path**: `app.py`
   - **Advanced settings → Python version**: `3.11`
   - **Secrets**: leave empty — no API-key dependencies.
4. Click **Deploy**.

First build takes ~3 minutes (installs CVXPY + dependencies).

## Local dev

```bash
pip install -r requirements.txt
streamlit run app.py
```

## What the app does

Interactive Robust Optimization Dashboard for a fabless semiconductor
supply chain:
- Supplier wafer allocation under demand uncertainty.
- Three uncertainty modes (nominal, robust, adaptive).
- Side-by-side comparison of cost breakdowns and supplier mixes.

Pure CVXPY; no external services.

## Notes

- `requirements.txt` sits at repo root next to `app.py`.
- The default solver path uses CVXPY's open-source backends (ECOS/SCS).
- If a deployment ever needs a commercial solver (Gurobi, Mosek), add
  it via **Advanced settings → Secrets** with the license key — never in
  the repo.
