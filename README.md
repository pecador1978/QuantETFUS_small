# QuantETFUS_small

Lightweight repo with **scripts + configs** for the QuantETFUS workflow.  
Data and heavy outputs are stored separately in **iCloud (QuantShared)**.

---

## 🔹 Project Structure

### GitHub (this repo)
- `scripts/` → all Python scripts  
- `config/` → main JSON configs (rules, parameters, features)  
- `requirements.txt` → Python dependencies (if present)  
- Runner scripts (`run_dashboard_build.sh`, etc.)  
- `.gitignore`  

✅ Purpose: version control for code and configs. Safe to clone/pull on any machine.

---

### iCloud (QuantShared)
- `data_raw/` → original downloads  
- `data_enriched/` → enriched datasets  
- `merged_datasets/` → merged views  
- `data_final_training/` → train/test/val parquet/CSV dumps  
- `param_results/` → parameter sweeps & backtests  
- `signals/` → gate outputs  
- `boards/` → decision boards & dashboards  
- `models/` → ML models (if used)  
- `output/` → plots, reports, HTML dashboards  
- `_trash_*/` → temporary folders  

✅ Purpose: heavy, changing files. Synced across Macs with iCloud. Not versioned in Git.

---

### Ignored Everywhere
- System/editor noise (`.DS_Store`, `*.swp`, `*.bak`, `*.tmp`)  
- Python cache (`__pycache__/`, `*.pyc`)  
- IDE settings (`.vscode/`, `.idea/`)  

---

## 🔹 Workflow

1. **Code sync (GitHub)**
   - Before editing: `git pull`
   - After editing:  
     ```bash
     git add .
     git commit -m "Message"
     git push
     ```

2. **Data sync (iCloud)**  
   - Auto-synced between Mini & MBP.  
   - Repo assumes `QuantShared` path is consistent on both machines.

---

## 🔹 Notes
- Keep **JSON configs** in GitHub (except backups).  
- Keep **all parquet/CSV data** in iCloud (not GitHub).  
- `.gitignore` protects repo from large files.  

---
