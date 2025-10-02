# QuantETFUS_small

Lightweight repo with **scripts + configs** for the QuantETFUS workflow.  
All heavy **data & outputs** live separately in **iCloud (QuantShared)**.

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
- Backup configs (`*_backup.json`)  

---

## 🔹 Daily Workflow

1. **Code sync (GitHub)**  
   - Before editing:  
     ```bash
     git pull
     ```
   - After editing:  
     ```bash
     git add .
     git commit -m "Describe change"
     git push
     ```

2. **Data sync (iCloud)**  
   - Handled automatically by iCloud.  
   - Scripts assume `QuantShared` path is consistent on all machines.

---

## 🔹 Setup Guide for New Machine (e.g. MBP)

1. **Install Git + Python**  
   ```bash
   git --version
   python3 --version