# =====================================================================
# _backfill_user_attrs.py — v2 (Direct SQLite write, Optuna 3.x compatible)
# =====================================================================
import os, sys, json, time, warnings, sqlite3, importlib
from pathlib import Path
from collections import Counter
warnings.filterwarnings('ignore')

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import optuna
import torch

try:
    apv = importlib.import_module("analyze_params_vs_psnr")
except Exception as e:
    print(f"❌ Cannot import analyze_params_vs_psnr: {e}")
    sys.exit(1)

STUDY_NAME   = apv.STUDY_NAME
STORAGE_PATH = apv.STORAGE_PATH
STORAGE_URL  = f"sqlite:///{STORAGE_PATH}"
DUMMY_H      = apv.FLOPS_IMG_H
DUMMY_W      = apv.FLOPS_IMG_W
DUMMY_C      = apv.FLOPS_IN_CH
MODEL        = apv.MODEL_DISPLAY_NAME

print("=" * 78)
print(f"🔧 Backfill user_attrs (direct SQLite) — {MODEL}")
print(f"📂 Folder: {HERE}")
print(f"🗄️  DB    : {STORAGE_PATH}")
print("=" * 78)

if not STORAGE_PATH.exists():
    print(f"❌ DB not found: {STORAGE_PATH}")
    sys.exit(1)

study = optuna.load_study(study_name=STUDY_NAME, storage=STORAGE_URL)
print(f"✅ Study: '{STUDY_NAME}' — {len(study.trials)} trials")

# ═══ اتصال SQLite مباشر ═══
conn = sqlite3.connect(str(STORAGE_PATH))
cur  = conn.cursor()

# تحقق من schema (تأكيد أن الأعمدة صحيحة)
cur.execute("PRAGMA table_info(trial_user_attributes)")
cols = [r[1] for r in cur.fetchall()]
print(f"ℹ️  Schema trial_user_attributes: {cols}")
if 'trial_id' not in cols or 'key' not in cols or 'value_json' not in cols:
    print(f"❌ Unexpected schema — abort.")
    sys.exit(1)


def _set_attr_sqlite(trial_id, key, value):
    cur.execute(
        "INSERT OR REPLACE INTO trial_user_attributes "
        "(trial_id, key, value_json) VALUES (?, ?, ?)",
        (trial_id, key, json.dumps(value))
    )
    conn.commit()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dummy  = torch.randn(1, DUMMY_C, DUMMY_H, DUMMY_W).to(device)

n_updated = n_skipped = n_failed = 0
breakdown = Counter()
t0 = time.time()

for t in study.trials:
    if t.state != optuna.trial.TrialState.COMPLETE:
        continue

    ua_np = t.user_attrs.get('n_params')
    ua_fl = t.user_attrs.get('flops_G')

    if ua_np is not None and ua_fl is not None:
        n_skipped += 1
        continue

    try:
        model = apv.build_model_for_flops(t.params).to(device)
        model.eval()

        n_params = int(sum(p.numel() for p in model.parameters()
                           if p.requires_grad))

        flops_val = ua_fl
        if ua_fl is None:
            from fvcore.nn import FlopCountAnalysis
            with torch.no_grad():
                fca = FlopCountAnalysis(model, dummy)
                fca.unsupported_ops_warnings(False)
                fca.uncalled_modules_warnings(False)
                flops_val = float(fca.total() / 1e9)

        if ua_np is None:
            _set_attr_sqlite(t._trial_id, 'n_params', n_params)
            breakdown['n_params'] += 1
        if ua_fl is None:
            _set_attr_sqlite(t._trial_id, 'flops_G', flops_val)
            breakdown['flops_G'] += 1

        n_updated += 1
        print(f"   ✅ Trial #{t.number:3d}: "
              f"n_params={n_params:,} | flops={flops_val:.2f}G")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    except Exception as e:
        n_failed += 1
        print(f"   ⚠️  Trial #{t.number}: {type(e).__name__}: {e}")

conn.close()

elapsed = time.time() - t0
print("\n" + "=" * 78)
print(f"🎉 Done in {elapsed:.1f}s")
print(f"   ✅ Updated   : {n_updated}")
print(f"   ⏭️  Skipped   : {n_skipped} (already had attrs)")
print(f"   ⚠️  Failed    : {n_failed}")
if breakdown:
    print(f"   🔧 Breakdown : {dict(breakdown)}")
print("=" * 78)