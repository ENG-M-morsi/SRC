# ===================================================================
# optuna_fine_search.py — بحث دقيق حول أفضل تكوين
# ===================================================================
import os, sys, json, argparse
import optuna
import torch
import torch.nn as nn
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from optuna_tuner_arch import build_loaders, calc_psnr, HUTCN

DATA_DIR = None
NUM_EPOCHS = 15


def objective(trial):
    # نطاق ضيق حول Trial 17
    nf              = trial.suggest_categorical('nf', [80, 88, 96])
    num_heads_dat   = trial.suggest_categorical('num_heads_dat', [2, 4])
    ws_dat          = trial.suggest_categorical('ws_dat', [4, 8])
    num_blocks_dat  = trial.suggest_int('num_blocks_dat', 2, 4)
    num_heads_elan  = trial.suggest_categorical('num_heads_elan', [2, 4])
    ws_elan         = trial.suggest_categorical('ws_elan', [4, 8, 12])
    num_blocks_elan = trial.suggest_int('num_blocks_elan', 1, 2)
    fusion_heads    = trial.suggest_categorical('fusion_heads', [4, 8])
    fusion_dropout  = trial.suggest_float('fusion_dropout', 0.0, 0.2, step=0.1)
    hfe_reduction   = trial.suggest_categorical('hfe_reduction', [2, 4])
    lr              = trial.suggest_float('lr', 3e-4, 7e-4, log=True)
    batch_size      = 8
    weight_decay    = trial.suggest_float('weight_decay', 1e-6, 1e-5, log=True)

    if nf % fusion_heads != 0 or nf % num_heads_dat != 0 or nf % num_heads_elan != 0:
        raise optuna.TrialPruned()

    device = torch.device('cuda')
    model = HUTCN(
        in_nc=3, nf=nf, out_nc=3, upscale=4,
        num_heads_dat=num_heads_dat, ws_dat=ws_dat,
        num_blocks_dat=num_blocks_dat,
        num_heads_elan=num_heads_elan, ws_elan=ws_elan,
        num_blocks_elan=num_blocks_elan,
        fusion_heads=fusion_heads, fusion_dropout=fusion_dropout,
        hfe_reduction=hfe_reduction,
    ).to(device)

    train_loader, val_loader = build_loaders(DATA_DIR, batch_size=batch_size, scale=4, patch_size=96)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=lr * 0.01)
    criterion = nn.L1Loss()

    best_psnr = 0.0
    try:
        for epoch in range(NUM_EPOCHS):
            model.train()
            for batch in train_loader:
                lr_img, hr_img = batch[0].to(device), batch[1].to(device)
                optimizer.zero_grad()
                sr = model(lr_img)
                loss = criterion(sr, hr_img)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()

            model.eval()
            psnr_sum = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    lr_img, hr_img = batch[0].to(device), batch[1].to(device)
                    sr = model(lr_img)
                    psnr_sum += calc_psnr(sr, hr_img)
            avg_psnr = psnr_sum / max(len(val_loader), 1)
            best_psnr = max(best_psnr, avg_psnr)
            trial.report(avg_psnr, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
    except torch.cuda.OutOfMemoryError:
        del model, optimizer
        torch.cuda.empty_cache()
        raise optuna.TrialPruned()

    del model, optimizer, scheduler
    torch.cuda.empty_cache()
    return best_psnr


def run(data_dir, n_trials, num_epochs, out_dir):
    global DATA_DIR, NUM_EPOCHS
    DATA_DIR = data_dir
    NUM_EPOCHS = num_epochs

    os.makedirs(out_dir, exist_ok=True)
    storage = f'sqlite:///{os.path.join(out_dir, "optuna_fine.db")}'

    study = optuna.create_study(
        study_name='dtcf_fine',
        storage=storage,
        direction='maximize',
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=3, n_warmup_steps=2
        ),
        sampler=optuna.samplers.TPESampler(
            seed=42,
            n_startup_trials=5,      # ← بدون x_distributions
        ),
    )

    # أضف Trial 17 كنقطة بداية (enqueue قبل optimize)
    study.enqueue_trial({
        'nf': 88,
        'num_heads_dat': 2,
        'ws_dat': 4,
        'num_blocks_dat': 3,
        'num_heads_elan': 2,
        'ws_elan': 8,
        'num_blocks_elan': 1,
        'fusion_heads': 8,
        'fusion_dropout': 0.1,
        'hfe_reduction': 2,
        'lr': 4.99e-4,
        'weight_decay': 6.99e-6,
    })

    def _cleanup(study, trial):
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=True,
        catch=(RuntimeError,),
        callbacks=[_cleanup],
        gc_after_trial=True,
    )

    print('=' * 60)
    print(f'BEST FINE-TUNED: {study.best_value:.4f} dB')
    for k, v in study.best_params.items():
        print(f'  {k}: {v}')
    print('=' * 60)

    study.trials_dataframe().to_csv(
        os.path.join(out_dir, 'optuna_fine_results.csv'), index=False
    )

    with open(os.path.join(out_dir, 'best_arch_fine.json'), 'w') as f:
        json.dump(study.best_params, f, indent=2)

    print(f'\nSaved:')
    print(f'  DB   : {storage}')
    print(f'  CSV  : {os.path.join(out_dir, "optuna_fine_results.csv")}')
    print(f'  JSON : {os.path.join(out_dir, "best_arch_fine.json")}')
    return study

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', type=str, required=True)
    p.add_argument('--n_trials', type=int, default=20)
    p.add_argument('--num_epochs', type=int, default=15)
    p.add_argument('--out_dir', type=str, default='./optuna_fine_out')
    args = p.parse_args()
    run(args.data_dir, args.n_trials, args.num_epochs, args.out_dir)