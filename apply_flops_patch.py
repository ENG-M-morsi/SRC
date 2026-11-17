# apply_flops_patch.py  (v2) — per-image FLOPs column in the test Excel files, for ANY model.
# Edits 3 files in the src folder:  utility.py , trainer.py , option.py
# Usage (from the src folder):   python apply_flops_patch.py
# - checks every text before changing anything; keeps *.bak backups
# - if the first version of this patch was applied, it restores the originals from *.bak first
import sys, shutil
from pathlib import Path

def load(p):
    raw = Path(p).read_bytes().decode('utf-8')
    return raw.replace('\r\n', '\n'), ('\r\n' in raw)

def must(text, old, new, label, count=1):
    n = text.count(old)
    if n != count:
        sys.exit(f"❌ [{label}] expected {count} match, found {n}. No file was changed.")
    return text.replace(old, new)

# ── if the v1 patch (calc_flops) is present, restore the originals first ──
for name in ('utility.py', 'trainer.py', 'option.py'):
    txt, _ = load(name)
    if ('def calc_flops(' in txt or 'utility.calc_flops(' in txt):
        bak = Path(name + '.bak')
        if not bak.exists():
            sys.exit(f"❌ {name} has the old patch and {name}.bak is missing. "
                     "Replace the three files with the complete ones that were sent, or restore the originals.")
        shutil.copy(bak, name)
        print(f"↩️  {name}: restored the original from {name}.bak before applying v2")

UTIL_NEW = '''# ============================================================
# Per-image FLOPs for ANY model (fvcore: 1 multiply-add = 1 FLOP, i.e. MACs;
# operations fvcore does not support are ignored).
# ============================================================
class _CallWithScale(torch.nn.Module):
    """Lets fvcore trace models whose forward is  forward(x, idx_scale)."""
    def __init__(self, net, idx_scale):
        super(_CallWithScale, self).__init__()
        self.net = net
        self.idx_scale = idx_scale

    def forward(self, x):
        return self.net(x, self.idx_scale)


class FlopsCounter():
    """
    GFLOPs of one forward pass for an exact LR input (1, C, H, W).
      * finds the real network inside the `model.Model` wrapper by itself
        (get_model() / .model / .module) and checks it holds ALL the parameters;
      * tries forward(x) and forward(x, idx_scale) and remembers the way that works;
      * FLOPs depend only on the input SHAPE: results are cached in memory and in
        `cache_file` (JSON, valid only for a model with the same parameter count),
        so later epochs / later runs do not trace again.
    Returns NaN (never raises) when FLOPs cannot be computed.
    """
    def __init__(self, model, cache_file=None):
        self.model = model
        self.cache_file = cache_file
        self.values = {}
        self.strategy = None
        self.disabled = False
        self.failures = 0
        self.warned = set()
        self.n_params = sum(p.numel() for p in model.parameters())
        self._load()

    # ---------------- cache ----------------
    def _load(self):
        if not self.cache_file or not os.path.isfile(self.cache_file):
            return
        try:
            import json
            with open(self.cache_file, 'r') as f:
                data = json.load(f)
            if data.get('n_params') == self.n_params:
                self.values = {k: float(v) for k, v in data.get('values', {}).items()}
        except Exception:
            self.values = {}

    def save(self):
        if not self.cache_file:
            return
        try:
            import json
            ok = {k: v for k, v in self.values.items() if v == v}      # drop NaN
            with open(self.cache_file, 'w') as f:
                json.dump({'n_params': self.n_params, 'values': ok}, f, indent=1)
        except Exception:
            pass

    # ---------------- finding the real network ----------------
    @staticmethod
    def _unwrap(model):
        net = model
        for _ in range(6):
            if isinstance(net, torch.nn.DataParallel):
                net = net.module
                continue
            getter = getattr(net, 'get_model', None)
            if callable(getter):
                try:
                    nxt = getter()
                except Exception:
                    nxt = None
                if nxt is not None and nxt is not net:
                    net = nxt
                    continue
            inner = getattr(net, 'model', None)
            if isinstance(inner, torch.nn.Module) and inner is not net:
                net = inner
                continue
            break
        return net

    def _candidates(self):
        inner = self._unwrap(self.model)
        out = []
        if inner is not self.model and sum(p.numel() for p in inner.parameters()) == self.n_params:
            out += [('inner network, forward(x)', inner, False),
                    ('inner network, forward(x, idx_scale)', inner, True)]
        out += [('whole model, forward(x, idx_scale)', self.model, True),
                ('whole model, forward(x)', self.model, False)]
        return out

    # ---------------- tracing ----------------
    @staticmethod
    def _trace(net, lr, idx_scale, use_scale):
        from fvcore.nn import FlopCountAnalysis
        target = _CallWithScale(net, idx_scale) if use_scale else net
        was_training = net.training
        net.eval()
        try:
            with torch.no_grad():
                fca = FlopCountAnalysis(target, (lr,))
                fca.unsupported_ops_warnings(False)
                fca.uncalled_modules_warnings(False)
                return float(fca.total()) / 1e9
        finally:
            net.train(was_training)

    def _warn(self, key, msg):
        if key not in self.warned:
            self.warned.add(key)
            print(msg)

    def __call__(self, lr, idx_scale=0):
        key = 'x'.join(str(s) for s in lr.shape)
        if key in self.values:
            return self.values[key]
        if self.disabled:
            return float('nan')
        try:
            import fvcore  # noqa: F401
        except ImportError:
            self._warn('fvcore', '[FLOPs] fvcore is not installed -> the FLOPs column will be empty. '
                                 'Install it with:  pip install fvcore')
            self.disabled = True
            return float('nan')

        order = [self.strategy] if self.strategy else self._candidates()
        first_err = None
        for name, net, use_scale in order:
            try:
                value = self._trace(net, lr, idx_scale, use_scale)
                if self.strategy is None:
                    self.strategy = (name, net, use_scale)
                    print('[FLOPs] tracing via: {}'.format(name))
                self.values[key] = value
                return value
            except Exception as e:
                if first_err is None:
                    first_err = (name, e)          # the first (most plausible) way is the most informative error
                try:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass

        self.failures += 1
        self._warn('fail', '[FLOPs] could not compute FLOPs for input {} (tried {} way(s); first error '
                           'while using "{}": {}: {}). NaN is written for such images.'.format(
                               tuple(lr.shape), len(order), first_err[0],
                               type(first_err[1]).__name__, first_err[1]))
        if self.strategy is None and self.failures >= 3:
            self.disabled = True
            print('[FLOPs] no way of tracing this model worked after 3 images -> FLOPs disabled for this run.')
        self.values[key] = float('nan')
        return float('nan')


'''
u, crlf_u = load('utility.py')
anchor_u = "# ============================================================\n# ✅ دالة make_optimizer"
u = must(u, anchor_u, UTIL_NEW + anchor_u, "utility.py: insert FlopsCounter")

t, crlf_t = load('trainer.py')
t = must(t, "        self.error_last = 1e8\n",
"""        self.error_last = 1e8

        # ---- per-image FLOPs in the test Excel files (cached by input shape) ----
        self.compute_flops = not getattr(args, 'no_flops', False)
        self.flops_counter = (
            utility.FlopsCounter(self.model, os.path.join(ckp.dir, 'flops_cache.json'))
            if self.compute_flops else None
        )
""", "trainer.py: init")
t = must(t, "                PSNR_values, SSIM_values, image_names = [], [], []\n",
"                PSNR_values, SSIM_values, image_names = [], [], []\n"
"                FLOPS_values, LR_sizes = [], []\n", "trainer.py: lists")
t = must(t, """                    image_names.append(filename[0])
                    lr, hr = self.prepare(lr, hr)
                    sr = self.model(lr, idx_scale)
""", """                    image_names.append(filename[0])
                    lr, hr = self.prepare(lr, hr)
                    if self.compute_flops:
                        FLOPS_values.append(self.flops_counter(lr, idx_scale))
                        LR_sizes.append('{}x{}'.format(lr.shape[-1], lr.shape[-2]))   # W x H
                    sr = self.model(lr, idx_scale)
""", "trainer.py: per image")
t = must(t, """                image_names.append('average')
                PSNR_values.append(avg_psnr)
""", """                image_names.append('average')
                PSNR_values.append(avg_psnr)
                if self.compute_flops:
                    _valid = [v for v in FLOPS_values if v == v]          # drop NaN
                    FLOPS_values.append(sum(_valid) / len(_valid) if _valid else float('nan'))
                    LR_sizes.append('mean')
""", "trainer.py: average row")

old_path = """                if self.args.load in ('.', ''):
                    xlsx_path = f'D:/Mohamed Morsi PhD- P2/src/experiment/{self.args.save}/results-{d.dataset.name}/{d.dataset.name}.xlsx'
                else:
                    xlsx_path = f'D:/Mohamed Morsi PhD- P2/src/experiment/{self.args.load}/results-{d.dataset.name}/{d.dataset.name}.xlsx'

                os.makedirs(os.path.dirname(xlsx_path), exist_ok=True)
"""
new_path = """                # dynamic path (any machine / any model folder):
                #   <src>/experiment/<save>/results-<dataset>/<dataset>.xlsx
                xlsx_path = self.ckp.get_path(
                    'results-{}'.format(d.dataset.name), '{}.xlsx'.format(d.dataset.name))
                os.makedirs(os.path.dirname(xlsx_path), exist_ok=True)
"""
t = must(t, old_path, new_path, "trainer.py: xlsx path")

old_df = """                if self.args.test_only:
                    avg_ssim = eval_acc_ssim / n_images
                    self.ckp.log_ssim[-1, idx_data, idx_scale] /= n_images
                    SSIM_values.append(avg_ssim)
                    DataFrame({
                        'image': np.array(image_names, dtype=object),
                        'psnr':  [round(v, 2) for v in PSNR_values],
                        'ssim':  [round(v, 4) for v in SSIM_values]
                    }).to_excel(xlsx_path, index=False)
                else:
                    DataFrame({
                        'image': np.array(image_names, dtype=object),
                        'psnr':  [round(v, 2) for v in PSNR_values],
                    }).to_excel(xlsx_path, index=False)
"""
new_df = """                if self.args.test_only:
                    avg_ssim = eval_acc_ssim / n_images
                    self.ckp.log_ssim[-1, idx_data, idx_scale] /= n_images
                    SSIM_values.append(avg_ssim)

                table = {
                    'image': np.array(image_names, dtype=object),
                    'psnr':  [round(v, 2) for v in PSNR_values],
                }
                if self.args.test_only:
                    table['ssim'] = [round(v, 4) for v in SSIM_values]
                if self.compute_flops:
                    table['lr_size'] = LR_sizes                                  # LR input W x H
                    table['flops_G'] = [round(v, 3) for v in FLOPS_values]       # G (fvcore, 1 MAC = 1 FLOP)
                try:
                    DataFrame(table).to_excel(xlsx_path, index=False)
                except PermissionError:                                          # file open in Excel
                    xlsx_path = xlsx_path.replace('.xlsx', '_alt.xlsx')
                    DataFrame(table).to_excel(xlsx_path, index=False)
                self.ckp.write_log('Excel saved: {}'.format(xlsx_path))
"""
t = must(t, old_df, new_df, "trainer.py: DataFrame")

old_log = """                    self.ckp.write_log(
                        '[{} x{}]\\tEpoch {}: Average PSNR: {:.3f} dB  (Best: {:.3f} @epoch {})'.format(
                            d.dataset.name, scale, epoch,
                            avg_psnr, best_psnr, best_epoch)
                    )
"""
t = must(t, old_log, old_log + """
                if self.compute_flops and len(FLOPS_values) > 0:
                    self.ckp.write_log(
                        '[{} x{}]\\tAverage FLOPs per image: {:.3f} G'.format(
                            d.dataset.name, scale, FLOPS_values[-1])
                    )
""", "trainer.py: log")
t = must(t, "        self.ckp.write_log('Forward: {:.2f}s\\n'.format(timer_test.toc()))\n",
"""        if self.compute_flops:
            self.flops_counter.save()          # reuse the per-shape FLOPs in later epochs / runs
        self.ckp.write_log('Forward: {:.2f}s\\n'.format(timer_test.toc()))
""", "trainer.py: save cache")

o, crlf_o = load('option.py')
if "--no_flops" not in o:
    o = must(o, """parser.add_argument('--save_gt', action='store_true',
                    help='save low-resolution and high-resolution images together')
""", """parser.add_argument('--save_gt', action='store_true',
                    help='save low-resolution and high-resolution images together')
parser.add_argument('--no_flops', action='store_true',
                    help='do not add the per-image FLOPs column to the test Excel files '
                         '(requires: pip install fvcore)')
""", "option.py: --no_flops")

for name, text, crlf in (('utility.py', u, crlf_u), ('trainer.py', t, crlf_t), ('option.py', o, crlf_o)):
    if not Path(name + '.bak').exists():
        shutil.copy(name, name + '.bak')
    Path(name).write_bytes((text.replace('\n', '\r\n') if crlf else text).encode('utf-8'))
print("✅ FLOPs patch v2 applied to utility.py, trainer.py, option.py (backups: *.bak)")
print("   → install once in the training environment:  pip install fvcore")
