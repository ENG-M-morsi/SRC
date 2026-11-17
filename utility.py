import os
import math
import time
import datetime
from multiprocessing import Process, Queue

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import imageio
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
from scipy.signal import convolve2d, windows


def bg_target(queue):
    while True:
        if not queue.empty():
            filename, tensor = queue.get()
            if filename is None:
                break
            imageio.imwrite(filename, tensor.numpy())


class timer():
    def __init__(self):
        self.acc = 0
        self.tic()

    def tic(self):
        self.t0 = time.time()

    def toc(self, restart=False):
        diff = time.time() - self.t0
        if restart:
            self.t0 = time.time()
        return diff

    def hold(self):
        self.acc += self.toc()

    def release(self):
        ret = self.acc
        self.acc = 0
        return ret

    def reset(self):
        self.acc = 0


class checkpoint():
    def __init__(self, args):
        self.args = args
        self.ok = True
        self.log = torch.Tensor()
        now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')

        # ✅ مسار ديناميكي — يعمل مع أي اسم مجلد
        _src_dir = os.path.dirname(os.path.abspath(__file__))
        _exp_dir = os.path.join(_src_dir, 'experiment')

        if not args.load:
            if not args.save:
                args.save = now
            self.dir = os.path.join(_exp_dir, args.save)
        else:
            self.dir = os.path.join(_exp_dir, args.save)
            if os.path.exists(self.dir):
                self.log = torch.load(
                    self.get_path('psnr_log.pt'),
                    weights_only=False
                )
                print('Continue from epoch {}...'.format(len(self.log)))
            else:
                args.load = ''

        if args.reset:
            os.system('rm -rf ' + self.dir)
            args.load = ''

        os.makedirs(self.dir, exist_ok=True)
        os.makedirs(self.get_path('model'), exist_ok=True)
        for d in args.data_test:
            os.makedirs(self.get_path('results-{}'.format(d)), exist_ok=True)

        open_type = 'a' if os.path.exists(self.get_path('log.txt')) else 'w'
        self.log_file = open(self.get_path('log.txt'), open_type)
        with open(self.get_path('config.txt'), open_type) as f:
            f.write(now + '\n\n')
            for arg in vars(args):
                f.write('{}: {}\n'.format(arg, getattr(args, arg)))
            f.write('\n')
        self.n_processes = 8

        # ================================================================
        # ✅ تحميل PSNR log مع دعم --resume N
        # ================================================================
        if args.load:
            psnr_log_path = self.get_path('psnr_log.pt')
            if os.path.exists(psnr_log_path):
                full_log = torch.load(psnr_log_path, weights_only=False)

                if args.resume == -1:
                    if len(self.log) == 0:
                        self.log = full_log

                elif args.resume > 0:
                    n = args.resume -1
                    if len(full_log) >= n:
                        self.log = full_log[:n]
                    else:
                        self.log = full_log
                    if os.getpid() == os.getpid():
                        print('Resuming from epoch {} — log has {} epochs.'.format(
                            n, len(self.log)))
                        if len(self.log) > 0:
                            best_val = self.log[:, 0, 0].max().item()
                            best_ep  = self.log[:, 0, 0].argmax().item() + 1
                            print('Best PSNR so far: {:.3f} dB @ epoch {}'.format(
                                best_val, best_ep))

    def get_path(self, *subdir):
        return os.path.join(self.dir, *subdir)

    def save(self, trainer, epoch, is_best=False):
        trainer.model.save(self.get_path('model'), epoch, is_best=is_best)
        trainer.loss.save(self.dir)
        trainer.loss.plot_loss(self.dir)
        self.plot_psnr()
        trainer.optimizer.save(self.dir)
        torch.save(self.log, self.get_path('psnr_log.pt'))

    def add_log(self, log):
        self.log = torch.cat([self.log, log])

    def write_log(self, log, refresh=False):
        print(log)
        self.log_file.write(log + '\n')
        if refresh:
            self.log_file.close()
            self.log_file = open(self.get_path('log.txt'), 'a')

    def done(self):
        self.log_file.close()

    def plot_psnr(self):
        total_epochs = len(self.log)
        if total_epochs == 0:
            return
        axis = np.linspace(1, total_epochs, total_epochs)

        for idx_data, d in enumerate(self.args.data_test):
            label = 'SR on {}'.format(d)
            fig = plt.figure()
            plt.title(label)
            for idx_scale, scale in enumerate(self.args.scale):
                plt.plot(
                    axis,
                    self.log[:, idx_data, idx_scale].numpy(),
                    label='Scale {}'.format(scale)
                )
            plt.legend()
            plt.xlabel('Epochs')
            plt.ylabel('PSNR')
            plt.grid(True)
            try:
                plt.savefig(self.get_path('test_{}.pdf'.format(d)))
            except PermissionError:
                alt_path = self.get_path('test_{}_alt.pdf'.format(d))
                plt.savefig(alt_path)
            plt.close(fig)

    def begin_background(self):
        self.queue = Queue()
        self.process = [
            Process(target=bg_target, args=(self.queue,))
            for _ in range(self.n_processes)
        ]
        for p in self.process:
            p.start()

    def end_background(self):
        for _ in range(self.n_processes):
            self.queue.put((None, None))
        while not self.queue.empty():
            time.sleep(1)
        for p in self.process:
            p.join()

    def save_results(self, dataset, filename, save_list, scale):
        if self.args.save_results:
            filename = self.get_path(
                'results-{}'.format(dataset.dataset.name),
                '{}_x{}_'.format(filename, scale)
            )
            postfix = ('SR', 'LR', 'HR')
            for v, p in zip(save_list, postfix):
                normalized = v[0].mul(255 / self.args.rgb_range)
                tensor_cpu = normalized.byte().permute(1, 2, 0).cpu()
                self.queue.put(('{}{}.png'.format(filename, p), tensor_cpu))


def quantize(img, rgb_range):
    pixel_range = 255 / rgb_range
    return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)


def calc_psnr(sr, hr, scale, rgb_range, dataset=None):
    h_min = min(sr.shape[-2], hr.shape[-2])
    w_min = min(sr.shape[-1], hr.shape[-1])
    sr = sr[..., :h_min, :w_min]
    hr = hr[..., :h_min, :w_min]

    if hr.nelement() == 1:
        return 0

    diff = (sr - hr) / rgb_range
    if dataset and dataset.dataset.benchmark:
        shave = scale
        if diff.size(1) > 1:
            gray_coeffs = [65.738, 129.057, 25.064]
            convert = diff.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
            diff = diff.mul(convert).sum(dim=1)
    else:
        shave = scale + 6

    valid = diff[..., shave:-shave, shave:-shave]
    mse = valid.pow(2).mean()
    return -10 * math.log10(mse)


def calc_ssim(img1, img2, scale=2, rgb_range=255, dataset=None):
    h_min = min(img1.shape[-2], img2.shape[-2])
    w_min = min(img1.shape[-1], img2.shape[-1])
    img1  = img1[..., :h_min, :w_min]
    img2  = img2[..., :h_min, :w_min]

    if dataset and dataset.dataset.benchmark:
        border = math.ceil(scale)
    else:
        border = math.ceil(scale) + 6

    if rgb_range != 255:
        img1 = img1 * 255.0 / rgb_range
        img2 = img2 * 255.0 / rgb_range

    img1 = img1.data.squeeze().float().clamp(0, 255).cpu().numpy()
    img1 = np.transpose(img1, (1, 2, 0))
    img2 = img2.data.squeeze().float().clamp(0, 255).cpu().numpy()
    img2 = np.transpose(img2, (1, 2, 0))

    img1_y = np.dot(img1, [65.738, 129.057, 25.064]) / 255.0 + 16.0
    img2_y = np.dot(img2, [65.738, 129.057, 25.064]) / 255.0 + 16.0

    if not img1_y.shape == img2_y.shape:
        raise ValueError('Input images must have the same dimensions.')

    h, w = img1_y.shape[:2]
    if border > 0:
        img1_y = img1_y[border:h - border, border:w - border]
        img2_y = img2_y[border:h - border, border:w - border]

    return ssim(img1_y, img2_y)


def ssim(img1, img2):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    kernel_1d = windows.gaussian(11, std=1.5)
    kernel    = np.outer(kernel_1d, kernel_1d)
    kernel    = kernel / np.sum(kernel)

    mu1 = convolve2d(img1, kernel, mode='same', boundary='symm')
    mu2 = convolve2d(img2, kernel, mode='same', boundary='symm')

    mu1_sq  = mu1 * mu1
    mu2_sq  = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = convolve2d(img1 * img1, kernel, mode='same', boundary='symm') - mu1_sq
    sigma2_sq = convolve2d(img2 * img2, kernel, mode='same', boundary='symm') - mu2_sq
    sigma12   = convolve2d(img1 * img2, kernel, mode='same', boundary='symm') - mu1_mu2

    numerator   = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    ssim_map    = numerator / (denominator + 1e-12)

    return ssim_map.mean()


# ============================================================
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


# ============================================================
# ✅ دالة make_optimizer المعدلة لدعم Cosine Annealing
# ============================================================
def make_optimizer(args, target):
    trainable = filter(lambda x: x.requires_grad, target.parameters())
    kwargs_optimizer = {'lr': args.lr, 'weight_decay': args.weight_decay}

    if args.optimizer == 'SGD':
        optimizer_class = optim.SGD
        kwargs_optimizer['momentum'] = args.momentum
    elif args.optimizer == 'ADAM':
        optimizer_class = optim.Adam
        kwargs_optimizer['betas'] = tuple(args.betas)
        kwargs_optimizer['eps'] = args.epsilon
    elif args.optimizer == 'ADAMW':
        optimizer_class = optim.AdamW
        kwargs_optimizer['betas'] = tuple(args.betas)
        kwargs_optimizer['eps'] = args.epsilon
    elif args.optimizer == 'RMSprop':
        optimizer_class = optim.RMSprop
        kwargs_optimizer['eps'] = args.epsilon

    # ------------------- تحديد نوع الـ scheduler -------------------
    decay_type = getattr(args, 'decay_type', 'step')   # افتراضي step للتوافق القديم
    if decay_type == 'cosine':
        scheduler_class = lrs.CosineAnnealingLR
        kwargs_scheduler = {'T_max': args.epochs, 'eta_min': getattr(args, 'lr_min', 1e-6)}
    elif decay_type == 'cosine_restart':
        scheduler_class = lrs.CosineAnnealingWarmRestarts
        kwargs_scheduler = {
            'T_0': getattr(args, 'T0', 10),
            'T_mult': getattr(args, 'Tmult', 2),
            'eta_min': getattr(args, 'lr_min', 1e-6)
        }
    else:   # step decay (default)
        scheduler_class = lrs.MultiStepLR
        milestones = list(map(lambda x: int(x), args.decay.split('-')))
        kwargs_scheduler = {'milestones': milestones, 'gamma': args.gamma}

    # ============================================================
    # CustomOptimizer يدعم كلاً من MultiStepLR و CosineAnnealingLR
    # ============================================================
    class CustomOptimizer(optimizer_class):
        def __init__(self, *args, **kwargs):
            super(CustomOptimizer, self).__init__(*args, **kwargs)

        def _register_scheduler(self, scheduler_class, **kwargs):
            self.scheduler = scheduler_class(self, **kwargs)

        def save(self, save_dir):
            torch.save(self.state_dict(), self.get_dir(save_dir))

        def load(self, load_dir, epoch=1):
            # استعادة حالة المحسن (اختياري، لكننا لا نحملها للحفاظ على LR الجديد)
            # ونحرك الـ scheduler بالعدد المناسب من الخطوات
            if epoch > 1:
                for _ in range(epoch - 1):
                    self.scheduler.step()

        def get_dir(self, dir_path):
            return os.path.join(dir_path, 'optimizer.pt')

        def schedule(self):
            self.scheduler.step()

        def get_lr(self):
            return self.scheduler.get_last_lr()[0]

        def get_last_epoch(self):
            return self.scheduler.last_epoch + 1

    optimizer = CustomOptimizer(trainable, **kwargs_optimizer)
    optimizer._register_scheduler(scheduler_class, **kwargs_scheduler)
    return optimizer