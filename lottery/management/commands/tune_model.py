"""
management command: tune_model
Hyperparameter tuning (random search) for the 2-digit predictor, scored by honest walk-forward
backtesting on historical draws.

Protocol (so the "best accuracy" is not just overfitting):
  * Draw t is predicted using ONLY draws before t (incl. transition tables, built incrementally).
  * Targets are split in time: TUNE (older 70%) / TEST (newest 30%, never used for selection).
  * Best config is picked on TUNE only, then reported on TEST.
  * NULL CONTROL: the same search is run on shuffled history (pure noise). If the "best" score on
    noise looks as good as on real data, the improvement is selection luck, not real skill.

Usage:
  python manage.py tune_model
  python manage.py tune_model --trials 1500 --nulls 5 --seed 7 --save
"""
import json
import math
import random
import time
from pathlib import Path

import numpy as np
from django.core.management.base import BaseCommand

from lottery.models import LotteryResult

PICKS = 3
WARMUP = 100          # draws used only as history before the first scored target

# Current production settings (predictor.py bayes_pool_v13) as baseline
BASELINE = dict(window=15, decay=0.80, anti_n=3, anti_f=0.4, ctw=0.50, w_l1=1.3, w_l2=0.8, w_sum=0.5)

SPACE = dict(
    window=[5, 8, 10, 15, 20, 30, 60, 120, 300],
    decay=[0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 1.0],
    anti_n=[0, 1, 2, 3, 5, 8],
    anti_f=[0.0, 0.2, 0.4, 0.7, 1.0],
    ctw=[0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0],
    w_l1=[0.0, 0.5, 1.0, 1.3, 2.0],
    w_l2=[0.0, 0.4, 0.8, 1.5],
    w_sum=[0.0, 0.25, 0.5, 1.0],
)


def sample_params(rng):
    return {k: rng.choice(v) for k, v in SPACE.items()}


def hits_for(params, a, b, sums):
    """Walk-forward hit vector (1/0) for targets t = WARMUP .. n-1 (no look-ahead)."""
    n = len(a)
    pair = a * 10 + b
    L1 = np.zeros((100, 100)); L2 = np.zeros((100, 100)); LS = np.zeros((64, 100))

    def add(i):  # transition ending at draw i (needs i >= 2)
        L1[pair[i - 1], pair[i]] += 1
        L2[pair[i - 2], pair[i]] += 1
        LS[sums[i - 1], pair[i]] += 1

    for i in range(2, WARMUP):
        add(i)

    W, r = params['window'], params['decay']
    wts_full = r ** np.arange(W - 1, -1, -1)           # oldest -> newest
    idx = np.arange(100)
    out = np.zeros(n - WARMUP, dtype=np.int8)
    for t in range(WARMUP, n):
        w = min(W, t)
        ww = wts_full[-w:]
        d1 = np.bincount(a[t - w:t], weights=ww, minlength=10)
        d2 = np.bincount(b[t - w:t], weights=ww, minlength=10)
        score = np.outer(d1, d2).ravel()
        if params['anti_n'] > 0 and params['anti_f'] < 1.0:
            rec = pair[t - params['anti_n']:t]
            score[rec] *= params['anti_f']
        if params['ctw'] > 0:
            c = (params['w_l1'] * L1[pair[t - 1]] + params['w_l2'] * L2[pair[t - 2]]
                 + params['w_sum'] * LS[sums[t - 1]])
            score = score + params['ctw'] * c
        top = np.lexsort((idx, -score))[:PICKS]         # score desc, tie -> smaller pair
        out[t - WARMUP] = 1 if pair[t] in top else 0
        add(t)
    return out


def split_idx(n_targets):
    return int(n_targets * 0.7)


def search(a, b, sums, trials, seed):
    """Random search; returns (best_params, tune_acc, test_acc, tune_curve)."""
    rng = random.Random(seed)
    cands = [dict(BASELINE)] + [sample_params(rng) for _ in range(trials)]
    cut = split_idx(len(a) - WARMUP)
    best = None
    for p in cands:
        h = hits_for(p, a, b, sums)
        tune = h[:cut].mean() * 100
        test = h[cut:].mean() * 100
        if best is None or tune > best[1]:
            best = (p, tune, test)
    return best


def load():
    rows = list(LotteryResult.objects.exclude(first_prize='').order_by('draw_date'))
    digs = np.array([[int(c) for c in r.first_prize.zfill(6)[-6:]] for r in rows])
    a, b = digs[:, -2].astype(int), digs[:, -1].astype(int)
    sums = digs.sum(axis=1).astype(int)
    return a, b, sums, digs


class Command(BaseCommand):
    help = "Random-search hyperparameter tuning of the 2-digit predictor with walk-forward validation"

    def add_arguments(self, parser):
        parser.add_argument('--trials', type=int, default=600)
        parser.add_argument('--nulls', type=int, default=3, help='shuffled-history control runs')
        parser.add_argument('--seed', type=int, default=42)
        parser.add_argument('--save', action='store_true', help='write tuning_results.json')

    def handle(self, *args, **o):
        a, b, sums, digs = load()
        n = len(a)
        n_t = n - WARMUP
        cut = split_idx(n_t)
        chance = PICKS / 100 * 100
        self.stdout.write(f"draws={n}  scored targets={n_t}  tune={cut}  test={n_t - cut}")
        self.stdout.write(f"pure chance = {chance:.1f}%  (1 sd on test ~ "
                          f"{100 * math.sqrt(.03 * .97 / (n_t - cut)):.1f}pp, "
                          f"on tune ~ {100 * math.sqrt(.03 * .97 / cut):.1f}pp)")

        base_h = hits_for(BASELINE, a, b, sums)
        self.stdout.write(f"baseline (current v13): tune {base_h[:cut].mean()*100:.1f}%  "
                          f"test {base_h[cut:].mean()*100:.1f}%")

        t0 = time.time()
        p, tune, test = search(a, b, sums, o['trials'], o['seed'])
        self.stdout.write(f"\nREAL data  best-of-{o['trials']}: tune {tune:.1f}%  -> TEST {test:.1f}%  "
                          f"({time.time()-t0:.0f}s)")
        self.stdout.write(f"  params: {p}")

        null_rows = []
        rng = np.random.default_rng(o['seed'])
        for k in range(o['nulls']):
            idx = rng.permutation(n)
            sa, sb = a[idx], b[idx]
            ssum = sums[idx]
            _, nt, nte = search(sa, sb, ssum, o['trials'], o['seed'] + 1 + k)
            null_rows.append((nt, nte))
            self.stdout.write(f"NULL shuffle #{k+1} best-of-{o['trials']}: tune {nt:.1f}%  -> TEST {nte:.1f}%")

        if null_rows:
            self.stdout.write(
                f"\nNULL avg: tune {np.mean([x[0] for x in null_rows]):.1f}%  "
                f"test {np.mean([x[1] for x in null_rows]):.1f}%")
        verdict = ("No real edge: tuned test accuracy is within noise of chance/baseline."
                   if test < chance + 2 * 100 * math.sqrt(.03 * .97 / (n_t - cut))
                   else "Test accuracy beats chance by >2 sd - verify on future draws before trusting.")
        self.stdout.write(f"\nVERDICT: {verdict}")

        if o['save']:
            Path('tuning_results.json').write_text(json.dumps(dict(
                draws=n, best_params=p, tune_acc=tune, test_acc=test,
                baseline_tune=float(base_h[:cut].mean() * 100), baseline_test=float(base_h[cut:].mean() * 100),
                null=null_rows, chance=chance, verdict=verdict), indent=2), encoding='utf-8')
            self.stdout.write("saved tuning_results.json")
