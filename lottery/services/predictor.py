"""
Prediction Engine สำหรับหวยลาวพัฒนา (v4 - Bayesian Pool + Walk-Forward Validation)

เขียนใหม่ทั้งหมด (ไม่ได้ต่อยอดจาก ensemble_v3) แนวคิดหลัก:

1. ทุกตำแหน่งหลัก → ประมาณ "ความน่าจะเป็นของเลข 0-9" ด้วย 4 สัญญาณแบบ Bayesian
   (Dirichlet smoothing แรงๆ เพื่อไม่ให้ overfit ข้อมูลน้อย):
     long     ความถี่ระยะยาว
     recent   ความถี่ถ่วงน้ำหนักงวดใหม่ (half-life 15 งวด)
     overdue  เลขที่หายไปนาน
     markov   เลขตำแหน่งนี้ งวดก่อนออกอะไร → งวดนี้ออกอะไรบ่อย
2. รวมสัญญาณด้วย log-linear pooling  p(d) ∝ Π q_k(d)^w_k
   โดย w_k เลือกจาก walk-forward validation (ทำนายย้อนหลังทีละงวดโดยใช้ข้อมูลก่อนหน้างวดนั้นเท่านั้น)
3. Significance gate: ถ้าชุดน้ำหนักที่ดีที่สุดไม่ชนะ "การสุ่มเท่าๆ กัน" อย่างมีนัยสำคัญ (t < 3)
   → ไม่เชื่อสัญญาณ ใช้น้ำหนักเกือบศูนย์ (ใกล้ uniform) แทนการแต่งเรื่อง
4. ไม่มีการสุ่ม/seed จากวันที่: ผลลัพธ์ขึ้นกับข้อมูลเท่านั้น (deterministic)
5. 'confidence' = ความน่าจะเป็นจริงที่ชุดเลข 2 ตัวที่ทายจะถูก ตามที่โมเดลคำนวณ (ไม่ใช่ตัวเลขปลอม 60-90%)
6. ทำนายงวด D ใช้เฉพาะผลที่ออกก่อนวัน D เท่านั้น (ไม่มี look-ahead)

หมายเหตุ: ทดสอบกับผล 600 งวดย้อนหลังแล้ว ผลหวยลาวพัฒนาไม่ต่างจากการสุ่มอย่างมีนัยสำคัญ
ดังนั้นโมเดลนี้ "ซื่อสัตย์" ว่าโอกาสถูกก็ใกล้เคียงการสุ่ม แต่จะไม่โกหกเรื่อง confidence
"""
import logging
import math
import threading
from collections import Counter
from datetime import date

import numpy as np

logger = logging.getLogger(__name__)

MODEL_VERSION = 'bayes_pool_v4'

# ── Hyper-parameters ─────────────────────────────
HISTORY_LIMIT      = 400    # จำนวนงวดล่าสุดที่ใช้
MIN_TRAIN          = 20     # ต้องมีอย่างน้อยกี่งวดก่อนเริ่มทำนาย
VALIDATION_DRAWS   = 60     # จำนวนงวดที่ใช้ validate น้ำหนัก
MIN_VALIDATION     = 20     # ต่ำกว่านี้ → ไม่ fit
PRIOR_STRENGTH     = 5.0    # Dirichlet pseudo-count ต่อเลขแต่ละตัว
MARKOV_PRIOR       = 20.0
RECENT_HALFLIFE    = 15.0
SIGNIFICANCE_T     = 3.0    # t-stat ขั้นต่ำที่ยอมเชื่อสัญญาณ
TIEBREAK_WEIGHT    = 0.05   # น้ำหนักเล็กๆ ไว้ตัดสินเสมอ เมื่อไม่มีสัญญาณที่เชื่อถือได้
WEIGHT_GRID        = (0.0, 0.5, 1.0, 1.5)
SIGNAL_NAMES       = ('long', 'recent', 'overdue', 'markov')
N_SIGNALS          = len(SIGNAL_NAMES)

_GRID = np.array(np.meshgrid(*([WEIGHT_GRID] * N_SIGNALS), indexing='ij')).reshape(N_SIGNALS, -1).T  # (G, K)


# ──────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────

def _get_history(limit=HISTORY_LIMIT, before: date = None):
    """ดึงผลล่าสุดจาก DB (ใหม่→เก่า) เฉพาะงวดที่ก่อนวัน `before` (กัน look-ahead)"""
    from lottery.models import LotteryResult
    qs = LotteryResult.objects.exclude(first_prize='')
    if before is not None:
        qs = qs.filter(draw_date__lt=before)
    return list(qs.order_by('-draw_date')[:limit])


def _extract_digits(lottery_result) -> list:
    """แปลง first_prize → list of int"""
    return [int(c) for c in lottery_result.first_prize if c.isdigit()]


def _pad_or_trim(digits, length=6, fill=0) -> list:
    d = list(digits[:length])
    while len(d) < length:
        d.append(fill)
    return d


def _to_matrix(draws_oldest_first, prize_len=6) -> np.ndarray:
    return np.array(
        [_pad_or_trim(_extract_digits(lr), prize_len) for lr in draws_oldest_first],
        dtype=np.int64,
    ).reshape(-1, prize_len)


# ──────────────────────────────────────────────
#  Signals: 4 ความน่าจะเป็นต่อหนึ่งตำแหน่ง
# ──────────────────────────────────────────────

def _signals(train: np.ndarray) -> np.ndarray:
    """
    train: (m, L) ผลที่รู้แล้ว เรียง เก่า→ใหม่
    return: (L, K, 10) ความน่าจะเป็นของแต่ละสัญญาณ (แต่ละแถวรวม = 1)
    """
    m, L = train.shape
    out = np.empty((L, N_SIGNALS, 10))
    age = np.arange(m)[::-1]
    w_recent = 0.5 ** (age / RECENT_HALFLIFE)

    for p in range(L):
        col = train[:, p]

        counts = np.bincount(col, minlength=10).astype(float)
        long_q = (counts + PRIOR_STRENGTH) / (counts.sum() + 10 * PRIOR_STRENGTH)

        rec = np.bincount(col, weights=w_recent, minlength=10)
        recent_q = (rec + PRIOR_STRENGTH) / (rec.sum() + 10 * PRIOR_STRENGTH)

        gaps = np.empty(10)
        for d in range(10):
            idx = np.nonzero(col == d)[0]
            gaps[d] = (m - 1 - idx[-1]) if idx.size else m
        overdue_q = np.sqrt(gaps + 1.0)
        overdue_q /= overdue_q.sum()

        prev = col[-1]
        nxt = col[1:][col[:-1] == prev]
        nxt_counts = np.bincount(nxt, minlength=10).astype(float)
        markov_q = (nxt_counts + MARKOV_PRIOR * long_q) / (nxt_counts.sum() + MARKOV_PRIOR)

        out[p, 0], out[p, 1], out[p, 2], out[p, 3] = long_q, recent_q, overdue_q, markov_q
    return out


def _signal_table(X: np.ndarray, start: int) -> np.ndarray:
    """
    table[i] = _signals(X[:start+i]) → สัญญาณที่ "รู้ได้ ณ ก่อนงวด t=start+i"
    สร้างถึง t = len(X) (คือสัญญาณสำหรับทำนายงวดถัดไป)
    """
    n = len(X)
    return np.stack([_signals(X[:t]) for t in range(start, n + 1)])


# ──────────────────────────────────────────────
#  Fit น้ำหนักด้วย walk-forward validation
# ──────────────────────────────────────────────

def _log_softmax_pool(weights: np.ndarray, log_ratio: np.ndarray) -> np.ndarray:
    """log_ratio (..., K, 10); weights (K,) → log-prob (..., 10)"""
    s = np.tensordot(log_ratio, weights, axes=([-2], [0]))
    s = s - s.max(axis=-1, keepdims=True)
    return s - np.log(np.exp(s).sum(axis=-1, keepdims=True))


def _fit_weights(table: np.ndarray, start: int, X: np.ndarray, t: int) -> dict:
    """
    เลือกน้ำหนักสัญญาณสำหรับทำนายงวด t โดยดู validation window = [t-V, t)
    แต่ละงวดใน window ถูกทำนายด้วยสัญญาณที่รู้ก่อนงวดนั้นเท่านั้น
    """
    uniform_w = np.full(N_SIGNALS, TIEBREAK_WEIGHT)
    info = {'weights': uniform_w, 'gated': True, 't_stat': 0.0, 'gain': 0.0}

    lo = max(start, t - VALIDATION_DRAWS)
    if t - lo < MIN_VALIDATION:
        return info

    sig = table[lo - start: t - start]          # (V, L, K, 10)
    actual = X[lo:t]                            # (V, L)
    V, L = actual.shape
    log_ratio = np.log(10.0 * sig)              # log(q / uniform)

    # s[v,l,g,d] = Σ_k grid[g,k] * log_ratio[v,l,k,d]
    s = np.einsum('vlkd,gk->vlgd', log_ratio, _GRID)
    s = s - s.max(axis=-1, keepdims=True)
    log_z = np.log(np.exp(s).sum(axis=-1))                                  # (V,L,G)
    s_actual = np.take_along_axis(s, actual[:, :, None, None], axis=3)[..., 0]  # (V,L,G)
    # ln p(actual) - ln(1/10)  → กำไรเทียบกับการสุ่ม
    diff = (s_actual - log_z + math.log(10.0)).reshape(V * L, -1)           # (N,G)

    means = diff.mean(axis=0)
    g = int(np.argmax(means))
    sd = diff[:, g].std(ddof=1) or 1e-9
    t_stat = float(means[g] / (sd / math.sqrt(diff.shape[0])))

    info.update(t_stat=t_stat, gain=float(means[g]))
    if t_stat >= SIGNIFICANCE_T and means[g] > 0:
        info.update(weights=_GRID[g].copy(), gated=False)
    return info


# ──────────────────────────────────────────────
#  Forecast
# ──────────────────────────────────────────────

def _forecast(table: np.ndarray, start: int, X: np.ndarray, t: int) -> dict:
    """ทำนายงวด t (ใช้ X[:t] เท่านั้น) → dict ผลลัพธ์"""
    n_pos = X.shape[1]
    fit = _fit_weights(table, start, X, t)
    sig = table[t - start]                                   # (L, K, 10)
    logp = _log_softmax_pool(fit['weights'], np.log(10.0 * sig))
    P = np.exp(logp)                                         # (L, 10) ความน่าจะเป็นต่อตำแหน่ง

    predicted_first = ''.join(str(int(np.argmax(P[p]))) for p in range(n_pos))

    # ── 2 ตัว: นับทั้ง "2 ตัวท้าย" และ "2 ตัวบน (หลักที่ 3-4)" ตามกติกาตรวจของระบบ ──
    pairs = {}
    for a in range(10):
        for b in range(10):
            bottom = P[n_pos - 2][a] * P[n_pos - 1][b]
            top = P[2][a] * P[3][b] if n_pos >= 4 else 0.0
            pairs[f"{a}{b}"] = (bottom, top)
    ranked2 = sorted(pairs, key=lambda k: (pairs[k][0] + pairs[k][1], k), reverse=True)
    two = ranked2[:3]
    p_bottom = sum(pairs[k][0] for k in two)
    p_top = sum(pairs[k][1] for k in two)
    p_two_hit = p_bottom + p_top - p_bottom * p_top

    # ── 3 ตัวท้าย ──
    if n_pos >= 3:
        joint3 = {}
        for a in range(10):
            for b in range(10):
                for c in range(10):
                    joint3[f"{a}{b}{c}"] = P[n_pos - 3][a] * P[n_pos - 2][b] * P[n_pos - 1][c]
        three = sorted(joint3, key=lambda k: (joint3[k], k), reverse=True)[:3]
    else:
        three = ["000", "001", "002"]

    vote_breakdown = []
    for p in range(n_pos):
        ranked = sorted(range(10), key=lambda d: (P[p][d], -d), reverse=True)
        vote_breakdown.append([(str(d), round(float(P[p][d]) * 100, 1)) for d in ranked])

    overall = P.sum(axis=0)
    order = sorted(range(10), key=lambda d: (overall[d], -d), reverse=True)

    return {
        'predicted_first':  predicted_first,
        'two_digit_pairs':  two,
        'three_digit_sets': three,
        'confidence':       round(float(p_two_hit) * 100, 1),
        'key_digit':        str(order[0]),
        'secondary_digit':  str(order[1]),
        'vote_breakdown':   vote_breakdown,
        'signal': {
            'weights': {n: float(w) for n, w in zip(SIGNAL_NAMES, fit['weights'])},
            't_stat':  round(fit['t_stat'], 2),
            'gain':    round(fit['gain'], 4),
            'reliable_signal': not fit['gated'],
        },
    }


def predict_from_history(history: list, prize_len=6) -> dict:
    """
    history: list ของ LotteryResult เรียง ใหม่→เก่า (เฉพาะที่รู้แล้ว ณ ตอนทำนาย)
    """
    ordered = list(reversed(history))
    if len(ordered) < 5:
        n_pos = max(prize_len, 1)
        return {
            'predicted_first': '0' * prize_len,
            'two_digit_pairs': ['00', '01', '02'],
            'three_digit_sets': ['000', '001', '002'],
            'confidence': 5.9, 'key_digit': '0', 'secondary_digit': '1',
            'vote_breakdown': [[(str(d), 10.0) for d in range(10)] for _ in range(n_pos)],
            'signal': {'weights': {}, 't_stat': 0.0, 'gain': 0.0, 'reliable_signal': False},
        }
    X = _to_matrix(ordered, prize_len)
    n = len(X)
    start = min(max(MIN_TRAIN, n - VALIDATION_DRAWS), n)
    table = _signal_table(X, start)
    return _forecast(table, start, X, n)


def walk_forward(draws_oldest_first: list, prize_len=6, first_target=MIN_TRAIN):
    """
    จำลอง "ถ้าทายตอนนั้น" ทีละงวด (ใช้เฉพาะผลก่อนงวดนั้น)
    yield (index, forecast_dict)   สำหรับ index >= first_target
    """
    X = _to_matrix(draws_oldest_first, prize_len)
    n = len(X)
    first_target = max(first_target, MIN_TRAIN)
    if n <= first_target:
        return
    # table[i] ↔ t = MIN_TRAIN + i ใช้ X[:t] เท่านั้น จึงไม่มี look-ahead
    table = _signal_table(X[:n - 1], MIN_TRAIN)
    for t in range(first_target, n):
        yield t, _forecast(table, MIN_TRAIN, X, t)


# ──────────────────────────────────────────────
#  Public API
# ──────────────────────────────────────────────

def _result_to_row(res: dict) -> dict:
    return {
        'predicted_first':  res['predicted_first'],
        'predicted_two':    ', '.join(res['two_digit_pairs']),
        'predicted_three':  ', '.join(res['three_digit_sets']),
        'confidence':       res['confidence'],
        'model_used':       MODEL_VERSION,
        'key_digit':        res['key_digit'],
        'secondary_digit':  res['secondary_digit'],
        'vote_breakdown':   res['vote_breakdown'],
    }


def predict_next(target_date: date = None) -> dict:
    """ทำนายงวดถัดไป โดยใช้เฉพาะผลที่ออกก่อน target_date"""
    if target_date is None:
        from lottery.services.utils import get_next_draw_date
        target_date = get_next_draw_date()

    history = _get_history(limit=HISTORY_LIMIT, before=target_date)
    if not history:
        logger.warning("ไม่มีข้อมูลประวัติ ไม่สามารถทำนายได้")
        return {
            'predicted_first': 'N/A', 'predicted_two': 'N/A', 'predicted_three': 'N/A',
            'confidence': 0.0, 'note': 'กรุณาดึงข้อมูลประวัติก่อน',
        }

    prize_len = len(_extract_digits(history[0])) or 6
    res = predict_from_history(history, prize_len)
    logger.info(f"[{MODEL_VERSION}] target={target_date} signal={res['signal']}")
    row = _result_to_row(res)
    row['based_on'] = len(history)
    row['signal'] = res['signal']
    return row


def save_prediction(target_date: date = None) -> 'Prediction':
    """สร้างและบันทึก prediction ลง DB (แทนที่ของเดิมของวันนั้น)"""
    from lottery.models import Prediction, LotteryResult

    if target_date is None:
        from lottery.services.utils import get_next_draw_date
        target_date = get_next_draw_date()

    Prediction.objects.filter(target_date=target_date).delete()
    result = predict_next(target_date)
    actual = LotteryResult.objects.filter(draw_date=target_date).first()

    pred = Prediction.objects.create(
        target_date=target_date,
        predicted_first=result['predicted_first'],
        predicted_two=result['predicted_two'],
        predicted_three=result['predicted_three'],
        confidence=result['confidence'],
        model_used=result.get('model_used', MODEL_VERSION),
        key_digit=result.get('key_digit', 'N/A'),
        secondary_digit=result.get('secondary_digit', 'N/A'),
        vote_breakdown=result.get('vote_breakdown', None),
        actual_result=actual,
    )
    if actual:
        pred.evaluate()

    logger.info(f"บันทึก prediction งวด {target_date}: {result['predicted_first']} "
                f"(P(2 ตัวถูก) ≈ {result['confidence']}%)")
    return pred


def prediction_is_stale(prediction) -> bool:
    """prediction ต้องสร้างใหม่ถ้า: ไม่มี / คนละโมเดล / ข้อมูลไม่ครบ / มีผลใหม่ออกหลังสร้าง"""
    from lottery.models import LotteryResult
    if prediction is None:
        return True
    if prediction.model_used != MODEL_VERSION:
        return True
    if prediction.key_digit == 'N/A' or prediction.vote_breakdown is None:
        return True
    newest = (LotteryResult.objects.filter(draw_date__lt=prediction.target_date)
              .exclude(first_prize='').order_by('-draw_date').first())
    return bool(newest and newest.created_at > prediction.created_at)


def rebuild_history(last_n: int = 120) -> int:
    """
    ลบ prediction เก่าที่มาจากโมเดลอื่น แล้วสร้างย้อนหลังใหม่แบบ walk-forward
    (แต่ละงวดทายด้วยข้อมูลก่อนงวดนั้นเท่านั้น) เพื่อให้สถิติความแม่นยำเป็นของจริง
    """
    from lottery.models import LotteryResult, Prediction

    results = list(LotteryResult.objects.exclude(first_prize='').order_by('draw_date'))
    Prediction.objects.exclude(model_used=MODEL_VERSION).delete()
    if len(results) <= MIN_TRAIN:
        return 0

    prize_len = len(_extract_digits(results[-1])) or 6
    first_target = max(MIN_TRAIN, len(results) - last_n)
    target_dates = {results[t].draw_date for t in range(first_target, len(results))}
    Prediction.objects.filter(target_date__in=target_dates).delete()

    created = 0
    for t, res in walk_forward(results, prize_len, first_target=first_target):
        actual = results[t]
        pred = Prediction.objects.create(
            target_date=actual.draw_date, actual_result=actual, **_result_to_row(res)
        )
        pred.evaluate()
        created += 1
    logger.info(f"[{MODEL_VERSION}] rebuilt {created} walk-forward predictions")
    return created


_refresh_lock = threading.Lock()
_refresh_running = False
_last_backfill_at = None
BACKFILL_COOLDOWN_SEC = 6 * 3600
EARLIEST_MTHAI_DATE = date(2022, 12, 1)   # MThai มีข้อมูลย้อนไปถึงปลายปี 2022


def has_missing_results() -> bool:
    """
    มีงวดที่ขาด? นับเฉพาะวันที่ควรมีผลจริง: จันทร์-ศุกร์ตั้งแต่ DAILY_SCHEDULE_START
    ยกเว้นวันงดออกรางวัล (NO_DRAW_DATES) และผลล่าสุดต้องครบถึงปัจจุบัน
    หรือประวัติสั้นกว่าที่ MThai มี
    """
    from datetime import time, timedelta
    from django.utils import timezone
    from lottery.models import LotteryResult
    from lottery.services.fetcher import DAILY_SCHEDULE_START, NO_DRAW_DATES

    dates = set(LotteryResult.objects.exclude(first_prize='').values_list('draw_date', flat=True))
    if not dates:
        return True
    if min(dates) > EARLIEST_MTHAI_DATE:
        return True

    now = timezone.localtime()
    last_due = now.date() if now.time() >= time(21, 0) else now.date() - timedelta(days=1)

    day = DAILY_SCHEDULE_START
    while day <= last_due:
        if day.weekday() < 5 and day not in NO_DRAW_DATES and day not in dates:
            return True
        day += timedelta(days=1)
    return False


def backfill_missing_results() -> int:
    """ดึงผลที่ขาดมาเติมให้ครบถึงปัจจุบัน คืนจำนวนงวดที่เพิ่มเข้ามา"""
    from lottery.models import LotteryResult
    from lottery.services.fetcher import fetch_history
    first = LotteryResult.objects.exclude(first_prize='').order_by('draw_date').values_list('draw_date', flat=True).first()
    deep = first is None or first > EARLIEST_MTHAI_DATE
    return len(fetch_history(pages=100 if deep else 5))


def kick_model_refresh():
    """
    Background thread (ไม่บล็อก request, ทำครั้งเดียวต่อรอบ):
      1) ถ้ามีงวดขาด → ดึงจาก MThai มาเติมให้ครบ
      2) ถ้ามีคำทำนายจากโมเดลเก่า หรือเพิ่งเติมผลเข้ามา → rebuild ประวัติแบบ walk-forward
    """
    global _refresh_running, _last_backfill_at
    import time as _time
    from lottery.models import Prediction
    try:
        legacy = Prediction.objects.exclude(model_used=MODEL_VERSION).exists()
        backfill_due = (
            (_last_backfill_at is None or _time.time() - _last_backfill_at > BACKFILL_COOLDOWN_SEC)
            and has_missing_results()
        )
        if not legacy and not backfill_due:
            return
    except Exception:
        return
    with _refresh_lock:
        if _refresh_running:
            return
        _refresh_running = True

    def _job():
        global _refresh_running, _last_backfill_at
        try:
            from django.db import connection
            added = 0
            if backfill_due:
                _last_backfill_at = _time.time()
                added = backfill_missing_results()
                logger.info(f"backfill: เติมผลที่ขาดได้ {added} งวด")
            if legacy or added:
                rebuild_history()
            connection.close()
        except Exception as e:
            logger.error(f"model refresh failed: {e}", exc_info=True)
        finally:
            _refresh_running = False

    threading.Thread(target=_job, daemon=True).start()


def get_accuracy_stats() -> dict:
    """คำนวณ accuracy ของ predictions ที่ผ่านมา (เฉพาะโมเดลปัจจุบัน)"""
    from lottery.models import Prediction
    from django.db.models import Q

    evaluated = Prediction.objects.filter(actual_result__isnull=False, model_used=MODEL_VERSION)
    total = evaluated.count()

    if total == 0:
        return {
            'total': 0,
            'correct_two': 0, 'correct_three': 0, 'correct_first': 0,
            'acc_two': 0, 'acc_three': 0, 'acc_first': 0,
        }

    correct_two   = evaluated.filter(Q(is_correct_two=True) | Q(is_correct_two_top=True)).count()
    correct_three = evaluated.filter(is_correct_three=True).count()
    correct_first = evaluated.filter(is_correct_first=True).count()

    return {
        'total':         total,
        'correct_two':   correct_two,
        'correct_three': correct_three,
        'correct_first': correct_first,
        'acc_two':   round(correct_two   / total * 100, 1),
        'acc_three': round(correct_three / total * 100, 1),
        'acc_first': round(correct_first / total * 100, 1),
    }


def get_statistical_analysis(limit=30) -> dict:
    """
    วิเคราะห์สถิติ:
    - Sum Window & Average Sum
    - Digital Root Frequency
    - Sample Size Features
    """
    history = _get_history(limit=limit)
    if not history:
        return {
            'avg_sum': 0, 'min_sum': 0, 'max_sum': 0,
            'digital_roots': {}, 'common_digital_root': '-', 'total_analyzed': 0,
        }

    sums = []
    digital_roots = Counter()

    for lr in history:
        digits = _extract_digits(lr)
        if digits:
            s = sum(digits)
            sums.append(s)
            dr = (s - 1) % 9 + 1 if s > 0 else 0
            digital_roots[dr] += 1

    avg_sum    = round(sum(sums) / len(sums), 1) if sums else 0
    common_dr  = digital_roots.most_common(1)[0][0] if digital_roots else '-'

    return {
        'avg_sum':            avg_sum,
        'min_sum':            min(sums) if sums else 0,
        'max_sum':            max(sums) if sums else 0,
        'digital_roots':      dict(digital_roots.most_common(3)),
        'common_digital_root': common_dr,
        'total_analyzed':     len(history),
    }
