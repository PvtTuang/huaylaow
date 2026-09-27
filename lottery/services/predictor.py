"""
ML Prediction Engine สำหรับหวยลาวพัฒนา (v3 - Lao-Optimized)
ใช้ Ensemble ของ 5 วิธีที่ปรับแต่งสำหรับหวยลาวโดยเฉพาะ:

1. Exponential Decay Frequency  — งวดล่าสุดมีน้ำหนักมากแบบ exp แทน 1/n
2. Multi-Window Hot/Cold         — วิเคราะห์ 5, 10, 20 งวดแยก แล้วรวมคะแนน
3. Gap + Frequency Combined      — เลขที่หายนาน AND ไม่ค่อยออกรวมกัน
4. Hot-Streak Detection          — เลขที่ออกติดต่อกัน 2+ งวด ได้ bonus
5. 2nd-Order Markov Chain        — ดู 2 งวดก่อนหน้าเพื่อ pattern ที่ละเอียดขึ้น

Adaptive Ensemble:
  — backtest ย้อนหลัง 10 งวดเพื่อหาว่า method ไหนแม่นกว่า แล้วชั่งน้ำหนักตาม
"""
import logging
import random
import hashlib
import math
from collections import Counter, defaultdict
from datetime import date, timedelta

import numpy as np

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────

def _get_history(limit=80):
    """ดึงประวัติ N งวดล่าสุดจาก DB (เพิ่มเป็น 80 งวดสำหรับ Markov 2nd-order)"""
    from lottery.models import LotteryResult
    qs = LotteryResult.objects.exclude(first_prize='').order_by('-draw_date')[:limit]
    return list(qs)


def _extract_digits(lottery_result) -> list:
    """แปลง first_prize → list of int"""
    fp = lottery_result.first_prize
    return [int(c) for c in fp if c.isdigit()]


def _pad_or_trim(digits, length=6, fill=0) -> list:
    """ทำให้ digit list มีความยาวคงที่"""
    d = list(digits[:length])
    while len(d) < length:
        d.append(fill)
    return d


def _date_seed(target_date: date) -> int:
    """สร้าง seed จากวันที่ เพื่อให้แต่ละวันได้เลขต่างกัน แต่คงที่ในวันเดียวกัน"""
    s = str(target_date)
    return int(hashlib.md5(s.encode()).hexdigest(), 16) % (2**31)


def _make_scores(position_weights: list, prize_len: int, rng: random.Random) -> str:
    """เลือก digit ที่ดีที่สุดจาก position_weights (dict ของ score ต่อ digit)"""
    predicted = []
    for i in range(prize_len):
        pw = position_weights[i]
        digits_list = list(range(10))
        weights_list = [max(pw.get(d, 0.01), 0.01) for d in digits_list]
        chosen = rng.choices(digits_list, weights=weights_list, k=1)[0]
        predicted.append(str(chosen))
    return ''.join(predicted)


# ──────────────────────────────────────────────
#  Method 1: Exponential Decay Frequency
# ──────────────────────────────────────────────

def exp_decay_predict(history: list, prize_len=6, rng: random.Random = None) -> str:
    """
    ให้น้ำหนักงวดล่าสุดแบบ exponential: w = exp(-λ * rank)
    λ = 0.15 → งวดก่อนนาน 10 งวดมีน้ำหนักเพียง ~22% ของงวดล่าสุด
    ดีกว่า 1/(rank+1) เพราะ drop ชันกว่า = งวดใหม่สำคัญกว่ามาก
    """
    if not history:
        return "0" * prize_len
    if rng is None:
        rng = random.Random()

    LAMBDA = 0.15
    # Laplace smoothing เล็กน้อย
    position_weights = [{d: 0.3 for d in range(10)} for _ in range(prize_len)]

    for rank, lr in enumerate(history):
        w = math.exp(-LAMBDA * rank)
        digits = _pad_or_trim(_extract_digits(lr), prize_len)
        for i, d in enumerate(digits):
            position_weights[i][d] += w

    return _make_scores(position_weights, prize_len, rng)


# ──────────────────────────────────────────────
#  Method 2: Multi-Window Hot/Cold
# ──────────────────────────────────────────────

def multi_window_predict(history: list, prize_len=6, rng: random.Random = None) -> str:
    """
    วิเคราะห์ใน 3 กรอบเวลา: 5, 10, 20 งวด
    - Hot window (5 งวด): เลขที่ออกบ่อยในช่วงสั้น → โอกาสออกต่อ (momentum)
    - Mid window (10 งวด): ภาพรวมระยะกลาง
    - Long window (20 งวด): ฐาน frequency ระยะยาว

    รวมคะแนนด้วย weight: hot=0.5, mid=0.3, long=0.2
    """
    if not history:
        return "0" * prize_len
    if rng is None:
        rng = random.Random()

    windows = [
        (min(5, len(history)),  0.50),   # hot window
        (min(10, len(history)), 0.30),   # mid window
        (min(20, len(history)), 0.20),   # long window
    ]

    position_weights = [{d: 0.1 for d in range(10)} for _ in range(prize_len)]

    for win_size, win_weight in windows:
        sub_history = history[:win_size]
        # นับ frequency ใน window นี้
        pos_count = [{d: 0 for d in range(10)} for _ in range(prize_len)]
        for lr in sub_history:
            digits = _pad_or_trim(_extract_digits(lr), prize_len)
            for i, d in enumerate(digits):
                pos_count[i][d] += 1

        for i in range(prize_len):
            total = max(sum(pos_count[i].values()), 1)
            for d in range(10):
                freq = pos_count[i][d] / total
                position_weights[i][d] += win_weight * freq

    return _make_scores(position_weights, prize_len, rng)


# ──────────────────────────────────────────────
#  Method 3: Gap + Frequency Combined
# ──────────────────────────────────────────────

def gap_freq_predict(history: list, prize_len=6, rng: random.Random = None) -> str:
    """
    รวม 2 สัญญาณ:
      gap_score  = จำนวนงวดที่ไม่ออก (ยิ่งนาน ยิ่งสูง)
      freq_score = ความถี่ใน 20 งวดล่าสุด (normalized)

    combined = 0.6 * gap_norm + 0.4 * (1 - freq_norm)
    → ชอบเลขที่ "หายนาน AND ออกน้อย" (ค้างยาว + cold)
    ใช้ sqrt เพื่อลด extreme bias
    """
    if not history:
        return "0" * prize_len
    if rng is None:
        rng = random.Random()

    n = len(history)
    WINDOW = min(20, n)
    sub = history[:WINDOW]

    # คำนวณ gap (งวดล่าสุดที่เห็นแต่ละ digit ในแต่ละ position)
    last_seen = [{d: n + 10 for d in range(10)} for _ in range(prize_len)]
    for rank, lr in enumerate(history):
        digits = _pad_or_trim(_extract_digits(lr), prize_len)
        for i, d in enumerate(digits):
            if last_seen[i][d] == n + 10:   # ยังไม่เคยเห็น
                last_seen[i][d] = rank

    # คำนวณ frequency ใน window
    pos_count = [{d: 0 for d in range(10)} for _ in range(prize_len)]
    for lr in sub:
        digits = _pad_or_trim(_extract_digits(lr), prize_len)
        for i, d in enumerate(digits):
            pos_count[i][d] += 1

    position_weights = []
    for i in range(prize_len):
        gaps = [last_seen[i][d] for d in range(10)]
        freqs = [pos_count[i][d] for d in range(10)]

        max_gap = max(gaps) if max(gaps) > 0 else 1
        max_freq = max(freqs) if max(freqs) > 0 else 1

        pw = {}
        for d in range(10):
            gap_norm = gaps[d] / max_gap
            freq_norm = freqs[d] / max_freq
            # combined: ชอบเลขที่ gap นาน + ออกน้อย
            combined = 0.6 * gap_norm + 0.4 * (1.0 - freq_norm)
            pw[d] = math.sqrt(max(combined, 0.01))   # sqrt ลด extreme
        position_weights.append(pw)

    return _make_scores(position_weights, prize_len, rng)


# ──────────────────────────────────────────────
#  Method 4: Hot-Streak Detection
# ──────────────────────────────────────────────

def hot_streak_predict(history: list, prize_len=6, rng: random.Random = None) -> str:
    """
    ตรวจหาเลขที่ออกซ้ำติดต่อกัน (streak) ใน 3 งวดล่าสุด
    streak=3: bonus สูงมาก
    streak=2: bonus กลาง
    streak=1: ออกงวดล่าสุดแค่ครั้งเดียว = bonus น้อย
    ไม่ออกเลย = baseline

    ใช้ร่วมกับ base frequency ระยะสั้น (5 งวด) เพื่อไม่ให้ overshoot
    """
    if not history:
        return "0" * prize_len
    if rng is None:
        rng = random.Random()

    STREAK_LOOK = min(5, len(history))
    sub = history[:STREAK_LOOK]

    # base: frequency ใน 10 งวดล่าสุด
    base_window = min(10, len(history))
    position_weights = [{d: 0.2 for d in range(10)} for _ in range(prize_len)]
    for lr in history[:base_window]:
        digits = _pad_or_trim(_extract_digits(lr), prize_len)
        for i, d in enumerate(digits):
            position_weights[i][d] += 0.1

    # streak bonus
    STREAK_BONUS = {3: 3.5, 2: 2.0, 1: 1.0}
    for i in range(prize_len):
        digit_streak = {}
        for d in range(10):
            streak = 0
            for lr in sub:
                digits = _pad_or_trim(_extract_digits(lr), prize_len)
                if digits[i] == d:
                    streak += 1
                else:
                    break  # streak ต้องติดต่อกัน
            digit_streak[d] = streak

        for d in range(10):
            s = digit_streak[d]
            bonus = STREAK_BONUS.get(s, 0.0)
            if s >= 4:
                bonus = 4.5   # rare แต่ให้ bonus สูงสุด
            position_weights[i][d] += bonus

    return _make_scores(position_weights, prize_len, rng)


# ──────────────────────────────────────────────
#  Method 5: 2nd-Order Markov Chain
# ──────────────────────────────────────────────

def markov2_predict(history: list, prize_len=6, rng: random.Random = None) -> str:
    """
    Markov Chain อันดับ 2: ดู 2 งวดก่อนหน้า → ทำนายงวดถัดไป
    pattern มีความละเอียดกว่า 1st-order มาก

    Fallback: ถ้า state ไม่เคยเห็น → ใช้ 1st-order
              ถ้า 1st-order ก็ไม่มี → ใช้ exp_decay
    """
    if len(history) < 5:
        return exp_decay_predict(history, prize_len, rng)
    if rng is None:
        rng = random.Random()

    ordered = list(reversed(history))   # เรียงจากเก่า → ใหม่
    results = [_pad_or_trim(_extract_digits(lr), prize_len) for lr in ordered]

    # สร้าง transition table สำหรับ 2nd-order
    trans2 = [defaultdict(Counter) for _ in range(prize_len)]
    trans1 = [defaultdict(Counter) for _ in range(prize_len)]

    for idx in range(2, len(results)):
        prev2 = results[idx - 2]
        prev1 = results[idx - 1]
        curr  = results[idx]
        for pos in range(prize_len):
            state2 = (prev2[pos], prev1[pos])
            trans2[pos][state2][curr[pos]] += 1
            trans1[pos][prev1[pos]][curr[pos]] += 1

    # state ปัจจุบัน = 2 งวดล่าสุด (history[0]=ล่าสุด, history[1]=ก่อนนั้น)
    last1 = _pad_or_trim(_extract_digits(history[0]), prize_len)
    last2 = _pad_or_trim(_extract_digits(history[1]), prize_len) if len(history) >= 2 else last1

    predicted = []
    for pos in range(prize_len):
        state2 = (last2[pos], last1[pos])
        state1 = last1[pos]

        if state2 in trans2[pos] and trans2[pos][state2]:
            # มี 2nd-order data
            counter = trans2[pos][state2]
            options = list(counter.keys())
            weights = [float(counter[d]) for d in options]
            chosen = rng.choices(options, weights=weights, k=1)[0]
        elif state1 in trans1[pos] and trans1[pos][state1]:
            # fallback 1st-order
            counter = trans1[pos][state1]
            options = list(counter.keys())
            weights = [float(counter[d]) for d in options]
            chosen = rng.choices(options, weights=weights, k=1)[0]
        else:
            # fallback exp_decay
            chosen = int(exp_decay_predict(history, prize_len, rng)[pos])
        predicted.append(str(chosen))

    return ''.join(predicted)


# ──────────────────────────────────────────────
#  Adaptive Ensemble Weights (Mini-Backtest)
# ──────────────────────────────────────────────

def _backtest_weights(history: list, prize_len=6, backtest_n=10) -> dict:
    """
    Backtest ย้อนหลัง `backtest_n` งวด:
    สำหรับแต่ละงวด i ใน backtest_n งวดล่าสุด:
      - ใช้ history[i+1:] เป็น training
      - ทำนาย history[i]
      - นับ digit ที่ถูกต้องต่อ method

    คืน dict: {method_name: normalized_weight}
    ถ้า backtest_n น้อยกว่า data ที่มี → ใช้ default weights
    """
    DEFAULT_WEIGHTS = {
        'exp_decay':    1.5,
        'multi_window': 1.5,
        'gap_freq':     1.5,
        'hot_streak':   1.0,
        'markov2':      1.5,
    }

    min_needed = backtest_n + 5   # ต้องการ data อีก >= 5 งวดสำหรับ training
    if len(history) < min_needed:
        logger.debug("ข้อมูลน้อยเกินไป ใช้ default weights")
        return DEFAULT_WEIGHTS

    method_fns = {
        'exp_decay':    exp_decay_predict,
        'multi_window': multi_window_predict,
        'gap_freq':     gap_freq_predict,
        'hot_streak':   hot_streak_predict,
        'markov2':      markov2_predict,
    }

    rng_bt = random.Random(42)   # fixed seed สำหรับ backtest เพื่อความสม่ำเสมอ
    scores = {m: 0 for m in method_fns}

    for i in range(backtest_n):
        actual = _pad_or_trim(_extract_digits(history[i]), prize_len)
        train = history[i + 1:]   # ประวัติที่ "รู้" ณ ตอนนั้น

        for method, fn in method_fns.items():
            rng_bt.seed(42 + i)
            try:
                pred_str = fn(train, prize_len, rng_bt)
                pred = [int(c) for c in pred_str]
                # นับ digit ที่ตรงในแต่ละ position
                hits = sum(1 for p, a in zip(pred, actual) if p == a)
                scores[method] += hits
            except Exception:
                pass   # ถ้า error ใน backtest → ได้ 0

    # Normalize: min weight = 0.5 ป้องกัน method ที่แย่ถูกตัดทิ้งหมด
    total = max(sum(scores.values()), 1)
    weights = {}
    for m, s in scores.items():
        raw = (s / total) * len(method_fns)   # scale ให้ mean ≈ 1.0
        weights[m] = max(raw, 0.5)

    logger.info(f"Backtest weights: {weights}")
    return weights


# ──────────────────────────────────────────────
#  Ensemble Voting (Soft Vote + Adaptive Weights)
# ──────────────────────────────────────────────

def ensemble_predict(history: list, prize_len=6, target_date: date = None) -> dict:
    """
    รวม 5 วิธีโดย Soft Voting พร้อม Adaptive Weights จาก Backtest

    Returns dict:
      predicted_first, two_digit_pairs, three_digit_sets,
      confidence, key_digit, secondary_digit, vote_breakdown
    """
    EMPTY = {
        'predicted_first': "0" * prize_len,
        'two_digit_pairs': ["00"] * 4,
        'three_digit_sets': ["000"] * 3,
        'confidence': 0.0,
        'key_digit': "0",
        'secondary_digit': "0",
        'vote_breakdown': [],
    }
    if not history:
        return EMPTY

    seed = _date_seed(target_date) if target_date else random.randint(0, 999999)
    rng = random.Random(seed)

    # ── Adaptive weights จาก backtest ──
    method_weights = _backtest_weights(history, prize_len, backtest_n=10)

    # ── รันทุก method ──
    preds = {
        'exp_decay':    exp_decay_predict(history, prize_len, rng),
        'multi_window': multi_window_predict(history, prize_len, rng),
        'gap_freq':     gap_freq_predict(history, prize_len, rng),
        'hot_streak':   hot_streak_predict(history, prize_len, rng),
        'markov2':      markov2_predict(history, prize_len, rng),
    }
    logger.info(f"Individual predictions: {preds}")
    logger.info(f"Method weights: {method_weights}")

    # ── Soft Vote per position ──
    final = []
    total_score = 0.0
    vote_breakdown = []
    overall_scores = defaultdict(float)
    top_digits_per_pos = []
    raw_scores_per_pos = []   # เก็บ raw score ทุก digit ทุก position สำหรับ joint prob

    for pos in range(prize_len):
        digit_scores = defaultdict(float)
        for method, pred in preds.items():
            if len(pred) > pos:
                d = pred[pos]
                digit_scores[d] += method_weights.get(method, 1.0)

        # normalize เป็น probability ต่อ position
        total_pos_score = sum(digit_scores.values()) or 1.0
        prob_this_pos = {}
        sorted_pos = []
        for d in range(10):
            d_str = str(d)
            score = digit_scores.get(d_str, 0.0)
            prob = score / total_pos_score
            prob_this_pos[d_str] = prob
            pct = round(prob * 100, 1)
            sorted_pos.append((d_str, pct))
            overall_scores[d_str] += score
        sorted_pos.sort(key=lambda x: x[1], reverse=True)
        vote_breakdown.append(sorted_pos)
        top_digits_per_pos.append([item[0] for item in sorted_pos])
        raw_scores_per_pos.append(prob_this_pos)

        # เลือก digit ชนะ
        if digit_scores:
            best_digit = max(digit_scores, key=digit_scores.get)
            best_score = digit_scores[best_digit]
            total_possible = sum(method_weights.values())
            total_score += best_score / total_possible
        else:
            best_digit = str(rng.randint(0, 9))
            total_score += 0.5
        final.append(best_digit)

    # ── Confidence ──
    avg_agreement = total_score / prize_len
    confidence = 60.0 + avg_agreement * 30.0
    confidence = min(90.0, max(60.0, confidence))

    # ── Key / Secondary digit ──
    sorted_overall = sorted(overall_scores.items(), key=lambda x: x[1], reverse=True)
    key_digit       = sorted_overall[0][0] if sorted_overall else "0"
    secondary_digit = sorted_overall[1][0] if len(sorted_overall) > 1 else "1"

    # ── เลขท้าย 2 ตัว: joint probability ranking ──
    # score(XY) = prob(X ที่ position สิบ) × prob(Y ที่ position หน่วย)
    # เลือก top-3 จาก 100 คู่ที่ joint score สูงสุด
    if prize_len >= 2:
        p_tens  = raw_scores_per_pos[prize_len - 2]
        p_units = raw_scores_per_pos[prize_len - 1]
        joint2 = []
        for t in range(10):
            for u in range(10):
                score = p_tens.get(str(t), 0.0) * p_units.get(str(u), 0.0)
                joint2.append((f"{t}{u}", score))
        joint2.sort(key=lambda x: x[1], reverse=True)
        two_digit_pairs = [pair for pair, _ in joint2[:3]]
    else:
        two_digit_pairs = ["00", "01", "10"]

    # ── เลขท้าย 3 ตัว: joint probability ranking ──
    # score(XYZ) = prob(X ที่ร้อย) × prob(Y ที่สิบ) × prob(Z ที่หน่วย)
    if prize_len >= 3:
        p_hunds = raw_scores_per_pos[prize_len - 3]
        p_tens  = raw_scores_per_pos[prize_len - 2]
        p_units = raw_scores_per_pos[prize_len - 1]
        joint3 = []
        for h in range(10):
            for t in range(10):
                for u in range(10):
                    score = (p_hunds.get(str(h), 0.0)
                             * p_tens.get(str(t), 0.0)
                             * p_units.get(str(u), 0.0))
                    joint3.append((f"{h}{t}{u}", score))
        joint3.sort(key=lambda x: x[1], reverse=True)
        three_digit_sets = [combo for combo, _ in joint3[:3]]
    else:
        three_digit_sets = ["000", "001", "010"]

    return {
        'predicted_first':  ''.join(final),
        'two_digit_pairs':  two_digit_pairs,
        'three_digit_sets': three_digit_sets,
        'confidence':       round(confidence, 1),
        'key_digit':        key_digit,
        'secondary_digit':  secondary_digit,
        'vote_breakdown':   vote_breakdown,
    }



# ──────────────────────────────────────────────
#  Public API
# ──────────────────────────────────────────────

def predict_next(target_date: date = None) -> dict:
    """
    ทำนายหวยงวดถัดไป
    Returns dict: predicted_first, predicted_two, predicted_three, confidence, ...
    """
    if target_date is None:
        from lottery.services.utils import get_next_draw_date
        target_date = get_next_draw_date()

    history = _get_history(limit=80)

    if not history:
        logger.warning("ไม่มีข้อมูลประวัติ ไม่สามารถทำนายได้")
        return {
            'predicted_first': 'N/A',
            'predicted_two':   'N/A',
            'predicted_three': 'N/A',
            'confidence': 0.0,
            'note': 'กรุณาดึงข้อมูลประวัติก่อน',
        }

    prize_len = 6
    if history:
        sample_digits = _extract_digits(history[0])
        if sample_digits:
            prize_len = len(sample_digits)

    res = ensemble_predict(history, prize_len, target_date)

    return {
        'predicted_first':  res['predicted_first'],
        'predicted_two':    ', '.join(res['two_digit_pairs']),
        'predicted_three':  ', '.join(res['three_digit_sets']),
        'confidence':       res['confidence'],
        'model_used':       'ensemble_v3',
        'based_on':         len(history),
        'key_digit':        res['key_digit'],
        'secondary_digit':  res['secondary_digit'],
        'vote_breakdown':   res['vote_breakdown'],
    }


def save_prediction(target_date: date = None) -> 'Prediction':
    """สร้างและบันทึก prediction ลง DB (ป้องกันรายการซ้ำซ้อน)"""
    from lottery.models import Prediction, LotteryResult

    if target_date is None:
        from lottery.services.utils import get_next_draw_date
        target_date = get_next_draw_date()

    # ลบ prediction เก่าทั้งหมดของวันนั้น
    Prediction.objects.filter(target_date=target_date).delete()

    result = predict_next(target_date)

    actual = LotteryResult.objects.filter(draw_date=target_date).first()

    pred = Prediction.objects.create(
        target_date=target_date,
        predicted_first=result['predicted_first'],
        predicted_two=result['predicted_two'],
        predicted_three=result['predicted_three'],
        confidence=result['confidence'],
        model_used=result.get('model_used', 'ensemble_v3'),
        key_digit=result.get('key_digit', 'N/A'),
        secondary_digit=result.get('secondary_digit', 'N/A'),
        vote_breakdown=result.get('vote_breakdown', None),
        actual_result=actual,
    )

    if actual:
        pred.evaluate()

    logger.info(
        f"บันทึก prediction งวด {target_date}: "
        f"{result['predicted_first']} (confidence: {result['confidence']}%)"
    )
    return pred


def get_accuracy_stats() -> dict:
    """คำนวณ accuracy ของ predictions ที่ผ่านมา"""
    from lottery.models import Prediction

    evaluated = Prediction.objects.filter(actual_result__isnull=False)
    total = evaluated.count()

    if total == 0:
        return {
            'total': 0,
            'correct_two': 0, 'correct_three': 0, 'correct_first': 0,
            'acc_two': 0, 'acc_three': 0, 'acc_first': 0,
        }

    from django.db.models import Q
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
    วิเคราะห์สถิติทางวิทยาศาสตร์:
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
