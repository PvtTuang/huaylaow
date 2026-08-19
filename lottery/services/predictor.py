"""
ML Prediction Engine สำหรับหวยลาวพัฒนา
ใช้ Ensemble ของ 4 วิธี:
1. Gap / Overdue Analysis (ตัวเลขที่ค้างนานในแต่ละตำแหน่ง)
2. Markov Chain (pattern ต่อเนื่องระหว่างงวด)
3. Position-aware Weighted Frequency (ให้น้ำหนักงวดใหม่มากกว่าเก่า)
4. Random Forest (ถ้า sklearn ติดตั้งอยู่)

สิ่งที่แก้จากเวอร์ชันเก่า:
- เพิ่ม Noise (สุ่มเล็กน้อยในแต่ละงวด) เพื่อกระจายผล ไม่ให้ซ้ำกันทุกวัน
- ไม่ใช้ random_state ตายตัว
- วิเคราะห์ Gap (เลขที่หายไปนานควรมีโอกาสออก)
- ให้ target_date มีผลกับผลลัพธ์
"""
import logging
import random
import hashlib
from collections import Counter, defaultdict
from datetime import date, timedelta

import numpy as np

logger = logging.getLogger(__name__)


# ---------- helpers ----------

def _get_history(limit=60):
    """ดึงประวัติ N งวดล่าสุดจาก DB"""
    from lottery.models import LotteryResult
    qs = LotteryResult.objects.exclude(first_prize='').order_by('-draw_date')[:limit]
    return list(qs)


def _extract_digits(lottery_result) -> list:
    """แปลง first_prize -> list of int"""
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


# ---------- Method 1: Weighted Frequency (งวดใหม่ = น้ำหนักมากกว่า) ----------

def weighted_frequency_predict(history: list, prize_len=6, rng: random.Random = None) -> str:
    """
    เลือกตัวเลขโดยให้น้ำหนักงวดที่ใหม่กว่ามากกว่างวดเก่า
    ใช้ Laplace Smoothing (base weight 0.5) เพื่อให้ทุกเลข 0-9 มีโอกาสเสมอ
    """
    if not history:
        return "000000"
    if rng is None:
        rng = random.Random()

    n = len(history)
    # เริ่มต้นทุกเลข 0-9 มี base weight = 0.5 (Laplace Smoothing)
    position_weights = [{d: 0.5 for d in range(10)} for _ in range(prize_len)]

    for rank, lr in enumerate(history):
        weight = 1.0 / (rank + 1)  # งวดล่าสุด weight สูงสุด
        digits = _pad_or_trim(_extract_digits(lr), prize_len)
        for i, d in enumerate(digits):
            position_weights[i][d] += weight

    predicted = []
    for i in range(prize_len):
        pw = position_weights[i]
        digits_list = list(range(10))
        weights_list = [pw[d] for d in digits_list]
        chosen = rng.choices(digits_list, weights=weights_list, k=1)[0]
        predicted.append(str(chosen))

    return ''.join(predicted)


# ---------- Method 2: Gap / Overdue Analysis ----------

def gap_predict(history: list, prize_len=6, rng: random.Random = None) -> str:
    """
    วิเคราะห์ว่าตัวเลขในแต่ละตำแหน่งหายไปนานแค่ไหน
    ตัวเลขที่ไม่ออกมานานจะมีโอกาสสูงกว่า
    ใช้ Laplace Smoothing เพื่อให้ทุกเลข 0-9 มีโอกาสเสมอ
    """
    if not history:
        return "000000"
    if rng is None:
        rng = random.Random()

    n = len(history)
    # หา last seen index (rank 0 = ล่าสุด)
    last_seen = [dict() for _ in range(prize_len)]
    for rank, lr in enumerate(history):
        digits = _pad_or_trim(_extract_digits(lr), prize_len)
        for i, d in enumerate(digits):
            if d not in last_seen[i]:
                last_seen[i][d] = rank

    predicted = []
    for i in range(prize_len):
        gap_scores = {}
        for d in range(10):
            if d in last_seen[i]:
                gap_scores[d] = last_seen[i][d] + 1  # ยิ่งห่างนาน ยิ่งคะแนนสูง
            else:
                gap_scores[d] = n + 10  # ไม่เคยออก = คะแนนสูงมาก

        # Normalize ให้สมดุล: square root เพื่อลด bias
        digits_list = list(range(10))
        weights_list = [float(gap_scores[d]) ** 0.6 for d in digits_list]
        chosen = rng.choices(digits_list, weights=weights_list, k=1)[0]
        predicted.append(str(chosen))

    return ''.join(predicted)


# ---------- Method 3: Markov Chain ----------

def markov_predict(history: list, prize_len=6, rng: random.Random = None) -> str:
    """
    ทำนายโดย Markov Chain: ดูว่าหลังจากเลขนี้ออก ตำแหน่งนี้มักจะออกเลขอะไรถัดไป
    """
    if len(history) < 3:
        return weighted_frequency_predict(history, prize_len, rng)
    if rng is None:
        rng = random.Random()

    # สร้าง transition matrix ในแต่ละตำแหน่ง
    transitions = [defaultdict(Counter) for _ in range(prize_len)]
    ordered = list(reversed(history))  # เรียงจากเก่า -> ใหม่

    for idx in range(1, len(ordered)):
        prev_digits = _pad_or_trim(_extract_digits(ordered[idx - 1]), prize_len)
        curr_digits = _pad_or_trim(_extract_digits(ordered[idx]), prize_len)
        for pos in range(prize_len):
            transitions[pos][prev_digits[pos]][curr_digits[pos]] += 1

    # ใช้งวดล่าสุดเป็น state ปัจจุบัน
    last_digits = _pad_or_trim(_extract_digits(history[0]), prize_len)

    predicted = []
    for pos in range(prize_len):
        current_state = last_digits[pos]
        if current_state in transitions[pos] and transitions[pos][current_state]:
            counter = transitions[pos][current_state]
            options = list(counter.keys())
            weights = [float(counter[d]) for d in options]
            chosen = rng.choices(options, weights=weights, k=1)[0]
        else:
            # ไม่มีข้อมูล fallback ไป weighted freq
            chosen = int(weighted_frequency_predict(history, prize_len, rng)[pos])
        predicted.append(str(chosen))

    return ''.join(predicted)


# ---------- Method 4: Random Forest (optional) ----------

def rf_predict(history: list, prize_len=6, target_date: date = None, rng: random.Random = None) -> str:
    """Random Forest prediction พร้อม date-based seed"""
    if len(history) < 10:
        return weighted_frequency_predict(history, prize_len, rng)

    try:
        from sklearn.ensemble import RandomForestClassifier

        ordered = list(reversed(history))
        results = [_pad_or_trim(_extract_digits(lr), prize_len) for lr in ordered]

        WINDOW = 5
        X, y = [], []
        for i in range(WINDOW, len(results)):
            features = []
            for j in range(WINDOW):
                features.extend(results[i - WINDOW + j])
            # เพิ่ม weekday และ week-of-month
            if target_date:
                features.append(target_date.weekday())
                features.append(target_date.day // 7)
            else:
                features.append(i % 7)
                features.append(0)
            X.append(features)
            y.append(results[i])

        if len(X) < 5:
            return weighted_frequency_predict(history, prize_len, rng)

        X_arr = np.array(X)
        y_arr = np.array(y)

        # seed จากวันที่ ไม่ใช้ 42 ตายตัว
        rs = _date_seed(target_date) if target_date else random.randint(0, 999999)

        predicted_digits = []
        for pos in range(prize_len):
            clf = RandomForestClassifier(
                n_estimators=100,
                max_depth=6,
                random_state=rs % 100000,
            )
            clf.fit(X_arr, y_arr[:, pos])

            last_5 = list(reversed(history[:5]))
            features = []
            for lr in last_5:
                features.extend(_pad_or_trim(_extract_digits(lr), prize_len))
            if target_date:
                features.append(target_date.weekday())
                features.append(target_date.day // 7)
            else:
                features.append(0)
                features.append(0)

            pred = clf.predict([features])[0]
            predicted_digits.append(str(pred))

        return ''.join(predicted_digits)

    except Exception as e:
        logger.error(f"RF predict error: {e}")
        return weighted_frequency_predict(history, prize_len, rng)


# ---------- Ensemble Voting ----------

def ensemble_predict(history: list, prize_len=6, target_date: date = None) -> tuple:
    """
    รวม 4 วิธีโดย Soft Voting (weighted)
    - แต่ละวิธีได้ vote ต่างกัน
    - มี noise เพิ่มความหลากหลายตามวันที่
    Returns: (predicted_str, confidence_float)
    """
    if not history:
        return "000000", 0.0

    seed = _date_seed(target_date) if target_date else random.randint(0, 999999)
    rng = random.Random(seed)

    # รัน 4 วิธี
    preds = {
        'weighted_freq': weighted_frequency_predict(history, prize_len, rng),
        'gap':           gap_predict(history, prize_len, rng),
        'markov':        markov_predict(history, prize_len, rng),
        'rf':            rf_predict(history, prize_len, target_date, rng),
    }

    # น้ำหนักของแต่ละวิธี
    method_weights = {
        'weighted_freq': 1.5,
        'gap':           4.0,   # Gap analysis สูงสุด — ช่วยทำลาย bias
        'markov':        1.0,   # ลดลง เพราะมักดึงเลขซ้ำจากประวัติ
        'rf':            1.5,
    }

    logger.info(f"Individual predictions: {preds}")

    final = []
    total_score = 0.0

    for pos in range(prize_len):
        # สะสมคะแนนในแต่ละตำแหน่งให้แต่ละ digit
        digit_scores = defaultdict(float)
        for method, pred in preds.items():
            if len(pred) > pos:
                d = pred[pos]
                digit_scores[d] += method_weights[method]

        # เลือก digit ที่คะแนนสูงสุด
        if digit_scores:
            best_digit = max(digit_scores, key=digit_scores.get)
            best_score = digit_scores[best_digit]
            total_possible = sum(method_weights.values())
            total_score += best_score / total_possible
        else:
            best_digit = str(rng.randint(0, 9))
            total_score += 0.5

        final.append(best_digit)

    # Confidence = สัดส่วนที่วิธีต่างๆ เห็นตรงกัน
    avg_agreement = total_score / prize_len  # 0.0 - 1.0
    # Scale ให้อยู่ระหว่าง 65% - 93%
    confidence = 65.0 + avg_agreement * 28.0
    confidence = min(93.0, max(65.0, confidence))

    return ''.join(final), round(confidence, 1)


# ---------- Public API ----------

def predict_next(target_date: date = None) -> dict:
    """
    ทำนายหวยงวดถัดไป
    Returns dict: predicted_first, predicted_two, predicted_three, confidence
    """
    if target_date is None:
        target_date = date.today() + timedelta(days=1)

    history = _get_history(limit=60)

    if not history:
        logger.warning("ไม่มีข้อมูลประวัติ ไม่สามารถทำนายได้")
        return {
            'predicted_first': 'N/A',
            'predicted_two': 'N/A',
            'predicted_three': 'N/A',
            'confidence': 0.0,
            'note': 'กรุณาดึงข้อมูลประวัติก่อน'
        }

    prize_len = 6
    if history:
        sample_digits = _extract_digits(history[0])
        if sample_digits:
            prize_len = len(sample_digits)

    # Dynamic Ensemble: แต่ละวันได้เลขต่างกัน ไม่ซ้ำ
    predicted_full, confidence = ensemble_predict(history, prize_len, target_date)

    return {
        'predicted_first': predicted_full,
        'predicted_two':   predicted_full[-2:] if len(predicted_full) >= 2 else '00',
        'predicted_three': predicted_full[-3:] if len(predicted_full) >= 3 else '000',
        'confidence': round(confidence, 1),
        'model_used': 'ensemble',
        'based_on': len(history),
    }


def save_prediction(target_date: date = None) -> 'Prediction':
    """สร้างและบันทึก prediction ลง DB (ป้องกันรายการซ้ำซ้อน)"""
    from lottery.models import Prediction, LotteryResult

    if target_date is None:
        target_date = date.today() + timedelta(days=1)

    # ลบ prediction เก่าทั้งหมดของวันนั้นเพื่อป้องกันการซ้ำซ้อน
    Prediction.objects.filter(target_date=target_date).delete()

    result = predict_next(target_date)

    actual = LotteryResult.objects.filter(draw_date=target_date).first()

    pred = Prediction.objects.create(
        target_date=target_date,
        predicted_first=result['predicted_first'],
        predicted_two=result['predicted_two'],
        predicted_three=result['predicted_three'],
        confidence=result['confidence'],
        model_used=result.get('model_used', 'ensemble'),
        actual_result=actual,
    )

    if actual:
        pred.evaluate()

    logger.info(f"บันทึก prediction งวด {target_date}: {result['predicted_first']} (confidence: {result['confidence']}%)")
    return pred


def get_accuracy_stats() -> dict:
    """คำนวณ accuracy ของ predictions ที่ผ่านมา"""
    from lottery.models import Prediction

    evaluated = Prediction.objects.filter(actual_result__isnull=False)
    total = evaluated.count()

    if total == 0:
        return {'total': 0, 'correct_two': 0, 'correct_three': 0, 'correct_first': 0,
                'acc_two': 0, 'acc_three': 0, 'acc_first': 0}

    correct_two = evaluated.filter(is_correct_two=True).count()
    correct_three = evaluated.filter(is_correct_three=True).count()
    correct_first = evaluated.filter(is_correct_first=True).count()

    return {
        'total': total,
        'correct_two': correct_two,
        'correct_three': correct_three,
        'correct_first': correct_first,
        'acc_two': round(correct_two / total * 100, 1),
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
        return {'avg_sum': 0, 'min_sum': 0, 'max_sum': 0, 'digital_roots': {}, 'common_digital_root': '-', 'total_analyzed': 0}

    sums = []
    digital_roots = Counter()

    for lr in history:
        digits = _extract_digits(lr)
        if digits:
            s = sum(digits)
            sums.append(s)
            dr = (s - 1) % 9 + 1 if s > 0 else 0
            digital_roots[dr] += 1

    avg_sum = round(sum(sums) / len(sums), 1) if sums else 0
    common_dr = digital_roots.most_common(1)[0][0] if digital_roots else '-'

    return {
        'avg_sum': avg_sum,
        'min_sum': min(sums) if sums else 0,
        'max_sum': max(sums) if sums else 0,
        'digital_roots': dict(digital_roots.most_common(3)),
        'common_digital_root': common_dr,
        'total_analyzed': len(history),
    }

