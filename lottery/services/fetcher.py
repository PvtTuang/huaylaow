"""
MThai Fetcher Service - ดึงข้อมูลหวยลาวพัฒนาจาก MThai เพื่อความแม่นยำ 100%
"""
import re
import json
import logging
import requests
from bs4 import BeautifulSoup
from datetime import date, timedelta
from django.conf import settings

logger = logging.getLogger(__name__)

MONTH_TH = {
    'มกราคม': 1, 'กุมภาพันธ์': 2, 'มีนาคม': 3, 'เมษายน': 4,
    'พฤษภาคม': 5, 'มิถุนายน': 6, 'กรกฎาคม': 7, 'สิงหาคม': 8,
    'กันยายน': 9, 'ตุลาคม': 10, 'พฤศจิกายน': 11, 'ธันวาคม': 12
}

def fetch_mthai_results(pages=1):
    """
    ดึงผลหวยจาก MThai
    Returns dict: {date_obj: '123456'}
    """
    results = {}
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    
    for page in range(1, pages + 1):
        url = 'https://lotto.mthai.com/lao' if page == 1 else f'https://lotto.mthai.com/lao/page/{page}'
        try:
            resp = requests.get(url, headers=headers, timeout=15)
            resp.raise_for_status()
            
            # Use regex directly on html text for simplicity
            matches = re.findall(r'งวดวัน.*?(\d{1,2})\s*([ก-๙]+)\s*(25\d{2})\s*เลข 6 ตัว\s*(\d{6})', resp.text)
            
            for d, m_th, y_th, digits in matches:
                try:
                    month = MONTH_TH.get(m_th)
                    year = int(y_th) - 543
                    day = int(d)
                    if month:
                        draw_date = date(year, month, day)
                        results[draw_date] = digits
                except Exception as e:
                    logger.warning(f"Error parsing date {d} {m_th} {y_th}: {e}")
                    
        except Exception as e:
            logger.error(f"MThai fetch error on page {page}: {e}")
            
    return results


def parse_lottery_numbers(six_digits: str) -> dict:
    return {
        'first_prize': six_digits,
        'two_digit': six_digits[-2:],
        'three_digit': six_digits[-3:],
        'raw_data': f'Scraped from MThai: {six_digits}'
    }


def fetch_and_save(target_date: date = None):
    """ดึงข้อมูลและบันทึกลง DB"""
    from lottery.models import LotteryResult, FetchLog
    
    if target_date is None:
        target_date = date.today()
        
    existing = LotteryResult.objects.filter(draw_date=target_date).first()
    if existing and existing.first_prize:
        return existing, "มีข้อมูลแล้ว ไม่ต้องดึงใหม่"
        
    # ดึงผล 1 หน้าจาก MThai
    mthai_data = fetch_mthai_results(pages=1)
    
    six_digits = mthai_data.get(target_date)
    
    if not six_digits:
        # ถ้าไม่มีของวันนี้ อาจจะยังไม่ออก หรือหวยงด
        # แต่เพื่อความแน่ใจ ให้บันทึกข้อมูลอื่นที่ได้มาด้วย
        saved_count = 0
        for d, digits in mthai_data.items():
            if not LotteryResult.objects.filter(draw_date=d).exists():
                parsed = parse_lottery_numbers(digits)
                obj = LotteryResult.objects.create(
                    draw_date=d,
                    first_prize=parsed['first_prize'],
                    two_digit=parsed['two_digit'],
                    three_digit=parsed['three_digit'],
                    raw_data=parsed['raw_data']
                )
                _evaluate_predictions(obj)
                saved_count += 1
                
        FetchLog.objects.create(
            status='fail' if not six_digits else 'success',
            message=f"ไม่พบผลหวยของวันที่ {target_date} แต่บันทึกข้อมูลงวดอื่นได้ {saved_count} งวด",
            records_saved=saved_count
        )
        return None, f"ไม่พบข้อมูลหวยงวด {target_date} (แต่มีอัปเดตข้อมูลงวดอื่น)"
        
    parsed = parse_lottery_numbers(six_digits)
    
    obj, created = LotteryResult.objects.update_or_create(
        draw_date=target_date,
        defaults={
            'first_prize': parsed['first_prize'],
            'two_digit': parsed['two_digit'],
            'three_digit': parsed['three_digit'],
            'raw_data': parsed['raw_data'],
        }
    )
    
    FetchLog.objects.create(
        status='success',
        message=f"บันทึกผลหวยงวด {target_date}: {six_digits}",
        records_saved=1
    )
    
    _evaluate_predictions(obj)
    
    action = "สร้างใหม่" if created else "อัปเดต"
    return obj, f"{action}ข้อมูลงวด {target_date} สำเร็จ"


def _evaluate_predictions(lottery_result):
    from lottery.models import Prediction
    preds = Prediction.objects.filter(
        target_date=lottery_result.draw_date,
        actual_result__isnull=True
    )
    for pred in preds:
        pred.actual_result = lottery_result
        pred.evaluate()


def fetch_history(days=30):
    """ดึงข้อมูลประวัติจาก MThai ย้อนหลัง"""
    from lottery.models import LotteryResult
    
    mthai_data = fetch_mthai_results(pages=3)  # ดึง 3 หน้า จะได้ราวๆ 30 งวด
    results = []
    
    for d, digits in mthai_data.items():
        existing = LotteryResult.objects.filter(draw_date=d).first()
        if not existing:
            parsed = parse_lottery_numbers(digits)
            obj = LotteryResult.objects.create(
                draw_date=d,
                first_prize=parsed['first_prize'],
                two_digit=parsed['two_digit'],
                three_digit=parsed['three_digit'],
                raw_data=parsed['raw_data']
            )
            _evaluate_predictions(obj)
            results.append(obj)
            
    logger.info(f"History fetch done: {len(results)} saved")
    return results
