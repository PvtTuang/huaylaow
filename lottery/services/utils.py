import datetime
from django.utils import timezone

def get_next_draw_date(now_dt: datetime.datetime = None) -> datetime.date:
    """
    คำนวณวันออกรางวัลถัดไป (จันทร์, อังคาร, พุธ, พฤหัสบดี, ศุกร์)
    เวลาออกรางวัลคือ 20:30 น. ตามเวลาประเทศไทย (Asia/Bangkok)
    """
    if now_dt is None:
        now_dt = timezone.localtime()
    
    current_date = now_dt.date()
    
    # เช็คว่าวันนี้เป็นวันออกรางวัลหรือไม่ (0=จันทร์, 1=อังคาร, 2=พุธ, 3=พฤหัสบดี, 4=ศุกร์)
    if current_date.weekday() < 5:
        # ถ้าวันนี้เป็นวันออกรางวัล และเวลาปัจจุบันยังไม่ถึง 20:30 น.
        if now_dt.time() < datetime.time(20, 30):
            return current_date
            
    # หาวันออกรางวัลถัดไป (จันทร์-ศุกร์)
    days_to_add = 1
    next_date = current_date + datetime.timedelta(days=days_to_add)
    while next_date.weekday() >= 5: # 5=เสาร์, 6=อาทิตย์
        days_to_add += 1
        next_date = current_date + datetime.timedelta(days=days_to_add)
    return next_date
