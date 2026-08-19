"""
APScheduler - ดึงข้อมูลหวยอัตโนมัติทุกวัน
"""
import logging
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from django.conf import settings

logger = logging.getLogger(__name__)
_scheduler = None


def fetch_daily():
    """Job: ดึงข้อมูลหวยวันนี้และทำนายงวดถัดไป"""
    try:
        from django.utils import timezone
        from lottery.services.fetcher import fetch_and_save
        from lottery.services.predictor import save_prediction
        from lottery.services.utils import get_next_draw_date

        today = timezone.localdate()
        obj, msg = fetch_and_save(today)
        logger.info(f"[Scheduler] fetch_daily: {msg}")

        if obj:
            next_date = get_next_draw_date()
            pred = save_prediction(next_date)
            logger.info(f"[Scheduler] prediction saved for {next_date}: {pred.predicted_first}")
    except Exception as e:
        logger.error(f"[Scheduler] error: {e}")


def start():
    """เริ่ม scheduler (เรียกจาก apps.py)"""
    global _scheduler
    if _scheduler and _scheduler.running:
        return

    _scheduler = BackgroundScheduler(timezone="Asia/Bangkok")

    # ดึงข้อมูลอัตโนมัติรอบเวลา 20:30 - 21:00 น. ทุก 5 นาที (เวลาหวยลาวพัฒนาออกผล)
    _scheduler.add_job(
        fetch_daily,
        trigger=CronTrigger(hour='20', minute='30,35,40,45,50,55'),
        id='fetch_daily_draw_time',
        replace_existing=True,
    )

    # ดึงเก็บตกช่วง 21:15 น. เผื่อเว็บต้นทางอัปเดตช้า
    _scheduler.add_job(
        fetch_daily,
        trigger=CronTrigger(hour=21, minute=15),
        id='fetch_daily_late_retry',
        replace_existing=True,
    )

    _scheduler.start()
    logger.info("[Scheduler] started - จะดึงข้อมูลอัตโนมัติช่วง 20:30 - 21:00 น. (ทุก 5 นาที)")


def stop():
    global _scheduler
    if _scheduler and _scheduler.running:
        _scheduler.shutdown()
        logger.info("[Scheduler] stopped")
