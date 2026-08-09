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
    """Job: ดึงข้อมูลหวยวันนี้และทำนายงวดพรุ่งนี้"""
    try:
        from datetime import date, timedelta
        from lottery.services.fetcher import fetch_and_save
        from lottery.services.predictor import save_prediction

        today = date.today()
        obj, msg = fetch_and_save(today)
        logger.info(f"[Scheduler] fetch_daily: {msg}")

        if obj:
            pred = save_prediction(today + timedelta(days=1))
            logger.info(f"[Scheduler] prediction saved: {pred.predicted_first}")
    except Exception as e:
        logger.error(f"[Scheduler] error: {e}")


def start():
    """เริ่ม scheduler (เรียกจาก apps.py)"""
    global _scheduler
    if _scheduler and _scheduler.running:
        return

    _scheduler = BackgroundScheduler(timezone="Asia/Bangkok")

    # ดึงข้อมูลทุกวัน เวลา 18:30 (หลังหวยออก)
    _scheduler.add_job(
        fetch_daily,
        trigger=CronTrigger(hour=18, minute=30),
        id='fetch_daily',
        replace_existing=True,
    )

    # ดึงอีกรอบ 21:00 เผื่อข้อมูลอัปเดตช้า
    _scheduler.add_job(
        fetch_daily,
        trigger=CronTrigger(hour=21, minute=0),
        id='fetch_daily_retry',
        replace_existing=True,
    )

    _scheduler.start()
    logger.info("[Scheduler] started - จะดึงข้อมูลอัตโนมัติ 18:30 และ 21:00 น.")


def stop():
    global _scheduler
    if _scheduler and _scheduler.running:
        _scheduler.shutdown()
        logger.info("[Scheduler] stopped")
