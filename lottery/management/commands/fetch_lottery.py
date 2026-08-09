"""
Management command: fetch_lottery
Fetch Lao Development Lottery results from SerpAPI

Usage:
    python manage.py fetch_lottery
    python manage.py fetch_lottery --date 2024-01-15
    python manage.py fetch_lottery --history 30
    python manage.py fetch_lottery --predict
"""
import sys
import io

# Fix Windows console encoding
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ('utf-8', 'utf8'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
from datetime import date
from django.core.management.base import BaseCommand
from lottery.services.fetcher import fetch_and_save, fetch_history
from lottery.services.predictor import save_prediction


class Command(BaseCommand):
    help = 'ดึงข้อมูลหวยลาวพัฒนาจาก SerpAPI'

    def add_arguments(self, parser):
        parser.add_argument('--date', type=str, help='วันที่ต้องการ (YYYY-MM-DD)')
        parser.add_argument('--history', type=int, default=0, help='ดึงย้อนหลัง N วัน')
        parser.add_argument('--predict', action='store_true', help='ทำนายงวดถัดไปหลังดึงข้อมูล')

    def handle(self, *args, **options):
        print(f"[fetch_lottery] Starting...", flush=True)
        if options['history'] > 0:
            print(f"Fetching history {options['history']} days...", flush=True)
            results = fetch_history(days=options['history'])
            print(f"Done: {len(results)} records saved", flush=True)
        else:
            if options['date']:
                target = date.fromisoformat(options['date'])
            else:
                target = date.today()

            print(f"Fetching draw date: {target}", flush=True)
            obj, msg = fetch_and_save(target)

            if obj:
                print(f"[OK] {msg}", flush=True)
                print(f"  first_prize : {obj.first_prize}", flush=True)
                print(f"  two_digit   : {obj.two_digit}", flush=True)
                print(f"  three_digit : {obj.three_digit}", flush=True)
            else:
                print(f"[WARN] {msg}", flush=True)

        if options['predict']:
            print("\nGenerating prediction...", flush=True)
            pred = save_prediction()
            print(f"[OK] Prediction for {pred.target_date}: {pred.predicted_first} "
                  f"(confidence: {pred.confidence}%)", flush=True)
