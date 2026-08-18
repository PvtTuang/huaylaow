from datetime import timedelta
import logging
from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.contrib import messages
from django.utils import timezone  # ใช้ timezone ของ Django แทน date.today()

from lottery.models import LotteryResult, Prediction, FetchLog
from lottery.services.predictor import predict_next, save_prediction, get_accuracy_stats
from lottery.services.fetcher import fetch_and_save
import traceback

# ตั้งค่า Logger สำหรับแสดงผลใน Render Logs
logger = logging.getLogger(__name__)


def dashboard(request):
    """หน้าหลัก - แสดงผลล่าสุด + prediction"""
    # ผลล่าสุด
    latest = LotteryResult.objects.first()
    
    # ดึงวันที่ปัจจุบันตาม Timezone ที่ตั้งไว้ใน settings.py (Asia/Bangkok)
    today = timezone.localdate()
    tomorrow = today + timedelta(days=1)
    
    # ค้นหา Prediction งวดถัดไป
    # ถ้ายังไม่มี prediction ของพรุ่งนี้ → สร้างใหม่ทันที
    prediction = Prediction.objects.filter(target_date=tomorrow).first()
    if not prediction:
        try:
            prediction = save_prediction(tomorrow)
        except Exception as e:
            print(f"[Dashboard Prediction Error]: {e}")
            traceback.print_exc()
            logger.error(f"Prediction failed for {tomorrow}: {e}", exc_info=True)
            prediction = None
    
    # ประวัติย้อนหลัง 10 งวด
    recent_results = LotteryResult.objects.order_by('-draw_date')[:10]
    
    # accuracy stats
    stats = get_accuracy_stats()
    
    # hot numbers (ออกบ่อยใน 20 งวดล่าสุด)
    hot_numbers = _calc_hot_numbers(20)
    
    context = {
        'latest': latest,
        'prediction': prediction,
        'recent_results': recent_results,
        'stats': stats,
        'hot_numbers': hot_numbers,
        'today': today,
        'tomorrow': tomorrow,
    }
    return render(request, 'lottery/dashboard.html', context)


def history(request):
    """หน้าประวัติผลหวยทั้งหมด"""
    results = LotteryResult.objects.order_by('-draw_date')
    predictions = Prediction.objects.order_by('-target_date')[:30]
    stats = get_accuracy_stats()
    
    context = {
        'results': results,
        'predictions': predictions,
        'stats': stats,
    }
    return render(request, 'lottery/history.html', context)


def fetch_now(request):
    """AJAX/POST: ดึงข้อมูลวันนี้ทันที"""
    if request.method == 'POST':
        today = timezone.localdate()
        obj, msg = fetch_and_save(today)
        if obj:
            # สร้าง prediction ใหม่หลังได้ข้อมูล
            try:
                save_prediction(today + timedelta(days=1))
            except Exception as e:
                print(f"❌ [Fetch Prediction Error]: {e}")
                logger.error(f"Fetch prediction failed: {e}", exc_info=True)
                
            return JsonResponse({
                'status': 'success', 
                'message': msg,
                'first_prize': obj.first_prize
            })
        return JsonResponse({'status': 'error', 'message': msg})
    return JsonResponse({'status': 'error', 'message': 'POST only'})


def refresh_prediction(request):
    """AJAX/POST: สร้าง prediction ใหม่"""
    if request.method == 'POST':
        try:
            today = timezone.localdate()
            tomorrow = today + timedelta(days=1)
            pred = save_prediction(tomorrow)
            
            if not pred:
                return JsonResponse({'status': 'error', 'message': 'Prediction returned None'})
                
            return JsonResponse({
                'status': 'success',
                'predicted_first': pred.predicted_first,
                'predicted_two': pred.predicted_two,
                'predicted_three': pred.predicted_three,
                'confidence': pred.confidence,
            })
        except Exception as e:
            print(f"❌ [Refresh Prediction Error]: {e}")
            logger.error(f"Refresh prediction failed: {e}", exc_info=True)
            return JsonResponse({'status': 'error', 'message': str(e)})
    return JsonResponse({'status': 'error', 'message': 'POST only'})


def _calc_hot_numbers(num_draws=20) -> list:
    """คำนวณ hot numbers"""
    from collections import Counter
    results = LotteryResult.objects.exclude(first_prize='').order_by('-draw_date')[:num_draws]
    counter = Counter()
    for r in results:
        for c in r.first_prize:
            if c.isdigit():
                counter[c] += 1
    return counter.most_common(5)