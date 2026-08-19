from datetime import timedelta
import logging
from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.contrib import messages
from django.utils import timezone  # ใช้ timezone ของ Django แทน date.today()

from lottery.models import LotteryResult, Prediction, FetchLog
from lottery.services.predictor import predict_next, save_prediction, get_accuracy_stats, get_statistical_analysis
from lottery.services.fetcher import fetch_and_save
import traceback

# ตั้งค่า Logger สำหรับแสดงผลใน Render Logs
logger = logging.getLogger(__name__)


def dashboard(request):
    """หน้าหลัก - แสดงผลล่าสุด + prediction"""
    # ผลล่าสุด
    # ดึงวันที่ปัจจุบันตาม Timezone ที่ตั้งไว้ใน settings.py (Asia/Bangkok)
    today = timezone.localdate()
    
    latest = LotteryResult.objects.first()
    
    # ถ้ายังไม่มีผลหวยของวันนี้ในระบบ (เช่น กรณีผู้ใช้เปิดหน้าเว็บหลัง 20:30 น.) ให้ลองดึงอัตโนมัติทันที
    if not latest or latest.draw_date < today:
        try:
            fetch_and_save(today)
            latest = LotteryResult.objects.first()
        except Exception as e:
            logger.error(f"Auto-fetch failed on dashboard: {e}")
    
    # ค้นหา Prediction งวดถัดไป
    from lottery.services.utils import get_next_draw_date
    now_local = timezone.localtime()
    next_draw_date = get_next_draw_date(now_local)
    
    prediction = Prediction.objects.filter(target_date=next_draw_date).first()
    if not prediction:
        try:
            prediction = save_prediction(next_draw_date)
        except Exception as e:
            print(f"[Dashboard Prediction Error]: {e}")
            traceback.print_exc()
            logger.error(f"Prediction failed for {next_draw_date}: {e}", exc_info=True)
            prediction = None
            
    # แยกและประมวลผลข้อมูลสำหรับการแสดงผลพรีเมียม
    predicted_twos = []
    predicted_threes = []
    vote_breakdown = []
    key_digit = "N/A"
    secondary_digit = "N/A"
    
    if prediction:
        predicted_twos = [x.strip() for x in prediction.predicted_two.split(',')] if prediction.predicted_two else []
        predicted_threes = [x.strip() for x in prediction.predicted_three.split(',')] if prediction.predicted_three else []
        
        try:
            from lottery.services.predictor import predict_next
            pred_data = predict_next(prediction.target_date)
            vote_breakdown = pred_data.get('vote_breakdown', [])
            key_digit = pred_data.get('key_digit', 'N/A')
            secondary_digit = pred_data.get('secondary_digit', 'N/A')
        except Exception as e:
            logger.error(f"Could not calculate vote breakdown: {e}")
    
    # ประวัติย้อนหลัง 10 งวด
    recent_results = LotteryResult.objects.order_by('-draw_date')[:10]
    
    # accuracy stats
    stats = get_accuracy_stats()
    
    # hot numbers (ออกบ่อยใน 20 งวดล่าสุด)
    hot_numbers = _calc_hot_numbers(20)
    
    # statistical analysis (Sum Window, Digital Root, etc.)
    stat_analysis = get_statistical_analysis(30)
    
    context = {
        'latest': latest,
        'prediction': prediction,
        'predicted_twos': predicted_twos,
        'predicted_threes': predicted_threes,
        'vote_breakdown': vote_breakdown,
        'key_digit': key_digit,
        'secondary_digit': secondary_digit,
        'recent_results': recent_results,
        'stats': stats,
        'hot_numbers': hot_numbers,
        'stat_analysis': stat_analysis,
        'today': today,
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
                from lottery.services.utils import get_next_draw_date
                next_draw_date = get_next_draw_date(timezone.localtime())
                save_prediction(next_draw_date)
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
            from lottery.services.utils import get_next_draw_date
            next_draw_date = get_next_draw_date(timezone.localtime())
            pred = save_prediction(next_draw_date)
            
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