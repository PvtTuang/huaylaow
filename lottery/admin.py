from django.contrib import admin
from lottery.models import LotteryResult, Prediction, FetchLog


@admin.register(LotteryResult)
class LotteryResultAdmin(admin.ModelAdmin):
    list_display = ['draw_date', 'first_prize', 'two_digit', 'three_digit', 'created_at']
    list_filter = ['draw_date']
    search_fields = ['first_prize']
    ordering = ['-draw_date']


@admin.register(Prediction)
class PredictionAdmin(admin.ModelAdmin):
    list_display = ['target_date', 'predicted_first', 'confidence',
                    'is_correct_two', 'is_correct_three', 'is_correct_first']
    list_filter = ['is_correct_two', 'is_correct_first']
    ordering = ['-target_date']


@admin.register(FetchLog)
class FetchLogAdmin(admin.ModelAdmin):
    list_display = ['fetched_at', 'status', 'records_saved', 'message']
    list_filter = ['status']
    ordering = ['-fetched_at']
