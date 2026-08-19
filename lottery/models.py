from django.db import models
from django.utils import timezone


class LotteryResult(models.Model):
    """เก็บผลหวยลาวพัฒนาแต่ละงวด"""
    draw_date = models.DateField(unique=True, verbose_name="วันที่ออกรางวัล")
    
    # รางวัลที่ 1 (ตัวเลข 6 หลัก)
    first_prize = models.CharField(max_length=20, blank=True, verbose_name="รางวัลที่ 1")
    
    # เลขท้าย 2 ตัว / 3 ตัว (หากมี)
    two_digit = models.CharField(max_length=10, blank=True, verbose_name="2 ตัวท้าย")
    two_digit_top = models.CharField(max_length=10, blank=True, verbose_name="2 ตัวบน")
    three_digit = models.CharField(max_length=10, blank=True, verbose_name="3 ตัวท้าย")
    four_digit = models.CharField(max_length=10, blank=True, verbose_name="4 ตัวท้าย")
    
    # ข้อมูลดิบจาก API
    raw_data = models.TextField(blank=True, verbose_name="ข้อมูลดิบ")
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-draw_date']
        verbose_name = "ผลหวยลาวพัฒนา"
        verbose_name_plural = "ผลหวยลาวพัฒนาทั้งหมด"
    
    def __str__(self):
        return f"งวด {self.draw_date} - {self.first_prize}"
    
    @property
    def digits(self):
        """แยกตัวเลขออกมาเป็น list"""
        return [int(d) for d in self.first_prize if d.isdigit()]
    
    @property
    def last_two(self):
        if self.two_digit:
            return self.two_digit
        if len(self.first_prize) >= 2:
            return self.first_prize[-2:]
        return ""
    
    @property
    def last_three(self):
        if self.three_digit:
            return self.three_digit
        if len(self.first_prize) >= 3:
            return self.first_prize[-3:]
        return ""
    
    @property
    def last_four(self):
        if self.four_digit:
            return self.four_digit
        if len(self.first_prize) >= 4:
            return self.first_prize[-4:]
        return ""
    
    @property
    def last_two_top(self):
        if self.two_digit_top:
            return self.two_digit_top
        if len(self.first_prize) >= 4:
            return self.first_prize[2:4]
        return ""


class Prediction(models.Model):
    """เก็บผลการทำนายแต่ละงวด"""
    target_date = models.DateField(verbose_name="งวดที่ทำนาย")
    predicted_first = models.CharField(max_length=20, verbose_name="ทำนายรางวัลที่ 1")
    predicted_two = models.CharField(max_length=10, blank=True, verbose_name="ทำนาย 2 ตัวท้าย")
    predicted_two_top = models.CharField(max_length=100, blank=True, verbose_name="ทำนาย 2 ตัวบน")
    predicted_three = models.CharField(max_length=10, blank=True, verbose_name="ทำนาย 3 ตัวท้าย")
    predicted_four = models.CharField(max_length=100, blank=True, verbose_name="ทำนาย 4 ตัวท้าย")
    confidence = models.FloatField(default=0.0, verbose_name="ความมั่นใจ (%)")
    model_used = models.CharField(max_length=50, default="ensemble", verbose_name="โมเดลที่ใช้")
    key_digit = models.CharField(max_length=5, default="N/A", verbose_name="เลขเด่นหลัก")
    secondary_digit = models.CharField(max_length=5, default="N/A", verbose_name="เลขเด่นรอง")
    vote_breakdown = models.JSONField(null=True, blank=True, verbose_name="รายละเอียดคะแนนโหวต")
    
    # เปรียบเทียบกับผลจริง
    actual_result = models.ForeignKey(
        LotteryResult, null=True, blank=True,
        on_delete=models.SET_NULL,
        related_name='predictions',
        verbose_name="ผลจริง"
    )
    is_correct_two = models.BooleanField(null=True, blank=True, verbose_name="ถูก 2 ตัวท้าย")
    is_correct_two_top = models.BooleanField(null=True, blank=True, verbose_name="ถูก 2 ตัวบน")
    is_correct_three = models.BooleanField(null=True, blank=True, verbose_name="ถูก 3 ตัวท้าย")
    is_correct_four = models.BooleanField(null=True, blank=True, verbose_name="ถูก 4 ตัวท้าย")
    is_correct_first = models.BooleanField(null=True, blank=True, verbose_name="ถูกรางวัลที่ 1")
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-target_date']
        verbose_name = "การทำนาย"
        verbose_name_plural = "ประวัติการทำนาย"
    
    def __str__(self):
        return f"ทำนายงวด {self.target_date} -> {self.predicted_first}"
    
    def evaluate(self):
        """เปรียบเทียบกับผลจริงและบันทึก (รองรับชุดเลขทำนายหลายชุด)"""
        if not self.actual_result:
            return
        real = self.actual_result
        
        twos = [t.strip() for t in self.predicted_two.replace('/', ',').split(',') if t.strip()]
        twos_top = [t.strip() for t in self.predicted_two_top.replace('/', ',').split(',') if t.strip()]
        threes = [t.strip() for t in self.predicted_three.replace('/', ',').split(',') if t.strip()]
        fours = [t.strip() for t in self.predicted_four.replace('/', ',').split(',') if t.strip()]
        firsts = [f.strip() for f in self.predicted_first.replace('/', ',').split(',') if f.strip()]

        self.is_correct_first = (real.first_prize in firsts) if firsts else False
        # ตรวจสอบ 2 ตัวล่าง (two) และ 2 ตัวบน (two_top) โดยอิงจากลิสต์แนะนำ
        all_predicted_twos = twos + twos_top
        self.is_correct_two = (real.last_two in all_predicted_twos) if all_predicted_twos else False
        self.is_correct_two_top = (real.last_two_top in all_predicted_twos) if all_predicted_twos else False
        self.is_correct_three = (real.last_three in threes) if threes else False
        self.is_correct_four = (real.last_four in fours) if fours else False
        self.save()


class FetchLog(models.Model):
    """บันทึก log การดึงข้อมูล"""
    STATUS_CHOICES = [
        ('success', 'สำเร็จ'),
        ('fail', 'ล้มเหลว'),
        ('partial', 'บางส่วน'),
    ]
    fetched_at = models.DateTimeField(auto_now_add=True)
    status = models.CharField(max_length=10, choices=STATUS_CHOICES)
    message = models.TextField(blank=True)
    records_saved = models.IntegerField(default=0)
    
    class Meta:
        ordering = ['-fetched_at']
        verbose_name = "Log การดึงข้อมูล"
    
    def __str__(self):
        return f"{self.fetched_at.strftime('%Y-%m-%d %H:%M')} - {self.status}"
