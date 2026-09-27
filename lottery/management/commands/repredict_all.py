"""
management command: repredict_all
Clear all Prediction records then re-predict every draw that has an actual result,
simulating "what we would have predicted on that day" using only data available
before each draw.

Usage:
  python manage.py repredict_all
  python manage.py repredict_all --keep-future
"""
from django.core.management.base import BaseCommand
from django.utils import timezone
from lottery.models import LotteryResult, Prediction
from lottery.services.predictor import ensemble_predict, _extract_digits, _pad_or_trim


class Command(BaseCommand):
    help = "Clear all Predictions and re-run ensemble_v3 for every historical draw"

    def add_arguments(self, parser):
        parser.add_argument(
            '--keep-future',
            action='store_true',
            default=False,
            help='If set, keep predictions for future draws and only clear past ones',
        )

    def handle(self, *args, **options):
        today = timezone.localdate()
        keep_future = options['keep_future']

        # ── 1) Clear old predictions ──────────────────────────────────────
        if keep_future:
            deleted, _ = Prediction.objects.filter(target_date__lte=today).delete()
            self.stdout.write(f"Deleted {deleted} past predictions")
        else:
            deleted, _ = Prediction.objects.all().delete()
            self.stdout.write(f"Deleted {deleted} predictions (all)")

        # ── 2) Load all actual results (oldest first) ─────────────────────
        all_results = list(
            LotteryResult.objects.exclude(first_prize='').order_by('draw_date')
        )
        if not all_results:
            self.stdout.write(self.style.WARNING("No lottery results found in DB"))
            return

        self.stdout.write(f"Found {len(all_results)} draws. Re-predicting with ensemble_v3 ...")
        self.stdout.write("-" * 72)

        ok_count  = 0
        err_count = 0

        # ── 3) Re-predict each draw ───────────────────────────────────────
        # For draw[i]: train on draws[0..i-1] only (simulate knowledge at that time)
        for idx, actual_lr in enumerate(all_results):
            target_date = actual_lr.draw_date

            # history available BEFORE this draw (newest first, as predictor expects)
            train_history = list(reversed(all_results[:idx]))
            prize_len = len(_pad_or_trim(_extract_digits(actual_lr), 6))

            try:
                if len(train_history) < 3:
                    # Too few draws for first entries — use safe defaults
                    pred_first = "0" * prize_len
                    pred_two   = "00, 00, 00, 00"
                    pred_three = "000, 000, 000"
                    confidence = 60.0
                    key_digit  = "0"
                    sec_digit  = "0"
                    vb         = []
                else:
                    res = ensemble_predict(train_history, prize_len, target_date)
                    pred_first = res['predicted_first']
                    pred_two   = ', '.join(res['two_digit_pairs'])
                    pred_three = ', '.join(res['three_digit_sets'])
                    confidence = res['confidence']
                    key_digit  = res['key_digit']
                    sec_digit  = res['secondary_digit']
                    vb         = res['vote_breakdown']

                pred = Prediction.objects.create(
                    target_date=target_date,
                    predicted_first=pred_first,
                    predicted_two=pred_two,
                    predicted_three=pred_three,
                    confidence=confidence,
                    model_used='ensemble_v3',
                    key_digit=key_digit,
                    secondary_digit=sec_digit,
                    vote_breakdown=vb,
                    actual_result=actual_lr,
                )
                pred.evaluate()

                actual_str = actual_lr.first_prize
                h2 = "HIT" if pred.is_correct_two   else "---"
                h3 = "HIT" if pred.is_correct_three else "---"
                h1 = "HIT" if pred.is_correct_first  else "---"
                self.stdout.write(
                    f"  {target_date}  pred:{pred_first}  actual:{actual_str}"
                    f"  2d:{h2}  3d:{h3}  6d:{h1}  conf:{confidence:.0f}%"
                )
                ok_count += 1

            except Exception as e:
                self.stdout.write(self.style.ERROR(f"  {target_date} ERROR: {e}"))
                err_count += 1

        # ── 4) Summary ────────────────────────────────────────────────────
        self.stdout.write("-" * 72)
        evaluated = Prediction.objects.filter(actual_result__isnull=False)
        total = evaluated.count()

        if total > 0:
            from django.db.models import Q
            c2 = evaluated.filter(Q(is_correct_two=True) | Q(is_correct_two_top=True)).count()
            c3 = evaluated.filter(is_correct_three=True).count()
            c1 = evaluated.filter(is_correct_first=True).count()
            self.stdout.write(self.style.SUCCESS(
                f"\nResult ({total} draws evaluated):\n"
                f"  2-digit  : {c2}/{total} = {c2/total*100:.1f}%\n"
                f"  3-digit  : {c3}/{total} = {c3/total*100:.1f}%\n"
                f"  6-digit  : {c1}/{total} = {c1/total*100:.1f}%\n"
                f"  OK: {ok_count}  ERR: {err_count}"
            ))
        else:
            self.stdout.write(self.style.WARNING("No evaluated predictions"))
