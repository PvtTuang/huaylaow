"""
management command: repredict_all
Clear old Prediction records and re-run the current model (bayes_pool_v4) walk-forward:
every draw is predicted using ONLY the results that came before it, so the stats are honest.

Usage:
  python manage.py repredict_all
  python manage.py repredict_all --last 120
"""
from django.core.management.base import BaseCommand
from django.db.models import Q
from lottery.models import LotteryResult, Prediction
from lottery.services.predictor import MODEL_VERSION, rebuild_history, save_prediction


class Command(BaseCommand):
    help = f"Clear all Predictions and re-run {MODEL_VERSION} walk-forward for historical draws"

    def add_arguments(self, parser):
        parser.add_argument(
            '--last', type=int, default=120,
            help='How many most-recent draws to rebuild (default 120)',
        )

    def handle(self, *args, **options):
        deleted, _ = Prediction.objects.all().delete()
        self.stdout.write(f"Deleted {deleted} predictions")

        total_results = LotteryResult.objects.exclude(first_prize='').count()
        self.stdout.write(f"Found {total_results} draws. Walk-forward with {MODEL_VERSION} ...")

        created = rebuild_history(last_n=options['last'])
        self.stdout.write(f"Created {created} walk-forward predictions")

        # prediction for the upcoming draw
        pred = save_prediction()
        self.stdout.write(f"Next draw {pred.target_date}: {pred.predicted_first} "
                          f"| 2d {pred.predicted_two} | 3d {pred.predicted_three} "
                          f"| P(2d hit) {pred.confidence}%")

        evaluated = Prediction.objects.filter(actual_result__isnull=False)
        total = evaluated.count()
        if total:
            c2 = evaluated.filter(Q(is_correct_two=True) | Q(is_correct_two_top=True)).count()
            c3 = evaluated.filter(is_correct_three=True).count()
            c1 = evaluated.filter(is_correct_first=True).count()
            self.stdout.write(self.style.SUCCESS(
                f"\nResult ({total} draws evaluated):\n"
                f"  2-digit  : {c2}/{total} = {c2/total*100:.1f}%   (pure chance ~5.9%)\n"
                f"  3-digit  : {c3}/{total} = {c3/total*100:.1f}%   (pure chance ~0.3%)\n"
                f"  6-digit  : {c1}/{total} = {c1/total*100:.1f}%   (pure chance ~0.0001%)"
            ))
        else:
            self.stdout.write(self.style.WARNING("No evaluated predictions"))
