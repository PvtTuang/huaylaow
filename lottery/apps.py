from django.apps import AppConfig


class LotteryConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'lottery'
    verbose_name = 'ระบบหวยลาวพัฒนา'

    def ready(self):
        """Start background scheduler when app is ready (only once, not on autoreload)"""
        import os
        # Avoid double-start: Django autoreloader sets RUN_MAIN=true in the child process.
        # We start only in the child (RUN_MAIN=true) OR when not using autoreloader at all.
        run_main = os.environ.get('RUN_MAIN')
        if run_main == 'true' or run_main is None:
            try:
                from lottery.services import scheduler
                scheduler.start()
            except Exception as e:
                import logging
                logging.getLogger(__name__).error(f"Scheduler start error: {e}")
