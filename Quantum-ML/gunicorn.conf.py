import os

bind = f"0.0.0.0:{os.environ.get('PORT', '10000')}"   # Render gives PORT
workers = 1          # 1 worker: models + warm-up thread stay in one process (low RAM)
threads = 4
timeout = 180        # first quantum model load can be slow
graceful_timeout = 30
accesslog = "-"
errorlog = "-"
