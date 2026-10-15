# Auto-loaded by gunicorn from the project root.
workers = 1                 # quantum models are RAM-heavy: 1 worker only
threads = 4
timeout = 300               # default is 30s -> that is what caused the 502 on VQC/QNN
graceful_timeout = 60
