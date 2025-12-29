"""
Gunicorn configuration file
https://docs.gunicorn.org/en/stable/settings.html
"""
import os
import multiprocessing

# Server socket
bind = f"0.0.0.0:{os.getenv('PORT', '8080')}"
backlog = 2048

# Worker processes
workers = int(os.getenv('WORKERS', '2'))
worker_class = 'sync'
worker_connections = 1000
timeout = 180  # 3 minutes timeout for long-running AI requests
keepalive = 5
max_requests = 1000
max_requests_jitter = 50

# Logging
accesslog = '-'
errorlog = '-'
loglevel = 'info'
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Process naming
proc_name = 'mental_health_app'

# Server mechanics
daemon = False
pidfile = None
umask = 0
user = None
group = None
tmp_upload_dir = None

# Graceful timeout
graceful_timeout = 30
