import os
from celery import Celery

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'book_tree.settings')

app = Celery('book_tree')
app.config_from_object('django.conf:settings', namespace='CELERY')
app.autodiscover_tasks()