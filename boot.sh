#!/bin/sh -e
exec gunicorn app:app -b 0.0.0.0:80 --workers $WORKERS --threads $THREADS --timeout $TIMEOUT