import os

def create_dirs(d):
    if not os.path.exists(d):
        os.makedirs(d, exist_ok=True)