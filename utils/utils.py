import os

def create_dir(dir):
    if not os.path.exists(dir):
        print(f"Creating output directory at {dir}")
        os.mkdir(dir)
