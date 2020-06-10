import shutil
import os

def onerror(func, path, exc_info):
    """
    Error handler for ``shutil.rmtree``.

    If the error is due to an access error (read only file)
    it attempts to add write permission and then retries.

    If the error is for another reason it re-raises the error.

    Usage : ``shutil.rmtree(path, onerror=onerror)``
    """
    import stat
    if not os.access(path, os.W_OK):
        # Is the error an access error ?
        os.chmod(path, stat.S_IWUSR)
        func(path)
    else:
        raise

def delete_tree(dir):
    if os.path.exists(dir):
        print(f"Deleting {dir}")
        shutil.rmtree(dir, onerror=onerror)
    else:
        print(f"{dir} does not exist")
