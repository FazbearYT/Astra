"""
Модуль прогресс-баров
"""

import sys
from typing import Optional, Iterable
from contextlib import contextmanager

PROGRESS_ENABLED = True


class DummyProgressBar:
    def __init__(self, *args, **kwargs):
        pass

    def update(self, n=1):
        pass

    def close(self):
        pass

    def set_description(self, desc):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def __iter__(self):
        return iter([])


def get_progress_bar(iterable: Optional[Iterable] = None,
                     total: Optional[int] = None,
                     desc: str = "",
                     unit: str = "it",
                     disable: bool = False,
                     **kwargs):
    if not PROGRESS_ENABLED or disable:
        if iterable is not None:
            return iterable
        else:
            return DummyProgressBar()

    try:
        from tqdm import tqdm
        return tqdm(iterable=iterable, total=total, desc=desc, unit=unit, **kwargs)
    except ImportError:
        if iterable is not None:
            return iterable
        else:
            return DummyProgressBar()


def progress_range(start, stop=None, step=1, desc=""):
    if stop is None:
        stop = start
        start = 0

    total = (stop - start) // step

    if not PROGRESS_ENABLED:
        return range(start, stop, step)

    try:
        from tqdm import trange
        return trange(start, stop, step, desc=desc, total=total)
    except ImportError:
        return range(start, stop, step)


@contextmanager
def progress_context(message: str = "", total: int = 1):
    if not PROGRESS_ENABLED:
        yield DummyProgressBar()
        return

    try:
        from tqdm import tqdm
        with tqdm(total=total, desc=message, file=sys.stdout) as pbar:
            yield pbar
    except ImportError:
        yield DummyProgressBar()


def enable_progress():
    global PROGRESS_ENABLED
    PROGRESS_ENABLED = True


def disable_progress():
    global PROGRESS_ENABLED
    PROGRESS_ENABLED = False


def is_progress_enabled():
    return PROGRESS_ENABLED