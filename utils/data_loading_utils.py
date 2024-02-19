import json
import sys

import numpy as np
import wget


def bar_progress(current, total, width=80):
    """Display progress bar while downloading.

    https://stackoverflow.com/questions/58125279/
    python-wget-module-doesnt-show-progress-bar"
    """

    progress_message = (
        f'Downloading: {current/total * 100:.0f} %% '
        f'[{current:.2e} / {total:.2e}] bytes')

    # Don't use print() as it will print in new line every time.
    sys.stdout.write("\r" + progress_message)
    sys.stdout.flush()


def download(paths, urls):
    # Download URLs to paths.

    if not isinstance(urls, list):
        urls = [urls]

    if not isinstance(paths, list):
        paths = [paths]

    if not len(urls) == len(paths):
        raise ValueError('Need exactly one path per URL.')

    for path, url in zip(paths, urls):
        print(f'Downloading {url}.')
        path.parent.mkdir(parents=True, exist_ok=True)
        wget.download(url, out=str(path), bar=bar_progress)