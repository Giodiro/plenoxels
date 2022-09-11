import os
from tqdm.auto import tqdm as original_tqdm
import functools

tqdm_disabled = os.environ.get('DISABLE_TQDM', False)
tqdm = original_tqdm
tqdm.__init__ = functools.partialmethod(
    tqdm.__init__, mininterval=15, maxinterval=60, delay=15, disable=tqdm_disabled)
