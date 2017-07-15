import threading
import subprocess
import datetime
import traceback
import shutil
import psutil
import sys
import os

import numpy as np

from beaker.cache import CacheManager
from beaker.util import parse_cache_config_options

import logging
import time
import colorlog
from tqdm import tqdm
import re
from operator import itemgetter


PDB_PATTERN = '[0-9][0-9a-z]{3}'    # check if first letter as to be numeral
ECOD_PATTERN = 'e[0-9][0-9a-z]{3}[0-9A-Za-z\.]{1,2}[0-9]'   # ecod domain pattern
PIFACE_PATTERN = '[0-9][0-9A-Z]{3}[A-Z][A-Z]'    # piface interface pattern


def split_iter(string):
    return (x.group(0) for x in re.finditer(r"[A-Za-z']+", string))


class TqdmHandler(logging.StreamHandler):
    def __init__(self):
        logging.StreamHandler.__init__(self)

    def emit(self, record):
        msg = self.format(record)
        tqdm.write(msg)


def get_logger(name):
    logger = colorlog.getLogger(name)
    logger.setLevel(logging.DEBUG)
    handler = TqdmHandler()
    handler.setFormatter(colorlog.ColoredFormatter(
        '%(log_color)s%(name)s | %(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%d-%d %H:%M:%S',
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'SUCCESS:': 'white',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white'}, ))
    logger.addHandler(handler)
    return logger


def unforamt(raw_string, fmt):
    tmpl = re.sub(r'\\{(\d+)\\}', r'(?P<_\1>.+)', re.escape(fmt))
    matches = re.match(tmpl, raw_string)
    return tuple(map(itemgetter(1), sorted(matches.groupdict().items())))

#########################################################################

def ispdbid(s):
    return re.match(PDB_PATTERN,s)

def isecodid(s):
    return re.match(ECOD_PATTERN,s)

def ispifaceid(s):
    return re.match(PIFACE_PATTERN,s)


cache_opts = {
    'cache.type':       'memory',
    'cache.lock_dir': '/tmp/cache/lock'
}

CACHE_MANAGER = CacheManager(**parse_cache_config_options(cache_opts))
#TR = tracker.SummaryTracker()


_proc_status = '/proc/%d/status' % os.getpid()

_scale = {'kB': 1024.0, 'mB': 1024.0*1024.0,
          'KB': 1024.0, 'MB': 1024.0*1024.0}


def _VmB(VmKey):
    '''Private.
    '''
    global _proc_status, _scale
     # get pseudo file  /proc/<pid>/status
    try:
        t = open(_proc_status)
        v = t.read()
        t.close()
    except:
        return 0.0  # non-Linux?
     # get VmKey line e.g. 'VmRSS:  9999  kB\n ...'
    i = v.index(VmKey)
    v = v[i:].split(None, 3)  # whitespace
    if len(v) < 3:
        return 0.0  # invalid format?
     # convert Vm value to bytes
    return float(v[1]) * _scale[v[2]]


def memory(since=0.0):
    '''Return memory usage in bytes.
    '''
    return _VmB('VmSize:') - since


def handleError(err, data=None, braise=False, btrace=True):
    print("Err: %s Data: %s" % (err, data))
    if btrace: traceback.print_exc()
    if braise: raise err


def popenAndCall(onExit, popenArgs):
    """
    Runs the given args in a subprocess.Popen, and then calls the function
    onExit when the subprocess completes.
    onExit is a callable object, and popenArgs is a list/tuple of args that
    would give to subprocess.Popen.
    """
    def runInThread(onExit, popenArgs):
        proc = subprocess.Popen(popenArgs)
        proc.wait()
        onExit()
        return
    thread = threading.Thread(target=runInThread, args=(onExit, popenArgs))
    thread.start()
    # returns immediately after the thread starts
    return thread


def call(popenArgs, errlog):
    try:
        # return subprocess.check_call(popenArgs, stdin=None, stdout=open('/dev/null'), stderr=subprocess.STDOUT, shell=False)
        return subprocess.call(popenArgs, stdin=None, stdout=open('/dev/null'), stderr=open(errlog, 'w+'), shell=False)
    except OSError as err:
        handleError(err, braise=True)

class ProgressBar(object):
    def __init__(self, n):
        self.n = n
        self.indx = 0
        sys.stdout.write('\n')
        self.start = time.time()

    def update(self, i):
        start = self.start
        curr = time.time()
        secs = 1./(1.*i/self.n) * (curr - start)
        eta = datetime.timedelta(seconds=secs)
        sys.stdout.write("\r %s of %s ETA: %s" % (i, self.n, eta))

    def increment(self, i=1):
        self.indx += i
        self.update(self.indx)

    def finish(self):
        sys.stdout.write('\n')


def cleanup(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)


def ensure_exists(dirname):
    if not os.path.exists(dirname):
        try:
            os.makedirs(dirname)
        except OSError:
            pass


if __name__ == '__main__':
    print('len(list()):%s\tlen([]):%s' % (len(list()), len([])))
    try:
        print('A'.split(":")[1])
    except IndexError as err:
        print(range(-1000, 1000000)[5])

    print(np.array([3., 7.]) < 4.)

    P = psutil.Process(os.getpid())
    print(P.memory_percent())
    print(memory())
