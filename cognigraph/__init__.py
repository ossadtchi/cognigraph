__all__ = ["node", "pipeline"]
__version__ = '0.1.1'

import numpy as np
import os
import os.path as op
import logging
# from .node import Node, SourceNode, OutputNode, ProcessorNode
# from .pipeline import Pipeline

# -------- setup logging -------- #
TIMING_LEVEL_NUM = 9
logging.addLevelName(TIMING_LEVEL_NUM, "TIMING")
logging.TIMING = TIMING_LEVEL_NUM


def timing(self, message, *args, **kws):
    if self.isEnabledFor(TIMING_LEVEL_NUM):
        # Yes, logger takes its '*args' as 'args'.
        self._log(TIMING_LEVEL_NUM, message, args, **kws)


logging.Logger.timing = timing
# ------------------------------- #

TIME_AXIS = 1
CHANNEL_AXIS = 1 - TIME_AXIS
PYNFB_TIME_AXIS = 0

MISC_CHANNEL_TYPE = 'misc'

DTYPE = np.dtype('float32')
COGNIGRAPH_ROOT = op.split(op.dirname(__file__))[0]

COGNIGRAPH_DATA = op.join(COGNIGRAPH_ROOT, 'data')
if not op.isdir(COGNIGRAPH_DATA):
    os.mkdir(COGNIGRAPH_DATA, 0o755)

MONTAGES_DIR = op.join(COGNIGRAPH_DATA, 'custom_montages')
if not op.isdir(MONTAGES_DIR):
    os.mkdir(MONTAGES_DIR, 0o755)
ANATOMY_DIR = op.join(COGNIGRAPH_DATA, 'anatomy')
if not op.isdir(ANATOMY_DIR):
    os.mkdir(ANATOMY_DIR, 0o755)
FORWARDS_DIR = op.join(COGNIGRAPH_DATA, 'forwards')
if not op.isdir(FORWARDS_DIR):
    os.mkdir(FORWARDS_DIR, 0o755)

PIPELINES_DIR = op.join(COGNIGRAPH_DATA, 'pipelines')
if not op.isdir(PIPELINES_DIR):
    os.mkdir(PIPELINES_DIR, 0o755)
