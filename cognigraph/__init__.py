__all__ = ["node", "pipeline"]
__version__ = '0.1.1'

import numpy as np
import os.path as op
# from .node import Node, SourceNode, OutputNode, ProcessorNode
# from .pipeline import Pipeline

# TODO: I wish this was an empty file

TIME_AXIS = 1
CHANNEL_AXIS = 1 - TIME_AXIS
PYNFB_TIME_AXIS = 0

MISC_CHANNEL_TYPE = 'misc'

DTYPE = np.dtype('float32')
COGNIGRAPH_ROOT = op.split(op.dirname(__file__))[0]
COGNIGRAPH_DATA = op.join(COGNIGRAPH_ROOT, 'data')
