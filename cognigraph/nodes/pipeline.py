import time
from typing import List
import json

from .node import Node

# from ..utils.decorators import accepts
# from ..utils.misc import class_name_of

import logging


class Pipeline(Node):
    """
    This class facilitates connecting data inputs to a sequence of signal
    processors and outputs.

    All elements in the pipeline are objects of class Node and inputs,
    processors and outputs should be objects of the
    corresponding subclasses of Node.

    """

    UPSTREAM_CHANGES_IN_THESE_REQUIRE_REINITIALIZATION = ()
    CHANGES_IN_THESE_REQUIRE_RESET = ()
    ALLOWED_CHILDREN = ("LSLStreamSource", "FileSource")
    _GUI_STRING = "Pipeline"

    def __init__(self):
        Node.__init__(self)
        self._logger = logging.getLogger(type(self).__name__)

    def _initialize(self):
        pass

    def _update(self):
        pass

    def _reset(self):
        pass

    def _check_value(self, key, value):
        pass

    def __repr__(self):
        return self._GUI_STRING

    @property
    def all_nodes(self) -> List[Node]:
        return list(self)

    @property
    def frequency(self) -> (int, float):
        try:
            return self.source.mne_info["sfreq"]
        except AttributeError:
            raise ValueError("No source has been set in the pipeline")

    def chain_initialize(self):
        self._logger.info("Start initialization")
        t1 = time.time()
        Node.chain_initialize(self)
        t2 = time.time()
        self._logger.info(
            "Finish initialization in {:.1f} ms".format((t2 - t1) * 1000)
        )

    def update(self):
        self._logger.debug("Start update " + ">" * 6)
        t1 = time.time()
        Node.update(self)
        t2 = time.time()
        self._logger.debug("Finish in {:.1f} ms".format((t2 - t1) * 1000))

    def save_pipeline(self, db_name):
        save_dict = self._save_dict()
        # with shelve.open(db_name, 'c') as db:
        with open(db_name, 'w') as db:
            json.dump(save_dict, db, indent=2)
