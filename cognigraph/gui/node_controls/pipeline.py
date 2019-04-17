from ...utils.pyqtgraph import MyGroupParameter
from ...nodes.pipeline import Pipeline
import logging

__all__ = ('PipelineControls',)


class PipelineControls(MyGroupParameter):
    CONTROLS_LABEL = 'Pipeline settings'

    def __init__(self, pipeline: Pipeline = None, **kwargs):
        kwargs['name'] = self.CONTROLS_LABEL
        super().__init__(**kwargs)

        self._pipeline = pipeline  # type: self.OUTPUT_CLASS
        self._create_parameters()

        self._logger = logging.getLogger(type(self).__name__)
        self._logger.debug('Constructor called')

    def _create_parameters(self):
        pass
