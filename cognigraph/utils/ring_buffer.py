import numpy as np


class RingBufferSlow(object):
    """Represents a multi-row deque object"""
    TIME_AXIS = 1

    def __init__(self, row_cnt, maxlen):
        self.maxlen = maxlen
        self.row_cnt = row_cnt
        self._data = np.zeros((row_cnt, maxlen))
        self._start = 0
        self._curr_samp_count = 0

    def extend(self, array):
        self._check_input_shape(array)
        new_sample_cnt = array.shape[self.TIME_AXIS]

        # If new data will take all the space, we can forget about the old data
        if new_sample_cnt >= self.maxlen:
            # last self.maxlen samples
            indices = np.arange(-self.maxlen, 0) + new_sample_cnt
            self._data = array.take(indices=indices, axis=1)
            self._start = 0
            self._curr_samp_count = self.maxlen

        else:
            # New data should start after the end of the old one
            start = ((self._start + self._curr_samp_count) % self.maxlen)

            # Put as much as possible after start.
            end = min(start + new_sample_cnt, self.maxlen)
            self._data[:, start:end] = array[:, :(end-start)]

            # Then wrap around if needed
            if end - start < new_sample_cnt:
                end = (start + new_sample_cnt) % self.maxlen
                self._data[:, :end] = array[:, -end:]

            self._curr_samp_count = min(self._curr_samp_count + new_sample_cnt,
                                        self.maxlen)

            if self._curr_samp_count == self.maxlen:
                # The buffer is fully populated
                self._start = end % self.maxlen

    def _check_input_shape(self, array):
        if array.shape[0] != self.row_cnt:
            msg = ('Wrong shape. You are trying to extend a buffer with {} '
                   'rows with an array with {} rows'.format(self.row_cnt,
                                                            array.shape[0]))
            raise ValueError(msg)

    def clear(self):
        self._curr_samp_count = 0
        self._start = 0

    @property
    def data(self):
        indices = self._start + np.arange(self._curr_samp_count)
        return self._data.take(indices=indices, axis=self.TIME_AXIS,
                               mode='wrap')

    @property
    def test_data(self):
        return np.concatenate((self._data[:, self._start:],
                               self._data[:, :self._start]), axis=1)


class RingBuffer(object):
    """
    Represents a multi-row deque object.
    Very memory-inefficient (all data is saved twice).
    This allows us to return views and not copies of the data.

    """

    TIME_AXIS = 1

    def __init__(self, row_cnt, maxlen):
        self.maxlen = maxlen
        self.row_cnt = row_cnt
        self._data = np.zeros((row_cnt, maxlen * 2))
        self._start = 0
        self._curr_samp_count = 0

    def extend(self, array):
        self._check_input_shape(array)
        new_sample_cnt = array.shape[self.TIME_AXIS]

        # If new data will take all the space, we can forget about the old data
        if new_sample_cnt >= self.maxlen:
            self._data[:, :self.maxlen] = array[:, -self.maxlen:]
            self._data[:, self.maxlen:] = array[:, -self.maxlen:]
            self._start = 0
            self._curr_samp_count = self.maxlen

        else:
            # New data should start after the end of the old one
            start = (self._start + self._curr_samp_count) % self.maxlen

            # Put as much as possible after start.
            end = min(start + new_sample_cnt, self.maxlen)
            self._data[:, start:end] = array[:, :(end-start)]
            self._data[:, (start + self.maxlen):(end + self.maxlen)] \
                = array[:, :(end-start)]

            # Then wrap around if needed
            if end - start < new_sample_cnt:
                end = (start + new_sample_cnt) % self.maxlen
                self._data[:, :end] = array[:, -end:]
                self._data[:, self.maxlen:(end + self.maxlen)] = (
                    array[:, -end:])

            self._curr_samp_count = min(
                self._curr_samp_count + new_sample_cnt, self.maxlen)
            if self._curr_samp_count == self.maxlen:
                # The buffer is fully populated
                self._start = end % self.maxlen

    def _check_input_shape(self, array):
        if array.shape[0] != self.row_cnt:
            msg = ('Wrong shape. You are trying to extend a buffer with {}'
                   ' rows with an array with {} rows'.format(self.row_cnt,
                                                             array.shape[0]))
            raise ValueError(msg)

    def clear(self):
        self._curr_samp_count = 0
        self._start = 0

    @property
    def data(self):
        return self._data[:, self._start:(self._start + self._curr_samp_count)]
