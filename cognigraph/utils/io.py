"""
Download cognigraph data from the cloud.

Exposed classes
---------------
DataDownloader: object
    Downloader class for data in the cloud

"""
import json
import os
import shutil
import logging
import os.path as op
from socket import timeout
from tempfile import NamedTemporaryFile
from io import DEFAULT_BUFFER_SIZE as DEFAULT_BLOCKSIZE
from hashlib import md5
from urllib.error import HTTPError, URLError
from urllib.request import urlopen
from cognigraph import COGNIGRAPH_ROOT
import ssl

_context = ssl._create_unverified_context()
_cur_dir = op.join(COGNIGRAPH_ROOT, op.dirname(__file__))


class _SafeConnection():
    """Recover from internet connection problems"""
    def __init__(self, logger):
        self._logger = logger
        self.is_stdout_broken = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type is not None and self.is_stdout_broken:
            print('\n')
        if exc_type in (HTTPError, URLError):
            self._logger.error('Connection problem: %s, %s', exc_value,
                               exc_traceback)
        elif exc_type is timeout:
            self._logger.error('Connection timed out.')
        elif exc_type is Exception:
            self._logger.exception(exc_value)
        return False  # don't propagate exceptions


class DataDownloader():
    """
    Get requested data either from local folders (if present) or online.

    Print pretty progressbar to stdout.
    Connection failures don't raise exceptions -- instead the log is updated.

    Parameters
    ----------
    cfg_path: str, optional
        Path to .json file with urls config

    Methods
    -------
    get_file()
        Return path to file; download if necessary

    """
    def __init__(self, cfg_path=op.join(_cur_dir, 'config/download_cfg.json')):
        self._config_path = (cfg_path)
        with open(self._config_path, 'r') as f:
            download_config = json.load(f)
            self._API_ENDPOINT = download_config['API_ENDPOINT']
        self._logger = logging.getLogger(type(self).__name__)

    def _dl_progress(self, fname, i_block, blocksize, totalsize):
        """
        Print terminal progressbar.

        Progressbar is designed to fit into 80 characters space.

        """
        NSTEP = 21  # number of progress bar steps
        scale = 1024 ** 2  # convert to MB
        readsofar = i_block * blocksize / scale
        totalsize = totalsize / scale
        if totalsize > 0:
            frac = min(1, readsofar / totalsize)
            percent = frac * 100
            i_step = int(frac * NSTEP)
            if i_step == 0:
                arr = '.'
            elif i_step < NSTEP:
                arr = '>'
            else:
                arr = '='
            s = ('\r{fname:20} {percent:5.1f}% ' +
                 '  [' + '=' * i_step + arr + '.' * (NSTEP - i_step) + '] ' +
                 '{readsofar:4.2f} / {totalsize:.2f} MB')
            s = s.format(fname=fname + ':', percent=percent,
                         readsofar=readsofar, totalsize=totalsize)
            print(s, end='')
            if readsofar >= totalsize:  # last block
                print(" >> DONE")

    def _fetch_fname(self, fname):
        with open(self._config_path, 'r') as f:
            download_config = json.load(f)
            dl_file = download_config['test_files'][fname]
            url_pub = dl_file['href']
            rel_path = dl_file['path']
            md5_hash = dl_file['md5']
            abs_path = op.join(COGNIGRAPH_ROOT, rel_path)
        return url_pub, abs_path, md5_hash

    def _download(self, url, dest_path, md5_hash, blocksize=DEFAULT_BLOCKSIZE):
        i_block = 1
        with _SafeConnection(self._logger) as s:
            with urlopen(url, timeout=5, context=_context) as r,\
                    NamedTemporaryFile(delete=False) as f:
                totalsize = int(r.getheader('content-length'))
                temp_fname = f.name
                cur_md5_hash = md5()
                while True:
                    chunk = r.read(blocksize)
                    if not chunk:
                        break
                    self._dl_progress(op.basename(dest_path), i_block,
                                      blocksize, totalsize)
                    # flag 'is_stdout_broken' signals _SafeConnection to fix
                    # stdout in case of exception (otherwise next print
                    # will be on the same line with the progress bar)
                    s.is_stdout_broken = True
                    i_block += 1
                    cur_md5_hash.update(chunk)
                    f.write(chunk)
                if cur_md5_hash.hexdigest() == md5_hash:
                    f.close()
                    os.makedirs(op.dirname(dest_path), exist_ok=True)
                    shutil.copy(temp_fname, dest_path)
                    os.remove(temp_fname)
                else:
                    self._logger.error(
                        'Bad checksum for file "%s"', op.basename(dest_path))

    def _get_download_url(self, url_pub):
        url = None
        with _SafeConnection(self._logger):
            with urlopen(self._API_ENDPOINT % url_pub,
                         timeout=10, context=_context) as r:
                raw_url = r.read().decode()
                url = raw_url.split(',')[0][9:-1]  # get href from url response
        return url

    def get_file(self, fname):
        try:
            url_pub, path, md5_hash = self._fetch_fname(fname)
        except json.JSONDecodeError:
            self._logger.error('Bad config file %s', self._config_path)
            return None
        except KeyError:
            self._logger.error('Requested entry %s is broken.'
                               ' Check %s' % (fname, self._config_path))
            return None
        except Exception as exc:
            self._logger.exception(exc)
            return None

        if not op.isfile(path):
            downloadable_url = self._get_download_url(url_pub)
            if downloadable_url:
                self._logger.info('Downloading "%s"', fname)
                self._download(downloadable_url, path, md5_hash)

        else:
            self._logger.info('Getting "%s" from "%s"', fname, path)
        return path


if __name__ == '__main__':
    logging.basicConfig(filename=None, level=logging.INFO)
    dloader = DataDownloader()
    path = dloader.get_file('Koleno_raw.fif')
