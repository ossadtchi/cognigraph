''' Config file for setuptools '''
from setuptools import setup

import os

# -------- setup VERSION -------- #
version = None
with open(os.path.join('cognigraph', '__init__.py'), 'r') as fid:
    for line in (line.strip() for line in fid):
        if line.startswith('__version__'):
            version = line.split('=')[1].strip().strip('\'')
            break
# ------------------------------- #

VERSION = version
DISTNAME = 'cognigraph'

setup(
    name=DISTNAME,
    version=VERSION,
    install_requires=['mne==0.16', 'scipy>=1.0.0'],
    entry_points={
        'console_scripts': 'cognigraph=scripts.launch_cognigraph:main'
    }
)
