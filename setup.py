# from nltools.version import __version__
from setuptools import setup, find_packages

__version__ = '0.0.5'

# try:
#     from setuptools.core import setup
# except ImportError:
#     from distutils.core import setup
extra_setuptools_args = dict(
    tests_require=['pytest']
)

setup(
    name='facesync',
    version=__version__,
    author='Jin Hyun Cheong',
    author_email='jcheong.gr@dartmouth.edu',
    url='https://github.com/jcheong0428/facesync',
    download_url = 'https://github.com/jcheong0428/facesync/tarball/0.5',
    install_requires=['numpy', 'scipy'],
    packages=find_packages(exclude=['facesync/tests']),
    package_data={'facesync': ['resources/*']},
    license='LICENSE.txt',
    description='A Python package to sync videos based on audio',
    long_description='facesync is a python package that allows users to synchronize multiple videos based on audio.',
    keywords = ['psychology', 'preprocessing', 'video','audio','facecam','syncing'],
    classifiers = [
        "Programming Language :: Python",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License"
    ],
    **extra_setuptools_args
)

