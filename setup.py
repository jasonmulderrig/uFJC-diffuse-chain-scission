import re
from os.path import join
from distutils.core import setup
from setuptools import find_packages


def read(fname):
    with open(fname) as fp:
        content = fp.read()
    return content


def get_version():
    VERSIONFILE = join('ufjc_diffuse_chain_scission', '__init__.py')
    with open(VERSIONFILE, 'rt') as f:
        lines = f.readlines()
    vgx = '^__version__ = \"[0-9+.0-9+.0-9+]*[a-zA-Z0-9]*\"'
    for line in lines:
        mo = re.search(vgx, line, re.M)
        if mo:
            return mo.group().split('"')[1]
    raise RuntimeError('Unable to find version in %s.' % (VERSIONFILE,))


setup(
    name='ufjc_diffuse_chain_scission',
    version=get_version(),
    package_dir={'ufjc_diffuse_chain_scission': 'ufjc_diffuse_chain_scission'},
    packages=find_packages(),
    description='The Python package for the (generalized composite) uFJC diffuse chain scission model, employing FEniCS.',
    long_description=read("README.rst"),
    author='Jason Mulderrig, Brandon Talamini, Nikolaos Bouklas',
    author_email='jpm445@cornell.edu, talamini1@llnl.gov, nb589@cornell.edu',
    url='https://github.com/jasonmulderrig/uFJC-diffuse-chain-scission',
    license='GPLv3+',
    keywords=['ufjc', 'diffuse chain scission', 'asymptotic matching', 'thermodynamics'],
    install_requires=['numpy', 'scipy'],
    extras_require={
      'docs': ['matplotlib',
               'sphinx', 'sphinx-rtd-theme', 'sphinxcontrib-bibtex'],
      'plotting': ['matplotlib'],
      'testing': ['matplotlib', 'pytest', 'pytest-cov'],
      'all': ['matplotlib', 'pytest', 'pytest-cov',
              'sphinx', 'sphinx-rtd-theme', 'sphinxcontrib-bibtex']
    },
    classifiers=[
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    project_urls={
      'GitHub': 'https://github.com/jasonmulderrig/uFJC-diffuse-chain-scission',
    },
)