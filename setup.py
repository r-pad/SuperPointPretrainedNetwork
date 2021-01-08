#!/usr/bin/env python

from distutils.core import setup
from setuptools import find_packages

setup(
    name='superpoint',
    version='0.1dev',
    author='Daniel DeTone',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    description='PyTorch pre-trained model for real-time interest point detection, description, and sparse tracking (https://arxiv.org/abs/1712.07629)',
    long_description=open('README.md').read(),
    package_data = {'': ['*.pth']},
)

