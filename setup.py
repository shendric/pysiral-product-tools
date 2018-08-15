# -*- coding: utf-8 -*-

from setuptools import setup, find_packages
import os

# Get the readme
with open('README.rst') as f:
    readme = f.read()

# Get the licence
with open('LICENSE') as f:
    license = f.read()

# Get the version
mypackage_root_dir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(mypackage_root_dir, "pysiral_product_tools", 'VERSION')) as version_file:
    version = version_file.read().strip()

# Package requirements
with open("requirements.txt") as f:
    requirements = f.read().splitlines()


setup(
    name='pysiral-product-tools',
    version=version,
    description='tools for working with pysiral data',
    long_description=readme,
    author='Stefan Hendricks',
    author_email='stefan.hendricks@awi.de',
    url='https://github.com/shendric/pysiral-product-tools',
    license=license,
    install_requires=requirements,
    packages=find_packages(exclude=('tests')),
    include_package_data=True
)