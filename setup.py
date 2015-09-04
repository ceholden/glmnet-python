import os
import sys

# Get version
with open('glmnet/version.py') as f:
    for line in f:
        if line.find('__version__') >= 0:
            version = line.split("=")[1].strip()
            version = version.strip('"')
            version = version.strip("'")
            continue

from numpy.distutils.misc_util import Configuration
from numpy.distutils.system_info import get_info

config = Configuration(
    'glmnet',
    parent_package=None,
    top_path=None
)

f_sources = ['glmnet/glmnet.pyf', 'glmnet/glmnet.f']
fflags = ['-fdefault-real-8', '-ffixed-form']

config.add_extension(name='_glmnet',
                     sources=f_sources,
                     extra_f77_compile_args=fflags,
                     extra_f90_compile_args=fflags)
config_dict = config.todict()

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(version=version,
          description='Python wrappers for the GLMNET package',
          author='David Warde-Farley',
          author_email='dwf@cs.toronto.edu',
          url='github.com/dwf/glmnet-python',
          license='GPL2',
          requires=['NumPy (>= 1.3)'],
          packages=['glmnet'],
          **config_dict)
