from setuptools import setup
from Cython.Build import cythonize

setup(
    name='browser_handler',
    ext_modules=cythonize("browser_handler.py", language_level=3),
)