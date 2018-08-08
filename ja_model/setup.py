from setuptools import setup


def readme():
    with open('README.rst') as f:
        return f.read()


setup(name='ja_model',
      version='0.1',
      description='Model building library',
      long_description=readme(),
      classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
        'Topic :: Model Building :: Quantitative Models',
      ],
      keywords='model plot statistics stats energy power',
      author='James Alvey',
      packages=['ja_model'],
      install_requires=['math',
                        'numpy',
                        'datetime',
                        'random',
                        'matplotlib',
                        'pandas',
                        'itertools',
                        'sklearn',
                        'statsmodels'
                       ],
      include_package_data=True,
      zip_safe=False)
