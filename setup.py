from setuptools import setup

setup(name='tfn', version='0.1.0',
      packages=['tfn', 'tfn/models', 'tfn/feature_extraction', 'tfn/data_augmentation'],
      package_data={'': ['data/*', 'misc/*']})
