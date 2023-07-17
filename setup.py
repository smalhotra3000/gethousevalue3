from setuptools import setup, find_packages

setup(
    name='gethousevalue3',
    version='1.0.0',
    description='This package trains & scores models for estimating housing value',
    author='Sameer Malhotra',
    author_email='sameer.malhotra@tigeranalytics.com',
    packages=find_packages(),
    install_requires=['numpy','pandas','six','scipy','scikit-learn']
)
