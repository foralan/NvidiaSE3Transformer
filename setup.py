from setuptools import setup, find_packages

setup(
    name='se3-transformer',
    packages=find_packages(exclude=['tests']),
    include_package_data=True,
    version='2.0.0',
    description='PyTorch + DGL implementation of SE(3)-Transformers removed unuseful codes, add compatibility for cpu',
    author='Alexandre Milesi',
    author_email='alexandrem@nvidia.com',
)
