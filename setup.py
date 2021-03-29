from setuptools import find_packages, setup

requirements = [
    'setuptools',
]

setup(
    name='noisy-cifar',
    version='0.1.0',
    author='Holim Lim',
    author_email='ihl7029@europa.snu.ac.kr',
    url='https://github.com/Holim0711/NoisyCIFAR-lightning',
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements,
)

