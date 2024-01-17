from setuptools import setup, find_packages

setup(
    name="pls",
    version="0.0.1",
    packages=find_packages(
        where='.'
    ),
    install_requires=[
        # environment
        "protobuf==3.20.0",
        "gym==0.21.0",
        "torch==1.9.0",
        "stable-baselines3[extra]==1.5.0",
        # shielding
        "problog==2.2.4",
        "pysdd",
        # # experiments -- dask
        # "dask[complete]",
        # "asyncssh",
        # "bokeh",
        # visialization
        "tensorboard==2.7.0",
        "matplotlib==3.4.1",
        "scikit-image>=0.19.2"
    ],
)
