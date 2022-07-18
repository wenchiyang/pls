from setuptools import setup, find_packages

setup(
    name="pls",
    version="0.0.1",
    packages=find_packages(
        where='.'
    ),
    install_requires=[
        # environment
        "gym",
        "torch",
        "torchvision",
        "cherry-rl",
        "stable-baselines3[extra]",
        "networkx",
        # shielding
        "problog",
        "pysdd",
        # experiments -- dask
        "dask[complete]",
        "asyncssh",
        "bokeh",
        # visialization
        "pyvirtualdisplay",
        "tensorboard",
        "altair",
        "altair_saver",
        "selenium",
        "matplotlib",
        "tqdm",
        "scikit-image"
    ],  # And any other dependencies required
)
