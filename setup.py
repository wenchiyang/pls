from setuptools import setup

setup(
    name="soft_shielding",
    version="0.0.1",
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
