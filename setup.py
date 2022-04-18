from setuptools import setup

setup(
    name="soft_shielding",
    version="0.0.1",
    install_requires=[
        # environment
        "gym",
        "torch",
        "cherry-rl",
        "stable-baselines3[extra]",
        "networkx",
        # shielding
        "problog",
        "pysdd",
        # experiments -- dask
        "dask[distributed]",
        "asyncssh",
        "bokeh",
        # visialization
        "tensorboard",
        "altair",
        "altair_saver",
        "selenium",
        "matplotlib",
    ],  # And any other dependencies required
)
