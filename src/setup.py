from setuptools import setup

setup(
    name="gradient_shielding",
    version="0.0.1",
    install_requires=[
        # environment
        "gym",
        "torch",
        "cherry-rl",
        "stable-baselines3[extra]",
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
        "matplotlib",
    ],  # And any other dependencies required
)
