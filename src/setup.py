from setuptools import setup

setup(
    name="gradient_shielding",
    version="0.0.1",
    install_requires=[
        "gym",
        "torch",
        "cherry-rl",
        "altair",
        "altair_saver",
        "matplotlib",
        "problog",
        "pysdd",
        "dask[distributed]",
        "asyncssh",
        "bokeh",
        "stable-baselines3[extra]"
    ],  # And any other dependencies required
)
