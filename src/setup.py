from setuptools import setup

setup(
    name="gradient_shielding",
    version="0.0.1",
    install_requires=[
        "gym",
        "torch",
        "cherry-rl",
        "altair",
        "matplotlib",
        "problog",
        "dask[distributed]",
        "asyncssh",
    ],  # And any other dependencies required
)
