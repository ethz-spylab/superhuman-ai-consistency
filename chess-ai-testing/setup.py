import setuptools

setuptools.setup(
    name="rl_testing",
    version="0.1dev",
    description="",
    # long_description=open("README.md").read(),
    url="https://github.com/luk-s/rl-testing-experiments",
    install_requires=[
        "asyncssh==2.12.0",
        "chess==1.9.4",
        "imgkit==1.2.2",
        "matplotlib==3.6.1",
        "netwulf==0.1.5",
        "networkx==2.8.8",
        "numpy==1.23.4",
        "pandas==1.5.1",
        "python-chess==1.999",
        "scipy==1.9.3",
        "wandb==0.13.10",
    ],
    author="Lukas Fluri",
    author_email="lukas.fluri@protonmail.com",
    license="MIT",
    packages=setuptools.find_packages(),
    zip_safe=True,
    entry_points={},
)
