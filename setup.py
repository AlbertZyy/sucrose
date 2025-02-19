import setuptools


with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
    name="sucrose",
    version="0.1.0",
    author="AlbertZyy",
    author_email="",
    description="PyTorch Experiment Project Manager",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AlbertZyy/sucrose",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python:: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
