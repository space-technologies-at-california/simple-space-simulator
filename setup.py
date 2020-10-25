import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="simple-space-simulator-michaelequi",
    version="0.0.1",
    author="Michael Equi",
    author_email="michaelequi@berkeley.edu",
    description="Simple Space Simulator or S-Cubed is a pure python LEO cubesat simulator",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/space-technologies-at-california/simple-space-simulator",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)