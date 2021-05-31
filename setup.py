from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='luna-fviz',
    version='0.0.2',
    description='Inspired by [Lucid](https://github.com/tensorflow/lucid), Luna is a Feature Visualization package for Tensorflow2.',
    packages=["luna", "luna.featurevis", "luna.pretrained_models"],
    package_dir={'': 'src'},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Software Development",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Software Development :: Libraries",
        "Topic :: Scientific/Engineering :: Visualization",
    ],
    url="https://github.com/Sparkier/luna",
    author="Alex Bäuerle",
    author_email="alex.baeuerle@uni-ulm.de",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=["tensorflow>=2.4.0",
                      "tensorflow_addons>=0.12.1", "keras"]
)
