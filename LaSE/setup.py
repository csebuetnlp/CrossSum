from setuptools import setup, find_packages

with open("README.md", mode="r", encoding="utf-8") as readme_file:
    desc = readme_file.read()



setup(
    name="LaSE",
    description="LaSE scoring module",
    long_description=desc,
    long_description_content_type="text/markdown",
    license="Apache License 2.0",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.6",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ],
)
