import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name = "rv_curve_jura",
    version = "0.1",
    author = "nicolas kaufmann",
    description = "test module",
    author_email= "nicolas.kaufmann@unibe.ch",
    packages=setuptools.find_packages(),
    long_description = long_description,
    long_description_content_tyype="text/markdown",
    url = "https://github.com/NIcolas-Kaufmann/Publish_test_JURA",
    install_requires = ["numpy", "matplotlib"],
    classifiers = ["Programming Language :: Python :: 3",
                    "Operating System :: OS Independent"
                    ],
)