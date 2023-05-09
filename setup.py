import setuptools

setuptools.setup(
    name = "rv_curve_jura",
    version = "0.1",
    author = "nicolas kaufmann",
    description = "test module",
    author_email= "nicolas.kaufmann@unibe.ch",
    packages=setuptools.find_packages(),
    install_requires = ["numpy", "matplotlib"],
    classifiers = ["Programming Language :: Python :: 3",
                    "Operating System :: OS Independent"
                    ],
)