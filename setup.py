from distutils.core import setup

setup(
    # Application name:
    name = "artools",

    # Version number:
    version = "0.1.3",

    # Application author details:
    author = "Andrew Nadolski",
    author_email = "andrew.nadolski@gmail.com",

    # Packages:
    packages = ['artools'],

    # Include additional files in the package:
    include_package_data = True,

    # Details:
    url = "https://github.com/anadolsk1/artools",
    download_url = "https://github.com/anadolsk1/artools/archive/v0.1.3-alpha.tar.gz",
    # License:
    license = "LICENSE.txt",

    # Description:
    description = "A simple anti-reflection coating simulator.",

    # Dependencies:
    install_requires = [
        "numpy",
        "matplotlib",
        "scipy",
    ],
)
