from distutils.core import setup

setup(
    # Application name:
    name = "artools",

    # Version number:
    version = "0.1.1b",

    # Application author details:
    author = "Andrew Nadolski",
    author_email = "andrew.nadolski@gmail.com",

    # Packages:
    packages = ['artools'],

    # Include additional files in the package:
    include_package_data = True,

    # Details:
    url = "https://github.com/anadolsk1/artools",

    # License:
    license = "LICENSE.txt",

    # Description:
    description = "Placeholder description. See README?",

    # Dependencies:
    install_requires = [
        "numpy",
        "matplotlib",
        "scipy",
    ],
)
