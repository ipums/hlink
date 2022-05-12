# This file is part of the ISRDI's hlink.
# For copyright and licensing information, see the NOTICE and LICENSE files
# in this project's top-level directory, and also on-line at:
#   https://github.com/ipums/hlink

from setuptools import setup, find_packages
import os


packages_with_templates = [
    "hlink.linking",
    "hlink.linking.preprocessing",
    "hlink.linking.matching",
    "hlink.linking.hh_matching",
]

package_data = {"hlink.spark": ["jars/hlink_lib-assembly-1.0.jar"]}
for package in packages_with_templates:
    package_path = package.replace(".", "/")
    template_files = []
    for root, dirs, files in os.walk(f"{package_path}/templates"):
        for file in files:
            template_files.append(
                os.path.relpath(os.path.join(root, file), package_path)
            )
    package_data[package] = template_files

package_data["hlink.linking"].append("table_definitions.csv")

install_requires = [
    "colorama==0.4.4",
    "ipython==7.0.1",
    "Jinja2==3.0.3",
    "numpy==1.19.5",
    "pandas==1.1.5",
    "pyspark==3.2.1",
    "scikit-learn==0.23.1",
    "toml==0.10.2",
]

dev_requires = [
    "pre-commit",
    "black==22.3.0",
    "flake8==4.0.1",
    "sphinx",
    "recommonmark",
    "pytest==7.0.1",
]

setup(
    name="hlink",
    version="3.1.0",
    packages=find_packages(),
    package_data=package_data,
    install_requires=install_requires,
    extras_require={"dev": dev_requires},
    entry_points="""
    [console_scripts]
    hlink=hlink.scripts.main:cli
  """,
)
