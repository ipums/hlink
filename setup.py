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
    "hlink.linking.model_exploration",
    "hlink.linking.hh_model_exploration",
    "hlink.linking.reporting",
    "hlink.linking.training",
    "hlink.linking.hh_training",
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
    "ipython==8.3.0",
    "Jinja2==3.1.2",
    "numpy==1.22.3",
    "pandas==1.4.2",
    "pyspark==3.3.0",
    "scikit-learn==1.1.0",
    "toml==0.10.2",
]

dev_requires = [
    "pre-commit",
    "black==22.3.0",
    "flake8==4.0.1",
    "sphinx",
    "recommonmark",
    "pytest==7.1.2",
    "twine",
]

setup(
    name="hlink",
    version="3.2.7",
    packages=find_packages(),
    description="Fast supervised pyspark record linkage software",
    package_data=package_data,
    install_requires=install_requires,
    extras_require={"dev": dev_requires},
    project_urls={"Homepage": "https://github.com/ipums/hlink"},
    entry_points="""
    [console_scripts]
    hlink=hlink.scripts.main:cli
  """,
)
