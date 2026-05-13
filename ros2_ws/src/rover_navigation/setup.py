from setuptools import setup, find_packages
import os
from glob import glob

package_name = "rover_navigation"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.py")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Rover Team",
    maintainer_email="rover@example.com",
    description="Nav2 integration and semantic navigator node",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "semantic_navigator = rover_navigation.semantic_navigator:main",
        ],
    },
)
