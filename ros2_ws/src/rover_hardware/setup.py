from setuptools import setup, find_packages

package_name = "rover_hardware"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Rover Team",
    maintainer_email="rover@example.com",
    description="ESP32 serial bridge and hardware interface nodes",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "esp32_bridge_node = rover_hardware.esp32_bridge_node:main",
        ],
    },
)
