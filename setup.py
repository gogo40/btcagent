from setuptools import setup, Extension, find_packages
import importlib
from wheel.bdist_wheel import bdist_wheel
import sys

print(importlib.machinery.EXTENSION_SUFFIXES)
pv = sys.version.split(".")
pvstr = f"{pv[0]}.{pv[1]}"
print(pv, pvstr)


class wheel(bdist_wheel):
    def get_tag(self):
        python, abi, plat = bdist_wheel.get_tag(self)
        print(python, abi, plat)
        return [f"cp310", abi, plat]


setup(
    name="btcagent",
    version="0.0.6",
    packages=find_packages(),
    description="BTC Agent",
    author="Péricles Lopes Machado",
    author_email="pericles@newavesoft.com",
    url=f"https://newavesoft.com",
    include_package_data=True,
    package_data={"btcagent": ["*.pyd", "*.py"]},
    zip_safe=False,
    install_requires=["numpy", "pandas", "numba", "yfinance"],
    cmdclass={"bdist_wheel": wheel},
)
