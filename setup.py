import os
import subprocess
import sys
from pathlib import Path
import platform

from setuptools import Extension, find_packages, setup, Command
from setuptools.command.build_ext import build_ext
from setuptools.command.install import install
from setuptools.command.develop import develop

class InstallWithBaselines(install):
    description = 'install baselines'
    user_options = install.user_options

    def run(self):
        install.run(self)
        # try:
        #     subprocess.check_call(['bash', 'install_baselines.sh'])
        # except subprocess.CalledProcessError as e:
        #     print(f"Error running bash command: {e}")

class DevelopWithBaselines(develop):
    """Override the develop command to run the baselines script."""
    def run(self):
        develop.run(self)
        # try:
        #     subprocess.check_call(['bash', 'install_baselines.sh'])
        # except subprocess.CalledProcessError as e:
        #     print(f"Error running bash command: {e}")

setup(
    name='sailor',
    version='0.1.0',
    packages=find_packages(),
    description='A framework for elastic, fault-tolerant training',
    cmdclass = {
        "install": InstallWithBaselines,
        "develop": DevelopWithBaselines
    },
    author=''
)