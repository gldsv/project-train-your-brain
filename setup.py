from setuptools import find_packages
from setuptools import setup

with open("requirements.txt") as f:
    content = f.readlines()
requirements = [x.strip() for x in content if "git+" not in x]

setup(name='train_your_brain',
      version="0.0.1",
      description="Train your Brain project",
      install_requires=requirements,
      packages=find_packages()
)
