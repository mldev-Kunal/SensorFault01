exit
from setuptools import setup, find_packages
from typing import List
# just update


def get_requirements(file_path: str)->List[str]:
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.strip() for req in requirements if req.strip()]
    return requirements        

setup(
    name="SensorFaultDetection",
    version="0.0.1",
    author="Kunal",
    author_email="tech02042001@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt")
)