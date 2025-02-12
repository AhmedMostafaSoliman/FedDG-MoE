from setuptools import setup, find_packages

setup(
    name="feddg_moe",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'torch',
        'torchvision',
        'timm',
        'tensorboard',
        'tqdm'
    ]
)