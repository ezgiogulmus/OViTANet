from setuptools import setup, find_packages


setup(
    name='surv',
    version='0.1.0',
    description='Ovarian Cancer Survival',
    url='https://github.com/ezgiogulmus/OViTANet',
    author='FEO',
    author_email='',
    license='GPLv3',
    packages=find_packages(exclude=['data_processing', 'results', 'datasets_csv', "splits"]),
    install_requires=[
        "torch>=2.3.0",
        "torchvision",
        "numpy==1.23.4", 
        "pandas==1.4.3", 
        "openpyxl",
        "h5py",
        "scikit-learn", 
        "scikit-survival",
        "tensorboardx",
        "tensorboard",
        "wandb",
        "mmsurv @ git+https://github.com/ezgiogulmus/MM-SurvModels",
        "wsisurv @ git+https://github.com/ezgiogulmus/WSI_Surv"
    ],

    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: MIT",
    ]
)