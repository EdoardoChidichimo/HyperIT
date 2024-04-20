from setuptools import setup, find_packages

setup(
    name="hyperit",
    version="0.1.0",
    author="Edoardo Chidichimo",
    author_email="ec750@cam.ac.uk",
    description="Information-theoretic tools for social neuroscientific endeavours",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url="https://github.com/EdoardoChidichimo/HyperIT",
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'mne',
        'matplotlib',
        'Pillow',
        'tqdm',
        'jpype',
        'git+https://github.com/Imperial-MIND-lab/integrated-info-decomp.git'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',
)