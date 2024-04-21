from setuptools import setup, find_packages

setup(
    name="hyperit",
    version="v1.0.0",
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
        'jpype1',
    ],
    dependency_links=[
        'https://github.com/EdoardoChidichimo/HyperIT/tarball/master#egg=hyperit-v1.0.0'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',
)
