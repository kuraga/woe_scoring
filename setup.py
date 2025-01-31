from setuptools import setup, find_packages
from woe_scoring import __version__

DISTNAME = "woe_scoring"
DESCRIPTION = "Weight Of Evidence Transformer and LogisticRegression model with scikit-learn API"

with open("README.md", encoding='utf-8') as f:
    LONG_DESCRIPTION = f.read()

MAINTAINER = "Stroganov Kirill"
MAINTAINER_EMAIL = "kiraplenkin@gmail.com"
URL = "https://github.com/kiraplenkin"
DOWNLOAD_URL = "https://pypi.org/project/woe-scoring/#files"
LICENSE = "MIT"

setup(
    name=DISTNAME,
    version=__version__,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    author=MAINTAINER,
    author_email=MAINTAINER_EMAIL,
    url=URL,
    download_url="https://github.com/kiraplenkin/woe_scoring/archive/refs/tags/v0.4.0.tar.gz",
    license=LICENSE,
    packages=find_packages(),
    include_package_data=True,
    keywords=[
        "WOE",
        "Weight Of Evidence",
        "Monotone Weight Of Evidence Transformation",
        "Scorecard",
        "LogisticRegression"
    ],
    install_requires=[
        "numpy>=1.17.0",
        "pandas>=1.0.0",
        "scikit-learn>=0.23.0",
        "statsmodels>=0.11.0",
        "scipy>=1.4.0",
        "lxml>=4.5.0"
    ],
    python_requires='>=3.7',
    zip_safe=False,
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering'
    ]
)
