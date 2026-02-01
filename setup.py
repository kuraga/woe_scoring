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
    download_url="https://github.com/kiraplenkin/woe_scoring/archive/refs/tags/v1.1.0.tar.gz",
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
        "numpy>=1.19.5",
        "pandas>=1.2.2",
        "scikit-learn>=0.24.1",
        "statsmodels>=0.12.2",
        "scipy>=1.6.1",
        "lxml>=4.8.0",
        "joblib>=1.1.0",
        "xlsxwriter>=3.0.0"
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
