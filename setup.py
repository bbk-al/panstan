import setuptools

with open("README.rst", "r") as fh:
  long_description = fh.read()

setuptools.setup(
  name="panstan",
  version="0.0.1",
  author="Adam Light",
  author_email="dev@allight.plus.com",
  description="Using Pandas and Stan to model a Pandemic",
  long_description=long_description,
  long_description_content_type="text/x-rst",
  url="https://github.com/bbk-al/panstan",
  packages=setuptools.find_packages(),
  classifiers=[
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
  ],
  python_requires='>=3.8',
  install_requires=['pystan','numpy','pandas','matplotlib','scipy',
                    'statsmodels','bottleneck'],
  keywords='Stan modelling pandas',
)
