from setuptools import setup 

setup(
  name = 'geGMM',
  version = '0.0.1',
  description = 'Spatial GMM tools for general equilibrium project analysis',
  author = 'Grady Killeen',
  author_email = 'gkilleen@berkeley.edu',
  python_requires='>=3.5',
  packages = ['geGMM'],
  install_requires=['numpy', 'pandas', 'statsmodels', 'patsy']
)

