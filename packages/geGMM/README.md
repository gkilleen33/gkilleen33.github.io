# geGMM 

This Python package implements flexible GMM estimation with spatial HAC standard errors (Conley 1999, 2008). It also has tools for assisting with analysis for research on the project "General Equilibrium Effects of Cash Transfers." This builds on the Statsmodels GMM implementation, and most features and documentation of that package remain valid.

## Installation 

1.) Download this folder. 

2.) Navigate to the folder in a terminal with your Python environment and type python `setup.py sdist.` 

3.) Now enter `pip install dist/geGMM-0.0.1.tar.gz` where the number may need to be updated to different versions. If you don't want to update dependencies, instead enter `pip install --no-deps dist/geGMM-0.0.1.tar.gz`

4.) You can now import the package using `import geGMM` in any Python script in this environment. 

