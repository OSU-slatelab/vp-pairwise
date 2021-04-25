from setuptools import setup, find_packages
setup(
    name='vp-pairwise',

    version='0.0.1',
    description='Pairwise training for class-imbalanced data',

    # Author details
    author='Vishal Sunder',
    
    # Choose your license
    license='MIT',

    packages=find_packages(),
    
    python_requires='>=3',
    install_requires=['torch==1.4.0', 'scikit-learn', 'numpy', 'faiss'],

)
