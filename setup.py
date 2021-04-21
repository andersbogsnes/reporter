from setuptools import setup, find_packages

setup(
    name='reporter',
    version=0.1,
    scripts=['cli.py'],
    author='Anders Bogsnes',
    author_email='andersbogsnes@gmail.com',
    description='A documentation generator for machine-learning models',
    license='MIT',
    install_requires=['bokeh>=0.12.15',
                      'scikit-learn>=0.19.1',
                      'pandas>=0.22.0',
                      'Jinja2 == 2.11.3'],
    python_requires='>=3.6',
    packages=find_packages('reporter'),
    package_dir={'': 'reporter'},
    package_data={
        'reporter': ['template/*.html']
    }
)