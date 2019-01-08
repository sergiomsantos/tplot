from setuptools import setup, find_packages
import platform
import tplot

if __name__ == "__main__":
    setup(name='tplot',
        version=tplot.__version__,
        url="https://github.com/sergiomsantos/tplot",
        description='A Python package for creating and displaying matplotlib plots in the console/terminal',
        long_description='A Python package for creating and displaying matplotlib plots in the console/terminal',
        packages=find_packages(),
        install_requires=[
            'numpy',
            'future',
            'matplotlib',
        ],
        include_package_data=True,
        license="Apache License 2.0",
        classifiers=[
            'Development Status :: 4 - Beta',
            'Intended Audience :: Developers',
            'Topic :: Scientific/Engineering :: Physics',
            'License :: OSI Approved :: Apache Software License',
            'Programming Language :: Python :: 2',
            'Programming Language :: Python :: 2.6',
            'Programming Language :: Python :: 2.7',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.2',
            'Programming Language :: Python :: 3.3',
            'Programming Language :: Python :: 3.4',
            'Programming Language :: Python :: 3.5',
            'Programming Language :: Python :: 3.6',
        ],
        keywords='plot terminal matplotlib console',
        python_requires='>=2.6, <4',
        entry_points = {
            'console_scripts': [
                'tplot = tplot:main',                  
            ],              
          },
    )
    