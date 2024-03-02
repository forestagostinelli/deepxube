from setuptools import setup

setup(
    name='deepxube',
    version='0.1.0',
    description='Solving classical planning problems in an explainable manner with deep learning, '
                'reinforcement learning, heuristic search, and logic',
    url='https://github.com/forestagostinelli/DeepXube/',
    author='Forest Agostinelli',
    author_email='foresta@cse.sc.edu',
    license='MIT',
    packages=['deepxube'],
    install_requires=['torch>=2.0',
                      'numpy', 'matplotlib'],

    # classifiers=[
    #    'Development Status :: 1 - Planning',
    #    'Intended Audience :: Science/Research',
    #    'License :: OSI Approved :: BSD License',
    #    'Programming Language :: Python :: 3.10',
    # ],
)
