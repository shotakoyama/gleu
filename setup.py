import setuptools


setuptools.setup(
        name = 'gleu',
        version = '1.0.0',
        packages = setuptools.find_packages(),
        install_requires = ['numpy'],
        entry_points = {'console_scripts':['gleu = gleu.main:main']})

