import setuptools


with open('README.md') as f:
    long_discription = f.read()


setuptools.setup(
        name = 'gleu',
        version = '1.0.0',
        author = 'Shota Koyama',
        author_email = 'koyamashota0@gmail.com',
        description = 'GLEU: evaluation metric for grammatical error correction',
        long_discription = long_discription,
        long_discription_content_type = 'text/markdown',
        url = 'https://github.com/shotakoyama/gleu',
        classifiers = [
            'Programming Language :: Python :: 3.11',
            'License :: OSI Approved :: MIT License',
            'Operating System :: OS Independent'],
        packages = setuptools.find_packages(),
        install_requires = ['numpy', 'prettytable'],
        entry_points = {'console_scripts':['gleu = gleu.main:main']})

