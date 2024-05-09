from setuptools import setup, find_packages


setup(
    name='imfun',
    version='0.1',
    license='MIT',
    author="Marlon Rodrigues Garcia",
    author_email='marlonrg@gmail.com',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    url='https://github.com/MarlonGarcia/imfun',
    keywords='image processing functions',
    install_requires=[
          'numpy',
          'cv2',
          'os',
          'matplotlib',
          'scipy',
          'winsound',
          'time',
          'ctypes',
          'pynput',
          'random',
      ],

)
