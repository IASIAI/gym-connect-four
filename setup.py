from setuptools import setup, find_packages

setup(name='gym_connect_four',
      version='0.0.1',
      packages=find_packages(),
      install_requires=['gym>=0.14',
                        'numpy>=1.17.0',
                        'tensorflow>=1.10.0',
                        'scikit-image==0.14.5',
                        'keras',
                        'h5py',
                        'Pillow',
                        'pygame',
                        'tqdm']
      )
