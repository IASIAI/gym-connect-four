from setuptools import setup, find_packages

setup(name='gym_connect_four',
      version='0.0.2',
      author='Iasi.ai Community',
      author_email='http://iasi.ai',
      packages=find_packages(),
      install_requires=['gym>=0.14',
                        'numpy>=1.17.0',
                        'tensorflow>=2.0.0',
                        'scikit-image>=0.14.5',
                        'keras>=2.3.0',
                        'h5py',
                        'Pillow',
                        'pygame>=1.9.6',
                        'tqdm',
                        'plotly']
      )
