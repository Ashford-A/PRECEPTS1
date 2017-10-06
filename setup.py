from distutils.core import setup

setup(name='HetMan',
      version='0.1',
      description='Mutation Heterogeneity Manifold',
      author='Michal Grzadkowski',
      author_email='grzadkow@ohsu.edu',
      package_dir={'HetMan': ''},
      packages=['HetMan',
                'HetMan/predict', 'HetMan/features', 'HetMan/experiments'],
     )

