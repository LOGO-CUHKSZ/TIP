from setuptools import setup, find_packages, Extension
from torch.utils import cpp_extension

# Define the C++ extension
persistent_homology_extension = cpp_extension.CppExtension(
    'topolayer/torch_persistent_homology.persistent_homology_cpu',
    ['topolayer/torch_persistent_homology/perisistent_homology_cpu.cpp'],
    extra_link_args=[
        '-Wl,-rpath,' + library_path
        for library_path in cpp_extension.library_paths(cuda=False)]
)

setup(
    name='torch-persistent-homology',
    version='0.1.0',
    author='',  # Add your name
    author_email='',  # Add your email
    description='',  # Add a short description
    long_description='',  # Add a long description
    long_description_content_type='text/markdown',
    url='',  # Add URL of your project if available
    packages=find_packages(),
    ext_modules=[persistent_homology_extension],
    cmdclass={
        'build_ext': cpp_extension.BuildExtension
    },
    install_requires=[
        # 'torch>=1.7.1,<1.8'  # Specify your dependencies
    ],
    classifiers=[
        # Add classifiers that describe your project
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7, <3.9',
)

# How to use
# python setup.py build_ext --inplace