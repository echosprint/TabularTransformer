from setuptools import setup, find_packages

setup(
    name='tabular-transformer',
    packages=find_packages(),
    version='0.1.0',
    license='MIT',
    description='Transformer model adapted for tabular data',
    long_description_content_type='text/markdown',
    author='Qiao Qian',
    author_email='qiaoqianda@gmail.com',
    url='https://github.com/huoyushequ/TabularTransformer',
    keywords=[
        'artificial intelligence',
        'transformers',
        'attention mechanism',
        'tabular data'
    ],
    install_requires=[
        'requests>=2.31.0',
        'torch>=2.3.0',
        'wandb>=0.17.2',
        'tqdm>=4.66.4',
        'scikit-learn>=1.3.2',
        'pandas>=2.1.0',
        'numpy>=1.26.4',
        'python>=3.10.12'
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.11',
    ],
)
