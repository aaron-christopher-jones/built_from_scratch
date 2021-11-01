import setuptools

with open("README.md", "r") as ld:
    long_description = ld.read()

with open("requirements.txt", "r") as rp:
    required_packages = rp.read().splitlines()

setuptools.setup(
    name="scratch",
    version="0.1",
    author="Aaron Jones",
    author_email="jonesstats3d@gmail.com",
    description="Python package containing statistical learning algorithms written from scratch.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/aaron-christopher-jones/scratch",
    install_requires=required_packages,
    python_requires='>=3.9.6',
)
