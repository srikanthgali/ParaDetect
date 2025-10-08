import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

__version__ = "0.0.0"
REPO_NAME = "para-detect"
AUTHOR_USER_NAME = "srikanthgali"
SRC_REPO = "para-detect"
AUTHOR_EMAIL = "srikanthgali137@gmail.com"

setuptools.setup(
    name=REPO_NAME,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="ParaDetect - Human vs AI Text Detection Tool",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{SRC_REPO}",
    project_urls={
        "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{SRC_REPO}/issues"
    },
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src")
)