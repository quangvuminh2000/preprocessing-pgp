from setuptools import setup

# ? setup() does all the work
setup(
    include_dirs=['preprocessing_pgp/data'],
    include_package_data=True,
    package_data={
        "preprocessing_pgp.data": ["*"],
    }
)
