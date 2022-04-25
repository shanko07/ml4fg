## How to build the cython

[Cython Tutorial](https://cython.readthedocs.io/en/latest/src/tutorial/cython_tutorial.html)

Basically just need to run
`python setup.py build_ext --inplace`
which will generate the necessary C file and .so on linux or .dll on Windows
