#! /bin/bash
rm -rf build/
python setup.py bdist_wheel
pip install --force dist/ter_spmm-1.0-cp312-cp312-linux_x86_64.whl
