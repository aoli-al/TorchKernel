DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

cd $DIR
rm -rf build
/root/miniconda3/bin/python3 setup.py build_ext -j8
/root/miniconda3/bin/python3 setup.py install
