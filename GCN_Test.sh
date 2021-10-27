#! /bin/bash

check_build_floder(){
    build_floder=`ls -a | grep "build"`
    if [[ -z "$build_floder" ]]; then
        mkdir build
    fi
}

check_build_floder
cd build
rm -rf ./*
cmake ../kernel
make
cp ./kernel.cpython* ../dl_code_python/
cd ../dl_code_python
python3 GCN_pubmed.py