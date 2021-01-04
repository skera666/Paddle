
cmake .. -DPY_VERSION=2.7 \
    -DWITH_GPU=ON \
    -DWITH_TESTING=ON \
    -DON_INFER=ON \
    -DCMAKE_BUILD_TYPE=Release
