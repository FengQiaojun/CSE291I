# nvcc = /usr/local/cuda-8.0/bin/nvcc
# cudalib = /usr/local/cuda-8.0/lib64/
# tensorflow = /usr/local/lib/python2.7/dist-packages/tensorflow/include

nvcc = /usr/local/cuda-9.0/bin/nvcc
cudalib = /usr/local/cuda-9.0/lib64/
tensorflow = /home/qiaojun/anaconda3/lib/python3.6/site-packages/tensorflow/include
tensorflow_lib = /home/qiaojun/anaconda3/lib/python3.6/site-packages/tensorflow

all: tf_emddistance_so.so 
.PHONY : all

tf_emddistance_so.so:  tf_emddistance.cpp
	g++ -std=c++11 tf_emddistance.cpp  -o tf_emddistance_so.so -shared -fPIC -I $(tensorflow) -I $(tensorflow)/external/nsync/public -ltensorflow_framework -L $(tensorflow_lib) -O2 -lcudart -L $(cudalib) -O2 -D_GLIBCXX_USE_CXX11_ABI=0

