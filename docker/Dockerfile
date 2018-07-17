FROM nvidia/cuda:8.0-cudnn5-devel-ubuntu16.04
MAINTAINER Benjamin Wild <b.w@fu-berlin.de>

# makes sure deb-src in sources.list is not commented out 
RUN sed -i 's/^# deb-src/deb-src/' /etc/apt/sources.list


RUN apt-get update && apt-get install -y --assume-yes --install-recommends \
        python \
        python-dev \
        python3 \
        python3-dev \
        python3-six \
        python3-tz \
        python3-babel \
        python3-roman \
        python3-docutils \
        python3-markupsafe \
        python3-jinja2 \
        python3-numpy \
        python3-pygments \
        checkinstall \
        git \
        build-essential \
        g++-4.9 \
        g++ \
        cmake \
        wget \
        libopenblas-dev \
        libavformat-dev \
        libavcodec-dev \
        libavdevice-dev \
        libavutil-dev \
        libswscale-dev \
        libavresample-dev \
        libavcodec-extra \
        libav-tools \
        qtbase5-dev \
        pkg-config \
        libbz2-dev \
        && rm -rf /var/lib/apt/lists

RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-4.9 20 && \
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-5 10 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-4.9 20 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-5 10 && \
    update-alternatives --install /usr/bin/cc cc /usr/bin/gcc 30 && \
    update-alternatives --set cc /usr/bin/gcc && \
    update-alternatives --install /usr/bin/c++ c++ /usr/bin/g++ 30 && \
    update-alternatives --set c++ /usr/bin/g++

RUN apt-get update && apt-get build-dep -y --assume-yes --no-install-recommends \
        libopencv-dev \
        libboost-all-dev && \
    rm -rf /var/lib/apt/lists

RUN mkdir -p /tmp/source && \
    cd /tmp/source && \
    git clone --branch 3.1.0 --depth 1 https://github.com/Itseez/opencv.git && \
    mkdir -p /tmp/build/opencv && \
    cd /tmp/build/opencv && \
    cmake -DCMAKE_BUILD_TYPE=RELEASE \
          -DCMAKE_INSTALL_PREFIX=/usr/ \
          -DWITH_CUDA=OFF \
          -DWITH_CUFFT=OFF \
          -DWITH_CUBLAS=OFF \
          -DBUILD_opencv_java=OFF \
          -DBUILD_TESTS=OFF \
          -DBUILD_PERF_TESTS=OFF \
          -DBUILD_EXAMPLES=OFF \
          -DPYTHON_EXECUTABLE=/usr/bin/python3 \
          /tmp/source/opencv && \
    make -j `nproc`

RUN cd /tmp/build/opencv && checkinstall


RUN cd /tmp/source && \
    wget https://sourceforge.net/projects/boost/files/boost/1.61.0/boost_1_61_0.tar.bz2/download?use_mirror=netcologne -O boost.tar.bz2 && \
    tar -xf boost.tar.bz2 && \
    ls && \
    cd boost_1_61_0 && \
    ls && \
    ./bootstrap.sh --prefix=/usr --with-python=/usr/bin/python3

RUN cd /tmp/source/boost_1_61_0 && \
    ./b2 -j `nproc` install

RUN apt-get update && apt-get install -y --assume-yes --install-recommends \
        python3-scipy \
        python3-setuptools \
        python3-pip \
        python3-nose \
        python3-pytest \
        python3-sklearn \
        python3-skimage \
        python3-h5py \
        python3-matplotlib \
        python3-seaborn \
        python3-cairocffi \
        tmux \
        gdb \
        capnproto \
        vim \
        vim-nox \
        libgflags-dev \
        libzmqpp-dev \
        libhdf5-dev \
        libhdf5-cpp-11 \
        hdf5-tools \
        libgoogle-glog-dev \
        jq \
        sshfs \
        zsh && \
    rm -rf /var/lib/apt/lists

RUN pip3 install \
    Theano \
    jupyter \
    xgboost \
    pytest-cov \
    pytest-benchmark \
    pytest-flake8 \
    shyaml \
    more_itertools \
    scikit-image \
    click \
    pandas

RUN pip3 install git+https://github.com/berleon/keras.git@losses#egg=Keras
RUN pip3 install git+https://github.com/BioroboticsLab/diktya.git@c21788bc0fd51f16f920d1ee0f3ae6c88b183ad1#egg=diktya
RUN pip3 install git+https://github.com/BioroboticsLab/bb_binary.git@c724a61d440878d6a44a43e1523c1cc8fb7c7de9#egg=bb-binary
RUN pip3 install git+https://github.com/BioroboticsLab/bb_pipeline.git@fd74cbd8660d30a28f2e8e23fe4c0863b5b531c6#egg=bb-pipeline
RUN pip3 install git+https://github.com/berleon/cfg.git@9484201fe0f80cb39fd26d7193477e400776d785#egg=cfg

RUN git clone https://github.com/berleon/pybeesgrid.git /opt/pybeesgrid && \
    cd /opt/pybeesgrid && export PIP=pip3 && ./build_and_install.sh

RUN ldconfig    # otherwise the cuda libaries are not found

RUN locale-gen en_US.UTF-8  # fix utf-8 encoding
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8
