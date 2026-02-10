FROM docker.io/nvidia/cuda@sha256:392c0df7b577ecae17a17f6ba7f2009c217bb4422f8431c053ae9af61a8c148a AS devel

# step1: start

# Install build tools over base image

# Python
RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        python3 && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        gcc-13-offload-nvptx \
        zlib1g \
        zlib1g-dev && \
    rm -rf /var/lib/apt/lists/*

# CMake version 3.31.4
RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        make \
        wget && \
    rm -rf /var/lib/apt/lists/*
RUN mkdir -p /var/tmp && wget -q -nc --no-check-certificate -P /var/tmp https://github.com/Kitware/CMake/releases/download/v3.31.4/cmake-3.31.4-linux-x86_64.sh && \
    mkdir -p /usr/local && \
    /bin/sh /var/tmp/cmake-3.31.4-linux-x86_64.sh --prefix=/usr/local --skip-license && \
    rm -rf /var/tmp/cmake-3.31.4-linux-x86_64.sh
ENV PATH=/usr/local/bin:$PATH

# Git, Pkgconf

RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        git \
        pkgconf && \
    rm -rf /var/lib/apt/lists/*

# step2: start

# Install network stack packages and OpenMPI

# Mellanox OFED version 24.04-0.7.0.0
RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ca-certificates \
        gnupg \
        wget && \
    rm -rf /var/lib/apt/lists/*
RUN wget -qO - https://www.mellanox.com/downloads/ofed/RPM-GPG-KEY-Mellanox | apt-key add - && \
    mkdir -p /etc/apt/sources.list.d && wget -q -nc --no-check-certificate -P /etc/apt/sources.list.d https://linux.mellanox.com/public/repo/mlnx_ofed/24.04-0.7.0.0/ubuntu24.04/mellanox_mlnx_ofed.list && \
    apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ibverbs-providers \
        ibverbs-utils \
        libibmad-dev \
        libibmad5 \
        libibumad-dev \
        libibumad3 \
        libibverbs-dev \
        libibverbs1 \
        librdmacm-dev \
        librdmacm1 && \
    rm -rf /var/lib/apt/lists/*

# KNEM version 1.1.4
RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ca-certificates \
        git && \
    rm -rf /var/lib/apt/lists/*
RUN mkdir -p /var/tmp && cd /var/tmp && git clone --depth=1 --branch knem-1.1.4 https://gitlab.inria.fr/knem/knem.git knem && cd - && \
    mkdir -p /opt/knem && \
    cd /var/tmp/knem && \
    mkdir -p /opt/knem/include && \
    cp common/*.h /opt/knem/include && \
    rm -rf /var/tmp/knem
ENV PATH=/opt/knem/include:$PATH

# XPMEM branch master
RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        autoconf \
        automake \
        ca-certificates \
        file \
        git \
        libtool \
        make && \
    rm -rf /var/lib/apt/lists/*
RUN mkdir -p /var/tmp && cd /var/tmp && git clone --depth=1 --branch master https://github.com/hjelmn/xpmem.git xpmem && cd - && \
    cd /var/tmp/xpmem && \
    autoreconf --install && \
    cd /var/tmp/xpmem &&   ./configure --prefix=/opt/xpmem --disable-kernel-module && \
    make -j$(nproc) && \
    make -j$(nproc) install && \
    rm -rf /var/tmp/xpmem
ENV PATH=/opt/xpmem/include:$PATH \
    LD_LIBRARY_PATH=/opt/xpmem/lib:$LD_LIBRARY_PATH 

# GDRCOPY version 2.4.4
RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        make \
        wget && \
    rm -rf /var/lib/apt/lists/*
RUN mkdir -p /var/tmp && wget -q -nc --no-check-certificate -P /var/tmp https://github.com/NVIDIA/gdrcopy/archive/v2.4.4.tar.gz && \
    mkdir -p /var/tmp && tar -x -f /var/tmp/v2.4.4.tar.gz -C /var/tmp -z && \
    cd /var/tmp/gdrcopy-2.4.4 && \
    mkdir -p /opt/gdrcopy/include /opt/gdrcopy/lib && \
    make prefix=/opt/gdrcopy lib lib_install && \
    rm -rf /var/tmp/gdrcopy-2.4.4 /var/tmp/v2.4.4.tar.gz
ENV PATH=/opt/gdrcopy/include:$PATH \
    LD_LIBRARY_PATH=/opt/gdrcopy/lib:$LD_LIBRARY_PATH 

# UCX https://github.com/openucx/ucx.git v1.17.0
RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        autoconf \
        automake \
        binutils-dev \
        ca-certificates \
        file \
        git \
        libnuma-dev \
        libtool \
        make \
        wget && \
    rm -rf /var/lib/apt/lists/*
RUN mkdir -p /var/tmp && cd /var/tmp && git clone --depth=1 --branch v1.17.0 https://github.com/openucx/ucx.git ucx && cd - && \
    cd /var/tmp/ucx && \
    ./autogen.sh && \
    cd /var/tmp/ucx &&   ./configure --prefix=/opt/ucx --disable-assertions --disable-debug --disable-doxygen-doc --disable-logging --disable-params-check --enable-mt --enable-optimizations --with-cuda=/usr/local/cuda --with-gdrcopy=/opt/gdrcopy --with-knem=/opt/knem --with-rdmacm --with-verbs --with-xpmem=/opt/xpmem && \
    make -j$(nproc) && \
    make -j$(nproc) install && \
    rm -rf /var/tmp/ucx
ENV PATH=/opt/ucx/include:$PATH \
    LD_LIBRARY_PATH=/opt/ucx/lib:$LD_LIBRARY_PATH \
    PATH=/opt/ucx/bin:$PATH

# PMIX version 3.1.5
RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        file \
        hwloc \
        libevent-dev \
        make \
        tar \
        wget && \
    rm -rf /var/lib/apt/lists/*
RUN mkdir -p /var/tmp && wget -q -nc --no-check-certificate -P /var/tmp https://github.com/openpmix/openpmix/releases/download/v3.1.5/pmix-3.1.5.tar.gz && \
    mkdir -p /var/tmp && tar -x -f /var/tmp/pmix-3.1.5.tar.gz -C /var/tmp -z && \
    cd /var/tmp/pmix-3.1.5 &&   ./configure --prefix=/opt/pmix && \
    make -j$(nproc) && \
    make -j$(nproc) install && \
    rm -rf /var/tmp/pmix-3.1.5 /var/tmp/pmix-3.1.5.tar.gz
ENV PATH=/opt/pmix/include:$PATH \
    LD_LIBRARY_PATH=/opt/pmix/lib:$LD_LIBRARY_PATH \
    PATH=/opt/pmix/bin:$PATH

# OpenMPI version 4.1.6
RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        bzip2 \
        file \
        hwloc \
        libnuma-dev \
        make \
        openssh-client \
        perl \
        tar \
        wget && \
    rm -rf /var/lib/apt/lists/*
RUN mkdir -p /var/tmp && wget -q -nc --no-check-certificate -P /var/tmp https://www.open-mpi.org/software/ompi/v4.1/downloads/openmpi-4.1.6.tar.bz2 && \
    mkdir -p /var/tmp && tar -x -f /var/tmp/openmpi-4.1.6.tar.bz2 -C /var/tmp -j && \
    cd /var/tmp/openmpi-4.1.6 &&   ./configure --prefix=/opt/openmpi --disable-getpwuid --enable-orterun-prefix-by-default --with-cuda --with-pmix=/opt/pmix --with-ucx=/opt/ucx --with-verbs && \
    make -j$(nproc) && \
    make -j$(nproc) install && \
    rm -rf /var/tmp/openmpi-4.1.6 /var/tmp/openmpi-4.1.6.tar.bz2
ENV LD_LIBRARY_PATH=/opt/openmpi/lib:$LD_LIBRARY_PATH \
    PATH=/opt/openmpi/bin:$PATH

# step5: start
# The installer requires curl (and certificates) to download the release archive
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates

# Download the latest installer
ADD https://astral.sh/uv/install.sh /uv-installer.sh

# Run the installer then remove it
RUN sh /uv-installer.sh && rm /uv-installer.sh

# Ensure the installed binary is on the `PATH`
ENV PATH="/root/.local/bin/:$PATH"

WORKDIR /app
COPY README.md /app/README.md
COPY uv.lock /app/uv.lock
COPY pyproject.toml /app/pyproject.toml
COPY src /app/src

RUN uv venv --no-managed-python .venv
# Enable venv
ENV PATH="/app/.venv/bin:$PATH"
RUN uv sync
RUN python3 -m cupyx.tools.install_library --cuda 12.x --library cutensor --prefix "/app/.venv/.cupy/cuda_lib"
RUN python3 -m cupyx.tools.install_library --cuda 12.x --library nccl     --prefix "/app/.venv/.cupy/cuda_lib"
RUN python3 -m cupyx.tools.install_library --cuda 12.x --library cudnn    --prefix "/app/.venv/.cupy/cuda_lib"
RUN uv pip install -e .


FROM docker.io/nvidia/cuda@sha256:392c0df7b577ecae17a17f6ba7f2009c217bb4422f8431c053ae9af61a8c148a

# Python
RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        python3 && \
    rm -rf /var/lib/apt/lists/*

# Mellanox OFED version 24.04-0.7.0.0
RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ca-certificates \
        gnupg \
        wget && \
    rm -rf /var/lib/apt/lists/*
RUN wget -qO - https://www.mellanox.com/downloads/ofed/RPM-GPG-KEY-Mellanox | apt-key add - && \
    mkdir -p /etc/apt/sources.list.d && wget -q -nc --no-check-certificate -P /etc/apt/sources.list.d https://linux.mellanox.com/public/repo/mlnx_ofed/24.04-0.7.0.0/ubuntu24.04/mellanox_mlnx_ofed.list && \
    apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ibverbs-providers \
        ibverbs-utils \
        libibmad-dev \
        libibmad5 \
        libibumad-dev \
        libibumad3 \
        libibverbs-dev \
        libibverbs1 \
        librdmacm-dev \
        librdmacm1 && \
    rm -rf /var/lib/apt/lists/*

# KNEM
COPY --from=devel /opt/knem /opt/knem
ENV PATH=/opt/knem/include:$PATH

# XPMEM
COPY --from=devel /opt/xpmem /opt/xpmem
ENV PATH=/opt/xpmem/include:$PATH \
    LD_LIBRARY_PATH=/opt/xpmem/lib:$LD_LIBRARY_PATH 

# GDRCOPY
COPY --from=devel /opt/gdrcopy /opt/gdrcopy
ENV PATH=/opt/gdrcopy/include:$PATH \
    LD_LIBRARY_PATH=/opt/gdrcopy/lib:$LD_LIBRARY_PATH 

# UCX
RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        libbinutils && \
    rm -rf /var/lib/apt/lists/*
COPY --from=devel /opt/ucx /opt/ucx
ENV PATH=/opt/ucx/include:$PATH \
    LD_LIBRARY_PATH=/opt/ucx/lib:$LD_LIBRARY_PATH \
    PATH=/opt/ucx/bin:$PATH

# PMIX
RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        libevent-2.* \
        libevent-pthreads-2.* && \
    rm -rf /var/lib/apt/lists/*
COPY --from=devel /opt/pmix /opt/pmix
ENV PATH=/opt/pmix/include:$PATH \
    LD_LIBRARY_PATH=/opt/pmix/lib:$LD_LIBRARY_PATH \
    PATH=/opt/pmix/bin:$PATH

# OpenMPI
RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        hwloc \
        openssh-client && \
    rm -rf /var/lib/apt/lists/*
COPY --from=devel /opt/openmpi /opt/openmpi
ENV LD_LIBRARY_PATH=/opt/openmpi/lib:$LD_LIBRARY_PATH \
    PATH=/opt/openmpi/bin:$PATH


# Libraries missing from CUDA runtime image

RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        gcc-13-offload-nvptx \
        libcurl4 \
        libgomp1 \
        libnuma1 \
        zlib1g \
        make \
        zlib1g-dev && \
    rm -rf /var/lib/apt/lists/*


# step5: start
# The installer requires curl (and certificates) to download the release archive
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates

COPY --from=devel /app /app

# Enable venv
ENV PATH="/app/.venv/bin:$PATH"


