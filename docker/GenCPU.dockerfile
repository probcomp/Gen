FROM ubuntu:20.04

LABEL maintainer="MIT Probabilistic Computing Project"

# Find current Julia version on https://julialang.org/downloads/
ARG JULIA_VERSION_SHORT="1.5"
ARG JULIA_VERSION_FULL="${JULIA_VERSION_SHORT}.1"
ENV JULIA_INSTALLATION_PATH=/opt/julia

ENV DEBIAN_FRONTEND=noninteractive
ENV JULIA_INSTALLATION_PATH=/opt/julia

RUN apt-get update -qq \
    && apt-get install -qq -y --no-install-recommends\
        build-essential \
        ca-certificates \
        curl \
        ffmpeg \
        git \
        graphviz \
        hdf5-tools \
        python3-dev \
        python3-pip \
        python3-tk \
        rsync \
        software-properties-common \
        wget \
        zlib1g-dev \
    && rm -rf /var/lib/apt/lists/* && \
    ln -s /usr/bin/python3 /usr/bin/python

RUN wget https://julialang-s3.julialang.org/bin/linux/x64/${JULIA_VERSION_SHORT}/julia-${JULIA_VERSION_FULL}-linux-x86_64.tar.gz && \
    tar zxf julia-${JULIA_VERSION_FULL}-linux-x86_64.tar.gz && \
    mkdir -p "${JULIA_INSTALLATION_PATH}" && \
    mv julia-${JULIA_VERSION_FULL} "${JULIA_INSTALLATION_PATH}/" && \
    ln -fs "${JULIA_INSTALLATION_PATH}/julia-${JULIA_VERSION_FULL}/bin/julia" /usr/local/bin/ && \
    rm julia-${JULIA_VERSION_FULL}-linux-x86_64.tar.gz && \
    julia -e 'import Pkg; Pkg.add("IJulia")'

RUN julia -e 'import Pkg; Pkg.add(["Gen"])'
