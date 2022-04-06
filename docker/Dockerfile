##############################################################################
# Build an image from the dockerfile:
# docker build -t carlhiggs/global_scorecards .
#
# To run bash in this container:
# FROM BASH:
# docker run --rm -it --shm-size=2g --net=host -v "$PWD":/home/ghsi/work carlhiggs/global_scorecards /bin/bash
# FROM CMD:
# docker run --rm -it --shm-size=2g --net=host -v "%cd%":/home/ghsi/work carlhiggs/global_scorecards /bin/bash

# =============================================================================
# Build environment
FROM continuumio/miniconda3:4.10.3-alpine as build

USER $NB_UID
COPY environment.yml .

# Install conda pack
RUN conda config --set show_channel_urls true && \
	conda config --set channel_priority strict && \
    conda config --prepend channels conda-forge && \
    conda update --yes -n base conda && \
    conda install -c conda-forge conda-pack && \
    conda env create -n env -f environment.yml && \
    conda pack -n env -o env.tar.gz && \
    conda list -n env --explicit > spec-file.txt

# =============================================================================
# Runtime environment
FROM debian:bullseye-slim as runtime

# Install the environment pack
COPY --from=build /env.tar.gz .
RUN mkdir /env && \
    tar xvzf /env.tar.gz -C /env && \
    rm -rv /env.tar.gz && \
    /env/bin/python /env/bin/conda-unpack && \
    useradd -u 8877 ghsi

USER ghsi
ENV PATH="/env/bin/:$PATH"

WORKDIR /home/ghsi/work
