# syntax=docker/dockerfile:experimental
# python:3.7-slim-buster is recommended for Python base image.
# See this https://pythonspeed.com/articles/base-image-python-docker-images/
FROM python:3.8-slim-buster

# Set envs and locale
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ Asia/Tokyo

# Install system commands
# See https://stackoverflow.com/a/47143497 for the meaning of `set -ex`.
RUN set -ex \
  && apt-get update \
  && apt-get install -y g++ git libsnappy-dev openssh-client pigz wget libblas-dev liblapack-dev gnupg software-properties-common make

# Install clang 8
RUN wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | apt-key add - \
  && add-apt-repository "deb http://apt.llvm.org/buster/ llvm-toolchain-buster-8 main" \
  && apt-get update \
  && apt-get -y install clang-8 lldb-8 lld-8 gfortran

# Set env option
ENV LLVM_CONFIG=/usr/lib/llvm-8/bin/llvm-config

# See https://stackoverflow.com/a/47652542 .
RUN mkdir -p -m 0600 ~/.ssh \
  && ssh-keyscan github.com >> ~/.ssh/known_hosts

# Install pipenv
RUN pip3 --no-cache-dir install pipenv

ENV DEBIAN_FRONTEND=dialog
