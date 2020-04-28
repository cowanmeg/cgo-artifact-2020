FROM ubuntu:18.04

RUN apt-get update && apt-get install -y \
  clang-7 \
  cmake \
  curl \
  g++ \
  g++-multilib \
  gcc-multilib \
  git \
  llvm-7 \
  llvm-7-dev \
  python3-pip \
  vim \
  wget

RUN pip3 --no-cache-dir install \
  decorator \
  gast==0.2.2 \
  matplotlib \
  numpy==1.16.4 \
  Pillow==6.2.1 \
  psutil \
  requests \
  scipy==1.3.0 \
  seaborn==0.9.0 \
  tensorflow==2.0.0b1 \
  tornado

# Download and install racket
ENV RACKET_VERSION 7.5
ENV RACKET_INSTALLER_URL http://mirror.racket-lang.org/installers/$RACKET_VERSION/racket-$RACKET_VERSION-x86_64-linux.sh
RUN wget --output-document=racket-install.sh -q ${RACKET_INSTALLER_URL} && \
    echo "yes\n1\n" | sh racket-install.sh --create-dir --unix-style --dest /usr/ && \
    rm racket-install.sh

# Install rosette
RUN raco setup
# note: ignore error rosette documentation won't install
RUN raco pkg install --batch --deps search-auto rosette; exit 0 

# Download and install boolector
RUN git clone https://github.com/Boolector/boolector.git
WORKDIR /boolector
RUN git checkout 76aafdf665e89e7a77681fe48ac5aaa5ebf95ebe
RUN ./contrib/setup-lingeling.sh
RUN ./contrib/setup-btor2tools.sh
RUN ./configure.sh  && cd build && make -j2
RUN ln -s /boolector/build/bin/boolector /root/.racket/7.5/pkgs/rosette/bin/boolector

# Setup environment variables - assumes the artifact directory is mounted at /artifact
ENV ARTIFACT_HOME=/artifact
ENV TVM_HOME=/artifact/tvm
ENV PYTHONPATH=/artifact/tvm/python:/artifact/tvm/topi/python:/artifact/tvm/Models
ENV PORT=9090
WORKDIR /
