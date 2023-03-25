FROM python:3.10.10-bullseye

# Update and upgrade packages
RUN apt-get update && apt-get upgrade -y

# Install necessary packages
RUN apt-get install -y software-properties-common git cmake make gcc g++ python3-dev

# Add the deadsnakes PPA and install Python 3.10
RUN add-apt-repository --yes ppa:deadsnakes/ppa && apt-get update && apt-get install -y python3.10 python3.10-dev

# Install TensorRT
RUN apt-get install -y tensorrt tensorrt-dev tensorrt-devel tensorrt-libs

# Clone the TensorRT repo
RUN git clone https://github.com/NVIDIA/TensorRT /opt/TensorRT
WORKDIR /opt/TensorRT
RUN git submodule update --init --recursive

# Upgrade pip and install TensorRT Python package
RUN pip3 install --upgrade pip && pip3 install --upgrade tensorrt

# Set the TRT_OSSPATH environment variable
ENV TRT_OSSPATH="/opt/TensorRT"

# Build the nvinfer plugin
WORKDIR /opt/TensorRT
RUN mkdir -p build && cd build && \
    cmake .. -DTRT_OUT_DIR=$PWD/out && \
    cd plugin && make -j$(nproc)
ENV PLUGIN_LIBS="/opt/TensorRT/build/out/libnvinfer_plugin.so"

# Install requirements for demo
WORKDIR /opt/TensorRT/demo/Diffusion
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt


WORKDIR /app
COPY . .
ENTRYPOINT ["LD_PRELOAD=${PLUGIN_LIBS}" "uvicorn", "app:app", "--host 0.0.0.0", "--port 8000", "--reload" ]
