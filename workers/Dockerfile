# Use an Ubuntu base image with CUDA 12.0
FROM nvidia/cuda:12.3.1-devel-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
# Install dependencies for pyenv and SQLite
RUN apt-get update && apt-get install -y \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    wget \
    libncurses5-dev \
    libncursesw5-dev \
    libffi-dev \
    liblzma-dev \
    git \
    tzdata \
    libsqlite3-dev  # Add this line to install SQLite development libraries

# Add Tini
ENV TINI_VERSION v0.19.0
RUN wget -O /tini https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini  && \
    chmod +x /tini

# Install pyenv
RUN git clone https://github.com/pyenv/pyenv.git ~/.pyenv

# Set environment variables for pyenv
ENV HOME /root
ENV PYENV_ROOT $HOME/.pyenv
ENV PATH $PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH

# Install Python 3.8.10
RUN pyenv install 3.8.10
RUN pyenv global 3.8.10

# Install virtualenv
RUN pip install virtualenv

# Create a virtual environment
RUN virtualenv /env
ENV PATH /env/bin:$PATH

# Copy the requirements.txt and install Python dependencies
COPY requirements.txt /tmp/
RUN pip install -r /tmp/requirements.txt

# Add any other installations or custom setup scripts
COPY startup_script.sh /usr/local/bin/startup_script.sh
RUN chmod +x /usr/local/bin/startup_script.sh

# Set the entrypoint to the startup script
ENTRYPOINT ["/tini", "-g", "--", "/usr/local/bin/startup_script.sh"]
