FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04 

# Setup general.
RUN apt update && \
    apt upgrade --yes && \
    apt install --no-install-recommends --allow-change-held-packages --yes \
    ca-certificates git git-lfs curl jq grep nano apt-utils dpkg gettext procps build-essential pkg-config software-properties-common htop \
    python3 python3-pip python3-dev python3-tk python-is-python3 \
    make cmake libssl-dev lldb llvm llvm-dev libclang-dev clang protobuf-compiler zlib1g swig \
    mesa-utils mesa-utils-extra glmark2 libnvidia-egl-xlib1 libnvidia-egl-xcb1 libcupti-dev libnccl2 libnccl-dev guvcview nvtop \
    portaudio19-dev libsndfile-dev libjack-jackd2-dev libportmidi-dev liblo-dev alsa-utils libasound2-dev libopencv-dev libgtk-3-dev libgl1 libglib2.0-0 libgl1-mesa-dev libxrender1 libxext6 libgl1-mesa-dri ffmpeg gstreamer1.0-tools gstreamer1.0-libav libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev && \
    update-ca-certificates && \
    apt full-upgrade --yes && \
    apt autoremove --yes && \
    apt clean --yes && \
    rm --recursive --force /var/lib/apt/lists/* /tmp/* /var/tmp/* 

# Setup git credential manager.
RUN REPOSITORY="git-ecosystem/git-credential-manager" && \
    VERSION=$(curl --location --silent --show-error --fail "https://api.github.com/repos/${REPOSITORY}/releases/latest" | jq --raw-output '.tag_name') && \
    curl --location --silent --show-error --fail "https://github.com/${REPOSITORY}/releases/download/${VERSION}/gcm-linux_amd64.${VERSION#v}.tar.gz" | \
    tar --extract --gzip --mode=0755 --directory="/usr/local/bin" --exclude=NOTICE && \
    git credential-manager configure

# Setup git credential oauth.
RUN REPOSITORY="hickford/git-credential-oauth" && \
    VERSION=$(curl --location --silent --show-error --fail "https://api.github.com/repos/${REPOSITORY}/releases/latest" | jq --raw-output '.tag_name') && \
    curl --location --silent --show-error --fail "https://github.com/${REPOSITORY}/releases/download/${VERSION}/git-credential-oauth_${VERSION#v}_linux_amd64.tar.gz" | \
    tar --extract --gzip --mode=0755 --directory="/usr/local/bin" --exclude=LICENSE.txt --exclude=README.md && \
    git credential-oauth configure

# Setup git configuration.
RUN git config --system pull.rebase true && \
    git config --system --add credential.helper "cache --timeout 7200" && \
    git config --system --add credential.helper oauth && \
    git config --system gpg.format ssh && \
    git config --system commit.gpgsign true && \
    git config --system uploadpack.allowFilter true

# Setup python.
RUN pip install --break-system-packages --upgrade ultralytics mediapipe pyo