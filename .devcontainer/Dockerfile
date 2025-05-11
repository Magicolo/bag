FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04 

# Setup general.
RUN apt update && \
    apt upgrade --yes && \
    apt install --no-install-recommends --yes \
    ca-certificates git git-lfs curl jq grep nano apt-utils dpkg gettext procps build-essential pkg-config software-properties-common htop \
    python3 python3-pip python-is-python3 \
    make cmake libssl-dev lldb llvm llvm-dev libclang-dev clang protobuf-compiler zlib1g \
    alsa-utils libasound2-dev libopencv-dev libgtk-3-dev libgl1 libglib2.0-0 libgl1-mesa-dev libxrender1 libxext6 libgl1-mesa-dri ffmpeg gstreamer1.0-tools gstreamer1.0-libav libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev && \
    update-ca-certificates && \
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

# Setup act.
RUN REPOSITORY="nektos/act" && \
    VERSION=$(curl --location --silent --show-error --fail "https://api.github.com/repos/${REPOSITORY}/releases/latest" | jq --raw-output '.tag_name') && \
    curl --location --silent --show-error --fail "https://github.com/${REPOSITORY}/releases/download/${VERSION}/act_Linux_x86_64.tar.gz" | \
    tar --extract --gzip --mode=0755 --directory="/usr/local/bin" --exclude=LICENSE --exclude=README.md

# Setup rust.
ENV RUSTUP_HOME="/usr/local/rustup"
ENV CARGO_HOME="/usr/local/cargo"
ENV PATH="${PATH}:${CARGO_HOME}/bin"
RUN curl --location --silent --show-error --fail "https://sh.rustup.rs" | sh -s -- -y --default-toolchain none && \
    rustup toolchain install stable nightly --component rustfmt --component llvm-tools-preview --component rust-src --component clippy && \
    rustup default stable

# Setup cargo.
RUN REPOSITORY="cargo-bins/cargo-binstall" && \
    VERSION=$(curl --location --silent --show-error --fail "https://api.github.com/repos/${REPOSITORY}/releases/latest" | jq --raw-output '.tag_name') && \
    curl --location --silent --show-error --fail "https://github.com/${REPOSITORY}/releases/download/${VERSION}/cargo-binstall-x86_64-unknown-linux-gnu.tgz" | \
    tar --extract --gzip --mode=0755 --directory="/usr/local/bin" && \
    cargo binstall cargo-watch cargo-update cargo-expand cargo-audit cargo-msrv cargo-hack cargo-semver-checks cargo-release cargo-nextest cargo-bloat cargo-llvm-cov && \
    chmod --recursive 0777 "$CARGO_HOME"

# Setup python.
ENV LIBTORCH_USE_PYTORCH=1
RUN pip install --break-system-packages --upgrade --index-url https://download.pytorch.org/whl/cu128 torch torchvision torchaudio && \
    pip install --break-system-packages --upgrade ultralytics mediapipe