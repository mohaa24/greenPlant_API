#!/bin/bash

# Install pyenv dependencies
sudo apt-get update
sudo apt-get install -y make build-essential libssl-dev zlib1g-dev libbz2-dev \
    libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev \
    xz-utils tk-dev libffi-dev liblzma-dev python3-openssl git

# Install pyenv
curl https://pyenv.run | bash

# Add pyenv to shell
echo -e '\nexport PATH="$HOME/.pyenv/bin:$PATH"' >> ~/.bashrc
echo -e 'eval "$(pyenv init --path)"' >> ~/.bashrc
echo -e 'eval "$(pyenv init -)"' >> ~/.bashrc
source ~/.bashrc

# Install Python 3.6.13 with pyenv
pyenv install 3.6.13

# Set the Python version to 3.6.13
pyenv global 3.6.13

# Verify installation
python --version

# Install dependencies
pip install -r requirements.txt

# Start the app (adjust as needed)
python index.py serve
