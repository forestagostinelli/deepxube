FROM mcr.microsoft.com/devcontainers/python:1-3.10-bookworm

RUN python -m pip install --upgrade pip

# Install the released package from PyPI.
RUN python -m pip install --no-cache-dir deepxube

# Ensure the CLI is available.
RUN deepxube --help