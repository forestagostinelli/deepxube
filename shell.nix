# simple nix dev environment https://wiki.nixos.org/wiki/Python
# the environment can be entered by running:
# `$ nix develop -f shell.nix`

let
  # We pin to a specific nixpkgs commit for reproducibility.
  # Last updated: 2024-04-29. Check for new commits at https://status.nixos.org.
  pkgs = import (fetchTarball "https://github.com/NixOS/nixpkgs/archive/cf8cc1201be8bc71b7cbbbdaf349b22f4f99c7ae.tar.gz") {};
in pkgs.mkShell {
  packages = [
    (pkgs.python3.withPackages (python-pkgs: with python-pkgs; [
      # select Python packages here
      torch
      numpy
      nptyping # type hints for numpy
      matplotlib
      pillow
      types-pillow
      imageio
      wget
      filelock
      pytest
      pytest-cov
      ecpy # for doing ecc point operations
    ]))

    # for logic (does not seem to be nix python package for it -- might need to make one)
    pkgs.clingo

    # for vs code (optional)
    # pkgs.vscode-extensions.ms-python.vscode-pylance
    # pkgs.vscode-extensions.ms-pyright.pyright

    # seems necessary to get vs code to work on NixOS
    pkgs.pipenv
  ];
}