{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (system:
    let
      pythonVersion = "python312";
      pkgs = nixpkgs.legacyPackages.${system};
      pythonPackages = pkgs.python312Packages;
      in
        {                          
          devShells.default = pkgs.mkShell {
            name = "zum kotzen";
            venvDir = "./.venv";
            packages = [ pythonPackages.python
                         pythonPackages.venvShellHook
                         pythonPackages.numpy
                         pythonPackages.jax
                         pythonPackages.jaxlib ];
          };
        });
}
