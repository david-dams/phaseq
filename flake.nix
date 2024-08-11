{
  description = "phaseQ";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
    in
    {
      devShells.default = pkgs.mkShell {
        buildInputs = [ pkgs.python312Full pkgs.python312Packages.pip ];
      };

      packages.default = pkgs.stdenv.mkDerivation {
        pname = "phaseQ";
        version = "0.1";

        src = ./.;

        buildInputs = [ pkgs.python312Full pkgs.python312Packages.pip ];

        buildPhase = ''
        pip install -e .
        '';

        meta = with pkgs.lib; {
          description = "";
          license = licenses.mit;
        };
      };
    });
}
