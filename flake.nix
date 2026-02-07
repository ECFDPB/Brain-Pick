{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
  };

  outputs =
    { nixpkgs, ... }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
      };
    in
    {
      packages.${system}.default = pkgs.callPackage ./package.nix { };

      devShells.${system}.default = pkgs.mkShell {
        name = "brain-pick";

        buildInputs = with pkgs.python3Packages; [
          opencv-python
          dlib

          flask
        ];

        nativeBuildInputs = with pkgs; [
          python3Packages.python-lsp-server
          ruff
          uv
        ];
      };
    };
}
