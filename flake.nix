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

        nativeBuildInputs = with pkgs; [
          python3Packages.python-lsp-server
          ruff
          uv
        ];
      };
    };
}
