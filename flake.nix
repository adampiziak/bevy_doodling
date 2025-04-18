{
  description = "A devShell example";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    rust-overlay.url = "github:oxalica/rust-overlay";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    {
      self,
      nixpkgs,
      rust-overlay,
      flake-utils,
      ...
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        overlays = [ (import rust-overlay) ];
        pkgs = import nixpkgs {
          inherit system overlays;
        };

        bevyBuildInputs = with pkgs; [
          rust-analyzer
          udev
          alsa-lib-with-plugins
          vulkan-loader
          xorg.libX11
          xorg.libXcursor
          xorg.libXi
          xorg.libXrandr
          libxkbcommon
          wayland
          rust-bin.beta.latest.default
        ];
      in
      {
        devShells.default = pkgs.mkShell {
          nativeBuildInputs = [ pkgs.pkg-config ];
          buildInputs = bevyBuildInputs;
          shellHook = ''
            export LD_LIBRARY_PATH=${pkgs.lib.makeLibraryPath bevyBuildInputs}
          '';
        };

        packages.default = pkgs.writeShellScriptBin "lod" ''
          export LD_LIBRARY_PATH=${pkgs.lib.makeLibraryPath bevyBuildInputs}:./target/release:./target/release/deps
          exec ./target/release/lod "$@"
        '';
      }
    );
}
