{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs =
    {
      nixpkgs,
      ...
    }:
    let
      forAllSystems = nixpkgs.lib.genAttrs [
        "aarch64-linux"
        "x86_64-linux"
        "aarch64-darwin"
      ];

      inherit (nixpkgs) lib;
    in
    {
      devShells = forAllSystems (system: {
        default =
          with nixpkgs.legacyPackages.${system}.pkgs;
          mkShell {
            nativeBuildInputs = with pkgs; [
              pkg-config
            ];
            buildInputs = with pkgs; [
              zig
              llvm
              gemini-cli
            ];

            # Set LLVM environment variables for build.zig
            LLVM_INCLUDE_DIR = "${llvm.dev}/include";
            LLVM_LIB_DIR = "${llvm.lib}/lib";
          };
      });
    };
}
