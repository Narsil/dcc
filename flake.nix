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
    in
    {
      devShells = forAllSystems (
        system:
        let
          pkgs = import nixpkgs {
            inherit system;
            config.allowUnfree = true;

          };
        in
        {
          default =
            with pkgs;
            mkShell {
              nativeBuildInputs = with pkgs; [
                pkg-config
              ];
              buildInputs = with pkgs; [
                zig
                llvm
                lld
                llvmPackages.mlir
                gemini-cli
                zls
                gdb
                claude-code
                zlib
              ];

              # Set LLVM environment variables for build.zig
              LLVM_INCLUDE_DIR = "${llvm.dev}/include";
              LLVM_LIB_DIR = "${llvm.lib}/lib";
              LLD_INCLUDE_DIR = "${lld.dev}/include";
              LLD_LIB_DIR = "${lld.lib}/lib";
              MLIR_INCLUDE_DIR = "${llvmPackages.mlir.dev}/include";
              MLIR_LIB_DIR = "${llvmPackages.mlir}/lib";
            };
        }
      );
    };
}
