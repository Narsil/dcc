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

          # Cross-compilation packages for x86_64-linux
          pkgsCross = import nixpkgs {
            inherit system;
            crossSystem = "x86_64-linux";
            config.allowUnfree = true;
          };
        in
        {
          default =
            with pkgs;
            mkShell {
              nativeBuildInputs = with pkgs; [
                pkg-config
                # Cross-compilation tools
                gcc
                binutils
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
                curl # For downloading CUDA toolkit

                # Cross-compilation sysroot
                pkgsCross.cudaPackages.cuda_cudart
              ];

              # Set LLVM environment variables for build.zig
              LLVM_INCLUDE_DIR = "${llvm.dev}/include";
              LLVM_LIB_DIR = "${llvm.lib}/lib";
              LLD_INCLUDE_DIR = "${lld.dev}/include";
              LLD_LIB_DIR = "${lld.lib}/lib";
              MLIR_INCLUDE_DIR = "${llvmPackages.mlir.dev}/include";
              MLIR_LIB_DIR = "${llvmPackages.mlir}/lib";

              # CUDA cross-compilation environment (manual setup)
              CUDA_INCLUDE_DIR = "${pkgsCross.cudaPackages.cuda_cudart.dev}/include";
              CUDA_LIB_DIR = "${pkgsCross.cudaPackages.cuda_cudart.lib}/lib";
              CUDA_STUB_DIR = "${pkgsCross.cudaPackages.cuda_cudart.lib}/lib/stubs";

            };
        }
      );
    };
}
