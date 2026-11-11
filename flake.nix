{
  description = "Development environment for dual-arm manipulation in PyBullet";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { nixpkgs, ... }:
    let
      systems = [ "x86_64-linux" "aarch64-linux" "aarch64-darwin" ];
      forAllSystems = nixpkgs.lib.genAttrs systems;
    in
    {
      devShells = forAllSystems (system:
        let
          pkgs = import nixpkgs { inherit system; };
          python = pkgs.python311;
          tkinter = pkgs.python311Packages.tkinter;
          runtimeLibraries = with pkgs; [
            stdenv.cc.cc.lib
            zlib
          ] ++ lib.optionals stdenv.hostPlatform.isLinux [
            glib
            libglvnd
            libusb1
            (lib.getLib udev)
            libx11
            libxext
            libxrender
            libxrandr
            libxi
            libxcursor
            libsm
            libice
          ];
        in
        {
          default = pkgs.mkShell {
            packages = with pkgs; [
              python
              tkinter
              uv
              cmake
              ninja
              pkg-config
              git
              ffmpeg
              wget
              unzip
            ] ++ runtimeLibraries;

            # Use Nix's interpreter, including its patched loader, for uv's venv.
            UV_PYTHON = "${python}/bin/python3.11";
            UV_PYTHON_DOWNLOADS = "never";

            shellHook = ''
              # uv isolates site-packages, so expose Nix's Tk extension explicitly.
              export PYTHONPATH="${tkinter}/${python.sitePackages}''${PYTHONPATH:+:$PYTHONPATH}"
            '' + pkgs.lib.optionalString pkgs.stdenv.hostPlatform.isLinux ''
              export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath runtimeLibraries}''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
              # NixOS provides the active host graphics/CUDA driver here.
              if [ -d /run/opengl-driver/lib ]; then
                export LD_LIBRARY_PATH="/run/opengl-driver/lib:$LD_LIBRARY_PATH"
              fi
            '';
          };
        });
    };
}
