"""Compiles the Mojo extension into the wheel, when a Mojo toolchain is present.

The gate is whether the Mojo package imports, never a platform list: Modular ships wheels for
Linux x86-64, Linux aarch64 and macOS arm64 today and may add more, and anywhere it does not ship
one this hook simply produces the pure-Python wheel.

A freshly built extension records its runtime dependencies by absolute path into the build
environment, which the installer deletes afterwards, so the libraries are copied in beside it and
the loader is repointed at the wheel's own directory. Without that the wheel installs and then
fails to import.
"""

import pathlib
import shutil
import subprocess
import sys
import tempfile

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

# The loaders differ in these three and nothing else, so nothing below asks which platform it is on.
if sys.platform == "darwin":
    extension_suffix = "dylib"
    list_linkage = ["otool", "-L"]
    set_search_path = ["install_name_tool", "-add_rpath", "@loader_path"]
else:
    extension_suffix = "so"
    list_linkage = ["ldd"]
    set_search_path = ["patchelf", "--set-rpath", "$ORIGIN"]


class MojoExtensionHook(BuildHookInterface):
    PLUGIN_NAME = "mojo"

    def initialize(self, version, build_data):
        try:
            from mojo.run import subprocess_run_mojo
        except ImportError:
            return  # No toolchain for this platform, so ship the reference alone.

        source = pathlib.Path(self.root) / "affinegaps.mojo"
        staging = pathlib.Path(tempfile.mkdtemp(prefix="affinegaps-"))
        extension = staging / f"affinegaps_mojo.{extension_suffix}"
        outcome = subprocess_run_mojo(["build", str(source), "--emit", "shared-lib", "-o", str(extension)])
        if outcome.returncode != 0 or not extension.exists():
            return

        # An unbundled extension installs and then fails to import, so ship the reference instead.
        if not self._loader_tools_present():
            return

        bundled = [extension, *self._bundle_runtime(extension, staging)]
        for library in bundled:
            self._repoint(library)
            build_data["force_include"][str(library)] = library.name
        # The extension links no libpython, so one artifact serves every Python version.
        build_data["tag"] = f"py3-none-{self._platform_tag()}"
        build_data["pure_python"] = False

    @staticmethod
    def _platform_tag() -> str:
        """The wheel's platform tag, with the punctuation wheels do not allow."""
        from packaging.tags import sys_tags

        return next(tag.platform for tag in sys_tags())

    @staticmethod
    def _loader_tools_present() -> bool:
        """Whether both loader tools resolve, since bundling has to read the linkage and rewrite it."""
        return all(shutil.which(argv[0]) is not None for argv in (list_linkage, set_search_path))

    @staticmethod
    def _bundle_runtime(extension: pathlib.Path, staging: pathlib.Path) -> list:
        """Copies in the Modular runtime libraries the extension resolves against.

        They are discovered by asking the loader rather than hardcoded, so a release that renames
        them does not silently produce a wheel that cannot import.
        """
        listing = subprocess.run([*list_linkage, str(extension)], capture_output=True, text=True)
        copied = []
        for token in dict.fromkeys(listing.stdout.split()):
            path = pathlib.Path(token)
            if path.is_absolute() and path.is_file() and "modular" in path.parts:
                copied.append(pathlib.Path(shutil.copy2(path, staging / path.name)))
        return copied

    @staticmethod
    def _repoint(library: pathlib.Path) -> None:
        """Points the library at whatever directory it ends up installed in."""
        # An rpath that is already recorded exits non-zero, which is not a failure.
        subprocess.run([*set_search_path, str(library)], check=False)
