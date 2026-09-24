"""Check that a Git tag and built Python distributions describe one release."""

from __future__ import annotations

import argparse
import re
import tarfile
import tomllib
import zipfile
from email.parser import Parser
from pathlib import Path


def distribution_identity(metadata: str) -> tuple[str, str]:
    fields = Parser().parsestr(metadata, headersonly=True)
    return fields.get("Name", ""), fields.get("Version", "")


def verify(project: Path, dist: Path, tag: str) -> None:
    package = tomllib.loads((project / "pyproject.toml").read_text())["project"]
    name = package["name"]
    version = package["version"]
    if tag != f"v{version}":
        raise ValueError(f"Tag {tag!r} must equal the project version v{version}")

    filename_name = re.sub(r"[-_.]+", "_", name).lower()
    wheel_files = sorted(dist.glob("*.whl"))
    source_files = sorted(dist.glob("*.tar.gz"))
    if len(wheel_files) != 1 or len(source_files) != 1:
        raise ValueError("Expected exactly one wheel and one source distribution")

    wheel, source = wheel_files[0], source_files[0]
    if not wheel.name.startswith(f"{filename_name}-{version}-"):
        raise ValueError(f"Unexpected wheel name: {wheel.name}")
    if source.name != f"{filename_name}-{version}.tar.gz":
        raise ValueError(f"Unexpected source distribution name: {source.name}")

    with zipfile.ZipFile(wheel) as archive:
        metadata_files = [path for path in archive.namelist() if path.endswith(".dist-info/METADATA")]
        if len(metadata_files) != 1:
            raise ValueError("Wheel must contain exactly one METADATA file")
        wheel_identity = distribution_identity(archive.read(metadata_files[0]).decode())
        if not any(path.startswith("max_ai/") for path in archive.namelist()):
            raise ValueError("Wheel does not contain the max_ai package")

    with tarfile.open(source, "r:gz") as archive:
        metadata_files = [entry for entry in archive.getmembers() if entry.name.endswith("/PKG-INFO")]
        if len(metadata_files) != 1:
            raise ValueError("Source distribution must contain exactly one PKG-INFO file")
        member = archive.extractfile(metadata_files[0])
        if member is None:
            raise ValueError("Could not read source distribution metadata")
        source_identity = distribution_identity(member.read().decode())

    expected = (name, version)
    if wheel_identity != expected or source_identity != expected:
        raise ValueError(
            f"Distribution metadata mismatch: expected {expected}, "
            f"wheel has {wheel_identity}, source has {source_identity}"
        )
    print(f"Verified {name} {version}: {wheel.name} and {source.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path("."))
    parser.add_argument("--dist", type=Path, default=Path("dist"))
    parser.add_argument("--tag", required=True)
    arguments = parser.parse_args()
    verify(arguments.project, arguments.dist, arguments.tag)
