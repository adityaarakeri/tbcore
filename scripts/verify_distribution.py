"""Validate release artifacts and smoke-test the wheel outside the checkout."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tarfile
import tempfile
import venv
import zipfile
from email import policy
from email.parser import BytesParser
from pathlib import Path, PurePosixPath

REQUIRED_RESOURCES = (
    "py.typed",
    "images/sandbox/Dockerfile",
    "images/proxy/Dockerfile",
    "images/proxy/proxy.py",
    "images/codex/Dockerfile",
)


def metadata_name_version(data: bytes) -> tuple[str, str]:
    metadata = BytesParser(policy=policy.default).parsebytes(data)
    name, version = metadata.get("Name"), metadata.get("Version")
    if not name or not version:
        raise ValueError("distribution metadata is missing Name or Version")
    return name, version


def artifact_metadata(dist_dir: Path) -> tuple[Path, str]:
    wheels = list(dist_dir.glob("*.whl"))
    sdists = list(dist_dir.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        raise ValueError("expected exactly one wheel and one sdist")
    wheel, sdist = wheels[0], sdists[0]

    with zipfile.ZipFile(wheel) as archive:
        members = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
        if len(members) != 1:
            raise ValueError("wheel must contain exactly one METADATA file")
        wheel_metadata = metadata_name_version(archive.read(members[0]))
        missing = {f"tbcore/{name}" for name in REQUIRED_RESOURCES} - set(archive.namelist())
        if missing:
            raise ValueError(f"wheel is missing package resources: {sorted(missing)}")

    with tarfile.open(sdist, "r:gz") as archive:
        members = [
            item
            for item in archive.getmembers()
            if item.name.endswith("/PKG-INFO") and len(PurePosixPath(item.name).parts) == 2
        ]
        if len(members) != 1:
            raise ValueError("sdist must contain exactly one root PKG-INFO file")
        source = archive.extractfile(members[0])
        if source is None:
            raise ValueError("sdist PKG-INFO is not a file")
        sdist_metadata = metadata_name_version(source.read())
        included_files = {PurePosixPath(item.name).parts[1:] for item in archive.getmembers()}
        required_files = {
            ("src", "tbcore", *PurePosixPath(name).parts) for name in REQUIRED_RESOURCES
        }
        if missing := required_files - included_files:
            raise ValueError(f"sdist is missing package resources: {sorted(missing)}")

    if wheel_metadata != sdist_metadata or wheel_metadata[0] != "tbcore":
        raise ValueError("wheel and sdist metadata must agree on tbcore name and version")
    return wheel, wheel_metadata[1]


def installed_smoke_test(expected_version: str) -> None:
    import tbcore

    if tbcore.__version__ != expected_version:
        raise ValueError(
            f"installed version {tbcore.__version__} does not match {expected_version}"
        )
    package_dir = Path(tbcore.__file__).resolve().parent
    if not package_dir.is_relative_to(Path(sys.prefix).resolve()):
        raise ValueError(f"tbcore resolved outside the isolated environment: {package_dir}")
    if not (package_dir / "py.typed").is_file():
        raise ValueError("installed wheel is missing py.typed")
    for name in ("sandbox", "proxy", "codex"):
        context = tbcore.image_context(name)
        if not (context / "Dockerfile").is_file():
            raise ValueError(f"installed wheel is missing the {name} Dockerfile")
    if not (tbcore.image_context("proxy") / "proxy.py").is_file():
        raise ValueError("installed wheel is missing proxy.py")
    print(f"Installed tbcore {tbcore.__version__} from {package_dir}")


def verify(dist_dir: Path, tag: str | None) -> None:
    wheel, version = artifact_metadata(dist_dir)
    if tag is not None and tag != f"v{version}":
        raise ValueError(f"tag {tag!r} does not match built tbcore version v{version}")

    with tempfile.TemporaryDirectory(prefix="tbcore-wheel-smoke-") as temporary:
        temp_dir = Path(temporary).resolve()
        python = temp_dir / "venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        venv.EnvBuilder(with_pip=True).create(python.parent.parent)
        clean_env = os.environ.copy()
        clean_env.pop("PYTHONPATH", None)
        clean_env.pop("PYTHONHOME", None)
        subprocess.run([str(python), "-m", "pip", "install", str(wheel)], env=clean_env, check=True)
        subprocess.run(
            [
                str(python),
                "-I",
                str(Path(__file__).resolve()),
                "--installed-smoke",
                "--expected-version",
                version,
            ],
            cwd=temp_dir,
            env=clean_env,
            check=True,
        )
    print(f"Validated wheel and sdist for tbcore {version}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dist_dir", nargs="?", type=Path)
    parser.add_argument("--tag", help="require exact v<package-version> match")
    parser.add_argument("--installed-smoke", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--expected-version", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.installed_smoke:
        if args.expected_version is None:
            parser.error("--expected-version is required for installed smoke test")
        installed_smoke_test(args.expected_version)
    elif args.dist_dir is None:
        parser.error("dist_dir is required")
    else:
        verify(args.dist_dir.resolve(), args.tag)


if __name__ == "__main__":
    main()
