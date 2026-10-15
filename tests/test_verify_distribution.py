"""Checks that CI rejects malformed or mismatched build artifacts."""

from __future__ import annotations

import importlib.util
import io
import tarfile
import zipfile
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "verify_distribution_under_test",
    Path(__file__).resolve().parents[1] / "scripts" / "verify_distribution.py",
)
assert _spec is not None and _spec.loader is not None
distribution = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(distribution)

REQUIRED_RESOURCES = distribution.REQUIRED_RESOURCES
artifact_metadata = distribution.artifact_metadata
verify = distribution.verify


def make_artifacts(
    directory: Path,
    *,
    wheel_version: str = "0.1.0",
    sdist_version: str = "0.1.0",
    missing_wheel_resource: str | None = None,
    missing_sdist_resource: str | None = None,
) -> None:
    wheel = directory / "tbcore-0.1.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(
            "tbcore-0.1.0.dist-info/METADATA",
            f"Metadata-Version: 2.4\nName: tbcore\nVersion: {wheel_version}\n",
        )
        for name in REQUIRED_RESOURCES:
            if name != missing_wheel_resource:
                archive.writestr(f"tbcore/{name}", b"test")

    sdist = directory / "tbcore-0.1.0.tar.gz"
    with tarfile.open(sdist, "w:gz") as archive:
        files = {
            "PKG-INFO": f"Metadata-Version: 2.4\nName: tbcore\nVersion: {sdist_version}\n".encode(),
            "src/tbcore.egg-info/PKG-INFO": b"nested metadata is not the root metadata",
        }
        files.update(
            {
                f"src/tbcore/{name}": b"test"
                for name in REQUIRED_RESOURCES
                if name != missing_sdist_resource
            }
        )
        for name, data in files.items():
            info = tarfile.TarInfo(f"tbcore-0.1.0/{name}")
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))


def test_matching_artifacts_ignore_nested_sdist_metadata(tmp_path: Path) -> None:
    make_artifacts(tmp_path)

    wheel, version = artifact_metadata(tmp_path)

    assert wheel.name == "tbcore-0.1.0-py3-none-any.whl"
    assert version == "0.1.0"


@pytest.mark.parametrize("source", ["wheel", "sdist"])
@pytest.mark.parametrize("missing", ["images/proxy/proxy.py", "images/codex/Dockerfile"])
def test_missing_package_resource_is_rejected(tmp_path: Path, source: str, missing: str) -> None:
    make_artifacts(
        tmp_path,
        missing_wheel_resource=missing if source == "wheel" else None,
        missing_sdist_resource=missing if source == "sdist" else None,
    )

    with pytest.raises(ValueError, match=f"{source} is missing package resources"):
        artifact_metadata(tmp_path)


def test_mismatched_artifact_metadata_is_rejected(tmp_path: Path) -> None:
    make_artifacts(tmp_path, sdist_version="0.2.0")

    with pytest.raises(ValueError, match="wheel and sdist metadata must agree"):
        artifact_metadata(tmp_path)


def test_mismatched_tag_is_rejected_before_install(tmp_path: Path) -> None:
    make_artifacts(tmp_path)

    with pytest.raises(ValueError, match="does not match built tbcore version"):
        verify(tmp_path, "v0.2.0")
