import pytest

from tbcore.sandbox import build_default_images
from tbcore.testing import docker_available


@pytest.fixture(scope="session")
def require_docker():
    if not docker_available():
        pytest.skip("docker runtime not available")


@pytest.fixture(scope="session")
def built_images(require_docker):
    return build_default_images(tag_prefix="tbcore-dev")


@pytest.fixture(scope="session")
def sandbox_image(built_images):
    return built_images["sandbox"]


@pytest.fixture(scope="session")
def proxy_image(built_images):
    return built_images["proxy"]
