import pytest


@pytest.fixture(scope="session")
def gpu_device() -> str:
    from timelapse.gpu import check_device, discover_intel_gpus

    devices = discover_intel_gpus()
    if not devices:
        pytest.skip("no Intel GPU render node")
    if error := check_device(devices[0], "hevc_vaapi"):
        pytest.skip(f"VA-API encoding unavailable: {error}")
    return devices[0]
