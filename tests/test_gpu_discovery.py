from timelapse.gpu import discover_intel_gpus


def make_node(drm, dev, name, vendor, present=True):
    (drm / name / "device").mkdir(parents=True)
    (drm / name / "device" / "vendor").write_text(vendor + "\n")
    if present:
        (dev / name).touch()


def test_finds_present_intel_render_nodes_in_order(tmp_path):
    drm, dev = tmp_path / "drm", tmp_path / "dri"
    dev.mkdir()
    make_node(drm, dev, "renderD130", "0x8086")
    make_node(drm, dev, "renderD129", "0x1002")  # AMD iGPU
    make_node(drm, dev, "renderD128", "0x8086")
    make_node(drm, dev, "renderD131", "0x8086", present=False)  # not passed into the container
    (drm / "card0").mkdir()
    assert discover_intel_gpus(drm, dev) == [str(dev / "renderD128"), str(dev / "renderD130")]
