import flex_dep_opt as fd


def test_every_exported_name_is_importable():
    assert all(hasattr(fd, name) for name in fd.__all__)
