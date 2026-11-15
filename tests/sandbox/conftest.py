import io
import tarfile

import pytest


@pytest.fixture
def skill_archive():
    """A skill packed the way the harness sends it: ``skill_archive({"SKILL.md": b"..."})``."""

    def pack(files: dict[str, bytes], *, also=None) -> bytes:
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
            for name, data in files.items():
                info = tarfile.TarInfo(name)
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
            if also is not None:
                also(tar)
        return buffer.getvalue()

    return pack
