"""The checks that decide which model files may be deleted.

Deleting follows a path, so a wrong answer here removes the wrong folder.
Nothing is downloaded or deleted: the torch hub folder is a temporary one.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pystereo_core import download
from pystereo_core.download import (
    DEPTH_MODEL_SPECS,
    SHARP_SPEC,
    STEREO_PACK,
    ArtifactSpec,
)


@pytest.fixture
def torch_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An empty torch hub folder; returns its ``checkpoints`` folder."""
    import torch.hub

    home = tmp_path / "torch"
    monkeypatch.setenv("TORCH_HOME", str(home))
    monkeypatch.delenv("LAMA_MODEL", raising=False)
    # A folder set by an earlier ``torch.hub.set_dir`` would win over the variable.
    monkeypatch.setattr(torch.hub, "_hub_dir", None)
    checkpoints = home / "hub" / "checkpoints"
    checkpoints.mkdir(parents=True)
    return checkpoints.resolve()


def test_a_repo_folder_is_named_as_the_hub_names_it() -> None:
    assert download._hf_repo_folder_name("org/name") == "models--org--name"


def test_relative_to_follows_the_real_location(tmp_path: Path) -> None:
    root = tmp_path / "root"
    (root / "inside").mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    assert download._is_relative_to(root / "inside", root)
    assert not download._is_relative_to(outside, root)
    # Written as a path under root, but it leaves it.
    assert not download._is_relative_to(root / ".." / "outside", root)


@pytest.mark.parametrize("repo_id", ["", "no-slash", " org/name", "org/name "])
def test_a_malformed_repo_id_has_no_folder_to_delete(repo_id: str) -> None:
    assert download._hf_repo_cache_dir(repo_id) is None


def test_a_known_checkpoint_resolves_inside_the_checkpoints_folder(
    torch_home: Path,
) -> None:
    path = download._safe_url_cache_file("https://example.com/models/big-lama.pt")
    assert path == torch_home / "big-lama.pt"


@pytest.mark.parametrize(
    "url",
    [
        "",
        "https://example.com/models/other.pt",
        "https://example.com/models/",
        "https://example.com/big-lama.pt/..",
    ],
)
def test_an_unknown_checkpoint_name_is_refused(torch_home: Path, url: str) -> None:
    assert download._safe_url_cache_file(url) is None


def test_dots_in_the_url_cannot_move_the_file_out_of_the_folder(
    torch_home: Path,
) -> None:
    path = download._safe_url_cache_file("https://example.com/../../big-lama.pt")
    assert path == torch_home / "big-lama.pt"


def test_a_checkpoint_is_deletable_only_once_it_exists(torch_home: Path) -> None:
    lama = next(spec for spec in STEREO_PACK if spec.id == "inpaint")
    assert download._artifact_deletable_path(lama) is None
    (torch_home / "big-lama.pt").write_bytes(b"weights")
    assert download._artifact_deletable_path(lama) == torch_home / "big-lama.pt"


def test_a_lama_file_the_user_supplied_is_never_deleted(
    torch_home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    lama = next(spec for spec in STEREO_PACK if spec.id == "inpaint")
    (torch_home / "big-lama.pt").write_bytes(b"weights")
    own = tmp_path / "my-lama.pt"
    own.write_bytes(b"weights")
    monkeypatch.setenv("LAMA_MODEL", str(own))
    assert download._artifact_deletable_path(lama) is None
    assert download._artifact_is_local(lama) is True


def test_the_sharp_checkpoint_is_not_guarded_by_the_lama_variable(
    torch_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    (torch_home / download.SHARP_CKPT_FILENAME).write_bytes(b"weights")
    monkeypatch.setenv("LAMA_MODEL", "/somewhere/else.pt")
    assert download._artifact_deletable_path(SHARP_SPEC) == (
        torch_home / download.SHARP_CKPT_FILENAME
    )


def test_a_spec_with_nothing_to_point_at_is_neither_local_nor_deletable() -> None:
    empty = ArtifactSpec(id="x", name="X", kind="url")
    assert download._artifact_is_local(empty) is False
    assert download._artifact_deletable_path(empty) is None


def test_the_three_depth_models_are_separate_downloads() -> None:
    assert set(DEPTH_MODEL_SPECS) == {"small", "base", "large"}
    repos = {spec.repo_id for spec in DEPTH_MODEL_SPECS.values()}
    assert len(repos) == 3
    # The small one is the depth model of the stereo pack.
    pack_depth = next(spec for spec in STEREO_PACK if spec.id == "depth")
    assert DEPTH_MODEL_SPECS["small"].repo_id == pack_depth.repo_id
