"""The web app's HTTP surface that needs no model: the integration service
switch, request checks, settings, result files and the result history."""

from __future__ import annotations

import io
import json
import os
import uuid
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from PIL import Image


def _jpeg(width: int = 64, height: int = 48) -> io.BytesIO:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), (120, 90, 60)).save(buf, format="JPEG")
    buf.seek(0)
    return buf


def _transform(client: Any, **fields: Any) -> Any:
    data: dict[str, Any] = {k: str(v) for k, v in fields.items()}
    data.setdefault("file", (_jpeg(), "photo.jpg"))
    return client.post("/transform", data=data, content_type="multipart/form-data")


# -- the integration service switch ----------------------------------------


def test_the_service_is_off_on_a_fresh_install(client: Any) -> None:
    health = client.get("/health")
    assert health.status_code == 503
    body = health.get_json()
    assert body["ok"] is False
    assert body["kind"] == "stereo"
    assert body["service_enabled"] is False

    refused = _transform(client)
    assert refused.status_code == 403
    assert "off" in refused.get_json()["error"]


def test_health_reports_ready_once_switched_on(
    service_on: ModuleType, client: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(service_on, "_model_status_payload", lambda: {"state": "ready"})
    body = client.get("/health").get_json()
    assert body["ok"] is True
    assert body["kind"] == "stereo"
    assert body["model_ready"] is True


def test_health_stays_ok_while_the_models_are_missing(
    service_on: ModuleType, client: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(service_on, "_model_status_payload", lambda: {"state": "missing"})
    health = client.get("/health")
    assert health.status_code == 200
    assert health.get_json()["model_ready"] is False


@pytest.mark.parametrize("value", ["1", "true", "YES", " on "])
def test_the_environment_variable_switches_it_on(
    web: ModuleType, monkeypatch: pytest.MonkeyPatch, value: str,
) -> None:
    monkeypatch.setenv("PYSTEREO_SERVICE", value)
    assert web._service_enabled() is True


@pytest.mark.parametrize("value", ["", "0", "false", "maybe"])
def test_other_values_of_the_variable_leave_it_off(
    web: ModuleType, monkeypatch: pytest.MonkeyPatch, value: str,
) -> None:
    monkeypatch.setenv("PYSTEREO_SERVICE", value)
    assert web._service_enabled() is False


def test_the_saved_setting_switches_it_on(web: ModuleType, client: Any) -> None:
    saved = client.put("/api/settings", json={"service_enabled": True})
    assert saved.status_code == 200
    assert web._service_enabled() is True
    client.put("/api/settings", json={"service_enabled": False})
    assert web._service_enabled() is False


def test_a_host_can_force_it_on(web: ModuleType) -> None:
    web.force_service_enabled(True)
    assert web._service_enabled() is True
    web.force_service_enabled(False)
    assert web._service_enabled() is False


# -- /transform request checks ----------------------------------------------


def test_transform_needs_a_file(service_on: ModuleType, client: Any) -> None:
    missing = client.post("/transform", data={}, content_type="multipart/form-data")
    assert missing.status_code == 400
    assert missing.get_json() == {"error": "Missing file field"}

    unnamed = client.post(
        "/transform",
        data={"file": (_jpeg(), "")},
        content_type="multipart/form-data",
    )
    assert unnamed.status_code == 400


def test_transform_refuses_an_unknown_output(service_on: ModuleType, client: Any) -> None:
    answer = _transform(client, output="hologram")
    assert answer.status_code == 400
    error = answer.get_json()["error"]
    assert "hologram" in error and "sbs" in error and "depth" in error


# -- settings ---------------------------------------------------------------


def test_settings_start_empty_and_are_read_back(web: ModuleType, client: Any) -> None:
    assert client.get("/api/settings").get_json() == {}
    saved = client.put("/api/settings", json={"outputs_keep": 3, "service_enabled": True})
    assert saved.get_json() == {"outputs_keep": 3, "service_enabled": True}
    assert client.get("/api/settings").get_json() == {"outputs_keep": 3, "service_enabled": True}
    on_disk = json.loads(web._settings_path().read_text(encoding="utf-8"))
    assert on_disk == {"outputs_keep": 3, "service_enabled": True}


def test_saving_one_setting_keeps_the_others(client: Any) -> None:
    client.put("/api/settings", json={"outputs_keep": 3})
    client.put("/api/settings", json={"service_enabled": True})
    assert client.get("/api/settings").get_json() == {"outputs_keep": 3, "service_enabled": True}


def test_unknown_settings_are_not_stored(web: ModuleType, client: Any) -> None:
    client.put("/api/settings", json={"outputs_keep": 3, "admin": True})
    assert client.get("/api/settings").get_json() == {"outputs_keep": 3}
    assert "admin" not in web._settings_path().read_text(encoding="utf-8")


@pytest.mark.parametrize("body", [None, [], "text", {}])
def test_settings_need_a_json_object(client: Any, body: Any) -> None:
    answer = client.put("/api/settings", json=body)
    assert answer.status_code == 400


def test_a_damaged_settings_file_reads_as_no_settings(web: ModuleType, client: Any) -> None:
    web._settings_path().write_text("{not json", encoding="utf-8")
    assert client.get("/api/settings").get_json() == {}
    assert web._service_enabled() is False


def test_setting_readers_fall_back_on_nonsense(web: ModuleType) -> None:
    assert web._setting_int({"n": "7"}, "n", 10) == 7
    assert web._setting_int({"n": "seven"}, "n", 10) == 10
    assert web._setting_int({"n": -5}, "n", 10, minimum=1) == 1
    assert web._setting_int({}, "n", 10) == 10
    assert web._setting_bool({"b": "On"}, "b") is True
    assert web._setting_bool({"b": "nope"}, "b") is False
    assert web._setting_bool({"b": 1}, "b") is True
    assert web._setting_bool({}, "b", default=True) is True


# -- result files -----------------------------------------------------------


def _result(web: ModuleType, age_s: int = 0, name: str | None = None) -> Path:
    """A result folder holding one file, last touched ``age_s`` seconds ago."""
    folder = web.OUTPUTS_DIR / (name or str(uuid.uuid4()))
    folder.mkdir()
    (folder / "sbs.jpg").write_bytes(b"jpeg bytes")
    if age_s:
        touched = folder.stat().st_mtime - age_s
        os.utime(folder, (touched, touched))
    return folder


def test_a_result_file_is_served_with_its_type(web: ModuleType, client: Any) -> None:
    folder = _result(web)
    answer = client.get(f"/api/results/{folder.name}/sbs.jpg")
    assert answer.status_code == 200
    assert answer.mimetype == "image/jpeg"
    assert answer.data == b"jpeg bytes"


def test_a_missing_result_file_is_not_found(web: ModuleType, client: Any) -> None:
    folder = _result(web)
    assert client.get(f"/api/results/{folder.name}/depth.png").status_code == 404
    assert client.get(f"/api/results/{uuid.uuid4()}/sbs.jpg").status_code == 404


@pytest.mark.parametrize("result_id", ["..", "outputs", "not-a-uuid", "."])
def test_a_result_id_must_be_a_uuid(web: ModuleType, client: Any, result_id: str) -> None:
    answer = client.get(f"/api/results/{result_id}/sbs.jpg")
    assert answer.status_code in (400, 404)
    assert answer.data != b"jpeg bytes"


@pytest.mark.parametrize(
    "filename",
    ["..", "../secret.txt", "..%2Fsecret.txt", "%2e%2e%2fsecret.txt", "..%5Csecret.txt"],
)
def test_a_file_name_cannot_reach_outside_its_result(
    web: ModuleType, client: Any, filename: str,
) -> None:
    folder = _result(web)
    (web.OUTPUTS_DIR / "secret.txt").write_text("outside the result")
    (web.OUTPUTS_DIR.parent / "secret.txt").write_text("outside the results")
    answer = client.get(f"/api/results/{folder.name}/{filename}")
    assert answer.status_code in (400, 404)
    assert b"outside" not in answer.data


@pytest.mark.parametrize("filename", ["../secret.txt", "..", "sub/../../secret.txt"])
def test_the_handler_itself_refuses_a_path_as_a_file_name(
    web: ModuleType, filename: str,
) -> None:
    """The routing above already keeps a slash out of the name. This is the
    handler's own check, for the day the route changes."""
    folder = _result(web)
    (web.OUTPUTS_DIR / "secret.txt").write_text("outside the result")
    with web.app.test_request_context():
        answer = web.app.make_response(web.get_result_file(folder.name, filename))
    assert answer.status_code == 400
    assert b"outside" not in answer.get_data()


# -- result history ---------------------------------------------------------


def test_only_the_newest_results_are_kept(web: ModuleType) -> None:
    old = _result(web, age_s=300)
    middle = _result(web, age_s=200)
    new = _result(web, age_s=100)
    assert web._prune_outputs(2) == 1
    assert not old.exists()
    assert middle.exists() and new.exists()


def test_keeping_zero_keeps_everything(web: ModuleType) -> None:
    folders = [_result(web, age_s=n * 100) for n in range(1, 4)]
    assert web._prune_outputs(0) == 0
    assert all(f.exists() for f in folders)


def test_the_result_on_screen_survives_pruning(web: ModuleType) -> None:
    old = _result(web, age_s=300)
    new = _result(web, age_s=100)
    assert web._prune_outputs(1, protect=old.name) == 0
    assert old.exists() and new.exists()


def test_folders_that_are_not_results_are_never_touched(web: ModuleType, client: Any) -> None:
    other = _result(web, age_s=900, name="my-own-folder")
    loose = web.OUTPUTS_DIR / "notes.txt"
    loose.write_text("keep me")
    _result(web, age_s=200)
    _result(web, age_s=100)

    assert web._prune_outputs(1) == 1
    assert other.exists() and loose.exists()

    cleared = client.post("/api/cache/clear", json={"target": "outputs"})
    assert cleared.status_code == 200
    assert cleared.get_json()["deleted"]["outputs"]["deleted_files"] == 1
    assert other.exists() and loose.exists()
    assert sorted(p.name for p in web.OUTPUTS_DIR.iterdir()) == ["my-own-folder", "notes.txt"]


def test_clearing_reports_what_is_left(web: ModuleType, client: Any) -> None:
    _result(web)
    _result(web)
    before = client.get("/api/cache").get_json()
    assert before["outputs"]["files"] == 2
    assert before["outputs_keep"] == web.DEFAULT_OUTPUTS_KEEP

    after = client.post("/api/cache/clear", json={"target": "outputs"}).get_json()
    assert after["outputs"] == {"path": str(web.OUTPUTS_DIR), "bytes": 0, "files": 0}


def test_clearing_refuses_an_unknown_target(client: Any) -> None:
    answer = client.post("/api/cache/clear", json={"target": "everything"})
    assert answer.status_code == 400


def test_clearing_waits_for_a_running_generation(web: ModuleType, client: Any) -> None:
    folder = _result(web)
    with web.PREDICT_LOCK:
        answer = client.post("/api/cache/clear", json={"target": "outputs"})
    assert answer.status_code == 409
    assert folder.exists()
