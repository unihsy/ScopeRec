import sys
from types import SimpleNamespace

import pytest

from scoperec.cli import main
from scoperec.official_recommend import recommend


def test_cli_dispatch_preserves_arguments_and_restores_argv(monkeypatch):
    arguments = ["scoperec", "recommend", "--device", "cpu", "--top-k", "3"]
    monkeypatch.setattr(sys, "argv", arguments)
    captured = []
    monkeypatch.setattr(
        "scoperec.cli.importlib.import_module",
        lambda name: SimpleNamespace(main=lambda: captured.append((name, list(sys.argv)))),
    )
    main()
    assert captured == [
        ("scoperec.official_recommend", ["scoperec recommend", "--device", "cpu", "--top-k", "3"])
    ]
    assert sys.argv is arguments


def test_released_demo_rejects_invalid_inputs_before_artifact_loading():
    with pytest.raises(ValueError, match="Unknown"):
        recommend("unknown", 2024, "cpu")
    with pytest.raises(ValueError, match="top-k"):
        recommend("tiger", 2024, "cpu", top_k=51)
