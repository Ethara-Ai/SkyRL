"""Tests for milo.lht_adapter.image_naming.

Exercises the canonical image-name helper against representative milo-bench
instances (covering org/repo/number variants we see on disk in
``/Users/piyush/github/freya/milo-bench/dataset``) per
``IMPLEMENTATION_PLAN.md`` v0.4 §1 ("use the canonical image-naming helper").
"""

from __future__ import annotations

import os
from unittest import mock

import pytest

from milo.lht_adapter.image_naming import (
    DEFAULT_ECR_PREFIX,
    get_image_name,
    get_image_prefix,
    parse_instance_id,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def instance_locustio_1541():
    """Canonical milo-bench instance — Cohort A, python."""
    return {
        "instance_id": "locustio__locust-1541",
        "org": "locustio",
        "repo": "locust",
        "number": 1541,
        "lang": "python",
    }


@pytest.fixture
def instance_amrex():
    """Multi-bundle, AMReX-Codes / amrex — bundle_size > 1."""
    return {
        "instance_id": "AMReX-Codes__amrex-4271",
        "org": "AMReX-Codes",
        "repo": "amrex",
        "number": 4271,
        "lang": "cpp",
    }


@pytest.fixture
def instance_slashrepo():
    """Instance where `repo` field is ``org/name`` rather than just ``name``."""
    return {
        "instance_id": "foo__bar-99",
        "repo": "foo/bar",
        "number": 99,
        "lang": "go",
    }


@pytest.fixture
def instance_no_number():
    """Number missing in payload — should fall back to parsing instance_id."""
    return {
        "instance_id": "kubernetes__kubernetes-12345",
        "org": "kubernetes",
        "repo": "kubernetes",
        "lang": "go",
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestGetImagePrefix:
    def test_default_prefix(self, monkeypatch):
        monkeypatch.delenv("EVAL_DOCKER_IMAGE_PREFIX", raising=False)
        assert get_image_prefix() == DEFAULT_ECR_PREFIX

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("EVAL_DOCKER_IMAGE_PREFIX", "myregistry.example.com/proj")
        assert get_image_prefix() == "myregistry.example.com/proj"


class TestGetImageName:
    def test_locustio_canonical(self, instance_locustio_1541, monkeypatch):
        monkeypatch.delenv("EVAL_DOCKER_IMAGE_PREFIX", raising=False)
        name = get_image_name(instance_locustio_1541)
        assert name == (
            "426628337772.dkr.ecr.ap-south-1.amazonaws.com/rfp-coding-q1/"
            "locustio_m_locust:pr-1541"
        )

    def test_amrex_mixedcase_lowercased(self, instance_amrex, monkeypatch):
        monkeypatch.delenv("EVAL_DOCKER_IMAGE_PREFIX", raising=False)
        name = get_image_name(instance_amrex)
        # The whole name MUST be lowercased per docker registry rules.
        assert name == name.lower()
        assert "amrex-codes_m_amrex:pr-4271" in name

    def test_repo_with_slash_split(self, instance_slashrepo, monkeypatch):
        monkeypatch.delenv("EVAL_DOCKER_IMAGE_PREFIX", raising=False)
        name = get_image_name(instance_slashrepo)
        assert name.endswith("/foo_m_bar:pr-99")

    def test_number_inferred_from_instance_id(self, instance_no_number, monkeypatch):
        monkeypatch.delenv("EVAL_DOCKER_IMAGE_PREFIX", raising=False)
        name = get_image_name(instance_no_number)
        assert name.endswith("/kubernetes_m_kubernetes:pr-12345")

    def test_custom_prefix(self, instance_locustio_1541):
        name = get_image_name(instance_locustio_1541, prefix="reg.example/foo")
        assert name == "reg.example/foo/locustio_m_locust:pr-1541"

    def test_env_prefix_applied(self, instance_locustio_1541, monkeypatch):
        monkeypatch.setenv("EVAL_DOCKER_IMAGE_PREFIX", "custom.example/x")
        name = get_image_name(instance_locustio_1541)
        assert name.startswith("custom.example/x/")
        assert name.endswith("locustio_m_locust:pr-1541")

    def test_missing_number_no_id_falls_back_to_base(self):
        instance = {"org": "foo", "repo": "bar"}
        name = get_image_name(instance, prefix="reg.example")
        # No PR number anywhere => tag is `base` per the helper.
        assert name.endswith(":base")


class TestParseInstanceId:
    def test_roundtrip_locustio(self, instance_locustio_1541):
        name = get_image_name(instance_locustio_1541, prefix="reg.example")
        iid = parse_instance_id(name)
        assert iid == "locustio__locust-1541"

    def test_roundtrip_amrex(self, instance_amrex):
        name = get_image_name(instance_amrex, prefix="reg.example")
        # Note: lowercasing means we can't recover the original casing.
        iid = parse_instance_id(name)
        assert iid == "amrex-codes__amrex-4271"

    def test_no_tag(self):
        # parse_instance_id should handle a name without a tag colon.
        assert parse_instance_id("foo_m_bar") == "foo__bar"
