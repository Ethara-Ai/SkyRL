"""Reward-module test suite — spec §10.4 property tests + targeted units.

These tests are CPU-only and intentionally have no SkyRL/Ray dependency:
they exercise the pure-function aggregator, shaping function, and TIR
function in isolation, so they can run on a laptop without the trainer
stack. Per spec §10.4, the property test runs 10K random inputs against
the §4.4 formula to within 1e-9.
"""
