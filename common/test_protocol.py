"""Round-trip tests for sccs_common.protocol's fragmentation helpers.

The original code review flagged the lack of any round-trip test for UDP
telemetry fragmentation/reassembly as a concrete testing gap. These tests
exercise pack_fragments()/unpack_header()/reassemble() together without any
socket, so they run anywhere.
"""

import pytest

from sccs_common.protocol import (
    PACKET_MAX_PAYLOAD,
    pack_fragments,
    unpack_header,
    reassemble,
)


def _round_trip(seq, payload):
    fragments_by_id = {}
    frag_count_seen = None
    for packet in pack_fragments(seq, payload):
        pkt_seq, frag_id, frag_count, chunk = unpack_header(packet)
        assert pkt_seq == seq & 0xFFFFFFFF
        frag_count_seen = frag_count
        fragments_by_id[frag_id] = chunk
    assert len(fragments_by_id) == frag_count_seen
    return reassemble(fragments_by_id)


def test_round_trip_small_single_fragment_payload():
    payload = b"hello world"
    assert _round_trip(1, payload) == payload


def test_round_trip_empty_payload():
    payload = b""
    assert _round_trip(1, payload) == payload


def test_round_trip_multi_fragment_payload():
    # Force at least 3 fragments.
    payload = b"x" * (PACKET_MAX_PAYLOAD * 2 + 137)
    assert _round_trip(42, payload) == payload


def test_round_trip_payload_exactly_at_fragment_boundary():
    payload = b"y" * PACKET_MAX_PAYLOAD
    fragments = list(pack_fragments(1, payload))
    assert len(fragments) == 1
    assert _round_trip(1, payload) == payload


def test_pack_fragments_rejects_non_bytes():
    with pytest.raises(TypeError):
        list(pack_fragments(1, "not bytes"))


def test_unpack_header_rejects_runt_packet():
    with pytest.raises(ValueError):
        unpack_header(b"\x00\x01")
