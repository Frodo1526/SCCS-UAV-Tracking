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
    '''
    Test helper that fragments a payload with pack_fragments(), parses
    each resulting packet's header with unpack_header() (asserting the
    sequence number round-trips correctly and collecting the reported
    fragment count), and reassembles the payload with reassemble().

    Args:
        seq (int): Sequence number to pack the payload with.
        payload (bytes): The payload to fragment and reassemble.

    Returns:
        bytes: The reassembled payload, expected to equal the original
            input payload.

    Raises:
        AssertionError: If a packet's decoded sequence number does not
            match seq & 0xFFFFFFFF, or if the number of fragments actually
            received does not match the frag_count reported in the
            packets' headers.
    '''
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
    '''
    Verifies that a small payload that fits in a single fragment survives
    a pack/unpack/reassemble round trip unchanged.

    Returns:
        None: This is a pytest-style test function; it does not return a
            value, and signals failure via an assert statement.
    '''
    payload = b"hello world"
    assert _round_trip(1, payload) == payload

def test_round_trip_empty_payload():
    '''
    Verifies that an empty payload (zero bytes) still round-trips
    correctly through pack_fragments()/unpack_header()/reassemble().

    Returns:
        None: This is a pytest-style test function; it does not return a
            value, and signals failure via an assert statement.
    '''
    payload = b""
    assert _round_trip(1, payload) == payload

def test_round_trip_multi_fragment_payload():
    '''
    Verifies that a payload large enough to require at least three
    fragments (more than 2x PACKET_MAX_PAYLOAD) round-trips correctly.

    Returns:
        None: This is a pytest-style test function; it does not return a
            value, and signals failure via an assert statement.
    '''
    # Force at least 3 fragments.
    payload = b"x" * (PACKET_MAX_PAYLOAD * 2 + 137)
    assert _round_trip(42, payload) == payload

def test_round_trip_payload_exactly_at_fragment_boundary():
    '''
    Verifies that a payload whose length is exactly PACKET_MAX_PAYLOAD
    produces exactly one fragment and round-trips correctly.

    Returns:
        None: This is a pytest-style test function; it does not return a
            value, and signals failure via assert statements.
    '''
    payload = b"y" * PACKET_MAX_PAYLOAD
    fragments = list(pack_fragments(1, payload))
    assert len(fragments) == 1
    assert _round_trip(1, payload) == payload

def test_pack_fragments_rejects_non_bytes():
    '''
    Verifies that pack_fragments() raises TypeError when given a payload
    that is not bytes or bytearray (e.g. a str).

    Returns:
        None: This is a pytest-style test function; it does not return a
            value.

    Raises:
        Nothing itself; it asserts (via pytest.raises) that pack_fragments()
        raises TypeError for its input.
    '''
    with pytest.raises(TypeError):
        list(pack_fragments(1, "not bytes"))

def test_unpack_header_rejects_runt_packet():
    '''
    Verifies that unpack_header() raises ValueError when given a packet
    shorter than the fixed header length.

    Returns:
        None: This is a pytest-style test function; it does not return a
            value.

    Raises:
        Nothing itself; it asserts (via pytest.raises) that unpack_header()
        raises ValueError for its input.
    '''
    with pytest.raises(ValueError):
        unpack_header(b"\x00\x01")
