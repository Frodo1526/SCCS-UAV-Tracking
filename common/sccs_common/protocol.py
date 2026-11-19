"""Shared wire-format constants and helpers for the drone_air <-> ground_station UDP telemetry link."""

import math
import struct

# seq (uint32), frag_id (uint16), frag_count (uint16)
PACKET_HEADER_FMT = "!IHH"
PACKET_HEADER_LEN = struct.calcsize(PACKET_HEADER_FMT)
PACKET_MAX_BYTES = 1400
PACKET_MAX_PAYLOAD = PACKET_MAX_BYTES - PACKET_HEADER_LEN

# Default network settings, kept here only as the single documented fallback
# used by get_setting() callers in each subsystem's own port constants.
DEFAULT_PORTS = {
    "video_port": 5602,
    "data_port": 5601,
    "command_port": 5603,
}

def pack_fragments(seq, payload_bytes):
    '''
    Splits a payload into one or more UDP-sized fragments, each prefixed
    with an 8-byte header (sequence number, fragment index, total fragment
    count) so the receiver can reassemble the original payload out of
    order. An empty payload still yields exactly one (header-only) packet.

    Args:
        payload_bytes (bytes | bytearray): The full message payload to
            fragment.
        seq (int): Sequence number identifying this message; masked to 32
            bits when packed into the header.

    Yields:
        bytes: Successive (header + chunk) packets, each at most
            PACKET_MAX_BYTES long, until the full payload has been
            covered.

    Raises:
        TypeError: If payload_bytes is not a bytes or bytearray instance.
    '''
    if not isinstance(payload_bytes, (bytes, bytearray)):
        raise TypeError("payload_bytes must be bytes or bytearray")

    total = len(payload_bytes)
    frag_count = 1 if total == 0 else int(math.ceil(total / PACKET_MAX_PAYLOAD))

    for frag_id in range(frag_count):
        start = frag_id * PACKET_MAX_PAYLOAD
        chunk = payload_bytes[start:start + PACKET_MAX_PAYLOAD]
        header = struct.pack(PACKET_HEADER_FMT, seq & 0xFFFFFFFF, frag_id, frag_count)
        yield header + chunk

def unpack_header(packet_bytes):
    '''
    Parses the fixed 8-byte fragmentation header off the front of a
    received UDP datagram.

    Args:
        packet_bytes (bytes): A raw datagram, expected to start with an
            8-byte header as packed by pack_fragments().

    Returns:
        tuple[int, int, int, bytes]: (seq, frag_id, frag_count, payload),
            where seq is the message sequence number, frag_id is this
            fragment's 0-based index, frag_count is the total number of
            fragments for the message, and payload is everything after
            the header.

    Raises:
        ValueError: If packet_bytes is shorter than PACKET_HEADER_LEN (a
            "runt" packet that cannot contain a full header).
    '''
    if len(packet_bytes) < PACKET_HEADER_LEN:
        raise ValueError(f"packet too short: {len(packet_bytes)} bytes")
    seq, frag_id, frag_count = struct.unpack(
        PACKET_HEADER_FMT, packet_bytes[:PACKET_HEADER_LEN]
    )
    return seq, frag_id, frag_count, packet_bytes[PACKET_HEADER_LEN:]

def reassemble(fragments_by_id):
    '''
    Joins a complete mapping of fragment index to payload bytes back into
    the original, contiguous payload, in fragment order.

    Args:
        fragments_by_id (dict[int, bytes]): Mapping from fragment index to
            that fragment's payload bytes. Must contain every integer key
            from 0 up to (but not including) len(fragments_by_id) for a
            correct reassembly.

    Returns:
        bytes: The concatenation of fragments_by_id[0], fragments_by_id[1],
            ..., fragments_by_id[len(fragments_by_id) - 1].

    Raises:
        KeyError: If any index in range(len(fragments_by_id)) is missing
            from fragments_by_id (i.e. the fragment set is incomplete or
            has gaps).
    '''
    return b"".join(fragments_by_id[i] for i in range(len(fragments_by_id)))
