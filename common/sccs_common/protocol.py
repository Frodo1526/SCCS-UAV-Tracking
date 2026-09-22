"""Shared wire-format constants and helpers for the drone_air <-> ground_station
UDP telemetry link.

Previously ``PACKET_HEADER_FMT`` was defined independently in both
``drone_air/vid_cmd_data.py`` and ``ground_station/data_cmd.py``, with a
comment on the receiving side reading "Must match the constants in air_pi.py
exactly" -- i.e. the two sides were already known to be a hand-synchronized
contract with nothing enforcing agreement between them (see the earlier
file-relationship report, "Shared Data Models" section). Importing both
format and length from one place removes that risk.

This module also provides ``pack_fragments`` / a symmetric reassembly helper
so the fragmentation logic itself can be unit tested without a socket -- the
original code review flagged the lack of a round-trip test for this as a
concrete testing gap.
"""

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
    """Split payload_bytes into a sequence of (header + chunk) packets.

    Mirrors the original private ``_pack_fragments`` generator from
    ``vid_cmd_data.py``, exposed here so both the sending side and a test
    suite can use the same implementation.
    """
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
    """Parse the fixed 8-byte header off the front of a received datagram.

    Returns (seq, frag_id, frag_count, payload). Raises ValueError if the
    packet is shorter than the header (a "runt" packet).
    """
    if len(packet_bytes) < PACKET_HEADER_LEN:
        raise ValueError(f"packet too short: {len(packet_bytes)} bytes")
    seq, frag_id, frag_count = struct.unpack(
        PACKET_HEADER_FMT, packet_bytes[:PACKET_HEADER_LEN]
    )
    return seq, frag_id, frag_count, packet_bytes[PACKET_HEADER_LEN:]


def reassemble(fragments_by_id):
    """Join an already-complete {frag_id: payload_bytes} mapping back into
    the original payload, in order. Raises KeyError if any frag_id in
    range(len(fragments_by_id)) is missing.
    """
    return b"".join(fragments_by_id[i] for i in range(len(fragments_by_id)))
