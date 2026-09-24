import socket
import struct
import time
import json
import os
import msgpack
import logging

from common.sccs_common.config import load_config, get_setting
from common.sccs_common.protocol import PACKET_HEADER_FMT, PACKET_HEADER_LEN

# Read config and apply defaults
_CFG = load_config()
_NET = _CFG.get("network", {})
RADXA_GADGET_IP = get_setting(_NET, "ground_gadget_ip", "10.55.0.1")
VIDEO_STREAM_PORT = int(get_setting(_NET, "video_port", 5602)) # Port to listen on for H.264 video (forwarded from drone)
DATA_PORT = int(get_setting(_NET, "data_port", 5601)) # Port to listen on for tracking data (forwarded from drone)
COMMAND_PORT = int(get_setting(_NET, "command_port", 5603)) # Port to send commands to (Radxa forwards to drone)

# PACKET_HEADER_FMT/LEN now come from sccs_common.protocol, imported by both
# drone_air and ground_station -- previously these were two independently
# hand-copied constants with a comment admitting they "must match exactly";
# now there's exactly one definition to keep in sync with itself.

# Drop any reassembly buffer that has not completed within this time (seconds).
# Default tuned for low-latency telemetry; can be overridden in config.toml
_TEL = _CFG.get("telemetry", {})
REASSEMBLY_TIMEOUT = float(_TEL.get("reassembly_timeout", 0.3))


def run_data_receiver(app_state):
    '''
    Runs a blocking UDP receive loop on DATA_PORT that reassembles
    fragmented msgpack telemetry messages sent by the Air Pi (via
    send_telemetry_udp()/pack_fragments()) and publishes each fully
    reassembled message's object list to app_state.tracking_data.

    Each received datagram is parsed as an 8-byte header (seq, frag_id,
    frag_count) plus payload; datagrams shorter than the header, or whose
    seq is less than or equal to the last fully-processed sequence number,
    are discarded (as runt or stale/duplicate packets). Fragments for a
    given seq are buffered in a per-message dict keyed by frag_id; once
    all frag_count fragments for a seq have arrived, they are joined in
    order and unpacked as msgpack, and the resulting wrapper's 'objects'
    list replaces app_state.tracking_data under
    app_state.tracking_data_lock. Any reassembly buffer whose first
    fragment arrived more than REASSEMBLY_TIMEOUT seconds ago and is still
    incomplete is dropped, so the display continues showing the last
    complete set of bounding boxes rather than a partial/corrupt frame.

    The socket uses a 1-second receive timeout so the loop can check
    app_state.data_thread_stop_event periodically; timeouts are treated as
    normal polling. msgpack decode failures are logged and that message is
    discarded; other exceptions are logged (unless the stop event is
    already set) and followed by a 1-second pause.

    Args:
        app_state (AppState): Shared application state providing
            tracking_data (and its lock) and data_thread_stop_event.

    Returns:
        None: This function does not return a value; it runs until
            app_state.data_thread_stop_event is set.
    '''
    # {seq: {'frags': {frag_id: bytes}, 'total': int, 'first_seen': float}}
    reassembly = {}
    last_processed_seq = -1

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(('', DATA_PORT))
        s.settimeout(1.0)   # allows the stop-event check on every timeout
        logging.info(f"Data receiver listening on port {DATA_PORT}")
        while not app_state.data_thread_stop_event.is_set():
            try:
                data, addr = s.recvfrom(65536)

                # ── Parse header ──────────────────────────────────────────────
                if len(data) < PACKET_HEADER_LEN:
                    continue   # runt packet, ignore
                seq, frag_id, frag_count = struct.unpack(PACKET_HEADER_FMT, data[:PACKET_HEADER_LEN])
                payload = data[PACKET_HEADER_LEN:]

                # ── Discard stale / duplicate datagrams ───────────────────────
                if seq <= last_processed_seq:
                    continue

                # ── Buffer this fragment ──────────────────────────────────────
                if seq not in reassembly:
                    reassembly[seq] = {
                        'frags':      {},
                        'total':      frag_count,
                        'first_seen': time.time(),
                    }
                reassembly[seq]['frags'][frag_id] = payload

                # ── Try to assemble once all fragments have arrived ───────────
                entry = reassembly[seq]
                if len(entry['frags']) == entry['total']:
                    body = b''.join(entry['frags'][i] for i in range(entry['total']))
                    try:
                        wrapper = msgpack.unpackb(body, raw=False)
                        tracking_list = wrapper.get('objects', [])
                        with app_state.tracking_data_lock:
                            app_state.tracking_data = tracking_list
                        last_processed_seq = seq
                    except Exception as e:
                        logging.error(f"Failed to decode telemetry: {e}")
                    del reassembly[seq]

                # ── Drop timed-out incomplete messages ────────────────────────
                # Keeps last-known-good boxes on screen; never renders partials.
                now = time.time()
                stale_seqs = [
                    s for s, e in reassembly.items()
                    if now - e['first_seen'] > REASSEMBLY_TIMEOUT
                ]
                for stale_seq in stale_seqs:
                    del reassembly[stale_seq]

            except socket.timeout:
                continue   # normal timeout, loop again
            except Exception as e:
                if not app_state.data_thread_stop_event.is_set():
                    logging.error(f"Data receiver error: {e}")
                    time.sleep(1)

# -----------------------------------------------------------------------------------------------
# Target Selection (Command Sender)
# -----------------------------------------------------------------------------------------------
def send_command_to_air_pi(command_dict):
    '''
    Opens a short-lived TCP connection to the Air Pi's command server
    (via the Radxa USB gadget address and COMMAND_PORT) and sends
    command_dict as a single JSON-encoded message.

    Args:
        command_dict (dict): The command payload to send, JSON-serialized
            before sending (e.g. {'target_id': <id or None>}).

    Returns:
        None: This function does not return a value. Connection timeouts,
            connection-refused errors, and any other exception are caught
            and logged rather than raised.
    '''
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(2.0) # 2-second timeout
            s.connect((RADXA_GADGET_IP, COMMAND_PORT))
            command_json = json.dumps(command_dict)
            s.sendall(command_json.encode('utf-8'))
            logging.info(f"Sent command to Air Pi: {command_json}")
    except socket.timeout:
        logging.error(f"Error: Connection to Air Pi command server timed out.")
    except ConnectionRefusedError:
        logging.error(f"Error: Connection to Air Pi command server refused.")
    except Exception as e:
        logging.error(f"Error sending command: {e}")

