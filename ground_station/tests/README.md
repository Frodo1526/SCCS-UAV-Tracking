# ground_station/tests

Intentionally empty for now. Unlike `common/` and `drone_air/command_control.py`,
none of `ground_station`'s own logic is currently pure/hardware-independent:

- `video.py` is GStreamer-coupled end to end.
- `data_cmd.py`'s `run_data_receiver` is a blocking socket loop.
- `ui.py`'s `select_target_by_click` unconditionally calls
  `send_command_to_air_pi`, which opens a real TCP connection (with a
  2-second timeout) -- calling it in a unit test would either need a mock
  Air Pi command server or a refactor to inject the send function, neither
  of which this pass attempts.

If `send_command_to_air_pi` is ever refactored to accept an injectable
sender (e.g. `select_target_by_click(click_pos, tracking_data, app_state,
send_fn=send_command_to_air_pi)`), the bbox-hit-testing logic in
`select_target_by_click` becomes straightforward to test purely -- that's
the natural first target once someone picks this up.
