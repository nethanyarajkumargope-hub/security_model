# SDN feature compatibility audit

The reference SDN simulator exposes aggregate edge and node telemetry. It does not currently
produce the bidirectional flow records expected by the UNSW-NB15 security model.

## Available reference telemetry

Edge telemetry:

`timestep`, `source_node`, `destination_node`, `bandwidth_capacity`,
`bandwidth_utilization`, `delay`, `packet_loss`, `jitter`, `throughput`,
`queue_length`, `packet_rate`.

Node telemetry:

`timestep`, `node`, `incoming_traffic`, `outgoing_traffic`, `queue_size`,
`flow_count`, `traffic_volume`, `node_load`.

These values describe topology links or nodes at a timestep. They are not interchangeable with
UNSW-NB15 flow features merely because names such as rate, loss, load, or jitter look similar.

## UNSW-NB15 feature assessment

| Features | Status | Requirement |
| --- | --- | --- |
| `dur`, `spkts`, `dpkts`, `sbytes`, `dbytes`, `rate` | Collectable | Per-flow start/end time plus directional packet and byte counters |
| `proto`, `service`, `state` | Collectable | Flow match fields, transport ports, and connection-state tracking |
| `sttl`, `dttl` | Collectable | Packet inspection or switch/controller metadata that preserves TTL |
| `sload`, `dload`, `sinpkt`, `dinpkt`, `sjit`, `djit`, `smean`, `dmean` | Derivable | Directional per-flow counters and timestamps over a defined observation window |
| `sloss`, `dloss` | Derivable with limits | Directional loss estimation; aggregate link `packet_loss` is not equivalent |
| `swin`, `dwin`, `stcpb`, `dtcpb`, `tcprtt`, `synack`, `ackdat` | Requires TCP inspection | TCP headers, sequence state, and handshake timing |
| `trans_depth`, `response_body_len` | Requires application inspection | HTTP transaction parsing |
| `ct_srv_src`, `ct_state_ttl`, `ct_dst_ltm`, `ct_src_dport_ltm`, `ct_dst_sport_ltm`, `ct_dst_src_ltm`, `ct_src_ltm`, `ct_srv_dst` | Requires stateful history | Rolling connection counters matching the UNSW-NB15 time-window definitions |
| `is_ftp_login`, `ct_ftp_cmd`, `ct_flw_http_mthd` | Requires application inspection | FTP/HTTP protocol parsing and rolling counters |
| `is_sm_ips_ports` | Derivable | Source/destination IP and port equality from flow identifiers |

## Decision

No current SDN telemetry column is a verified drop-in replacement for the 42 model predictors.
Training on renamed QoS columns would create a model that appears integrated but receives data
with different meaning and scale.

Keep the current 42-feature Random Forest as the offline UNSW-NB15 benchmark. For later SDN use,
add a per-flow collector first, define its observation window and units, then train a separate
deployable model using only the fields that collector can reproduce consistently.

The minimum useful collector record is:

`flow_id`, `source`, `destination`, `source_port`, `destination_port`, `protocol`,
`start_time`, `end_time`, directional packet counts, directional byte counts, and TCP state when
available.
