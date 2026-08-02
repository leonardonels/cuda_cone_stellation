# cuda_cone_stellation

Local planner for autocross: takes a SLAM cone map, triangulates it, and grows a
centerline (the *Way*) one midpoint at a time by searching the Delaunay
midpoints.

Almost all of the runtime is that search. It is available in three
interchangeable implementations that produce **byte-identical Ways** and differ
only in how they get there.

## Backends

Selected by the `autocross/search_backend` parameter.

| `search_backend` | wall / callback | CPU / callback | core @ 20 Hz | what it is |
|---|---|---|---|---|
| `cpu` | 46.5 ms | 46.5 ms | 93% | frozen reference. Kept to validate the others against; too slow. |
| `cpu-fast` *(new default)* | 11.8 ms | 11.6 ms | 23% | Lowest latency. |
| `cuda` | 16.3 ms | **2.8 ms** | **5.6%** | device backend. Higher latency, ~4× less CPU usage. |

Measured on 2026 Cremona Rosbags, 1500 callbacks (~3 laps), `float32`. Bags 7, 8 and 13
agree within ~10%. All three produced identical path digests on all four bags.

**Which to pick:** `cpu-fast` if the constraint is how fresh the centerline is;
`cuda` if the constraint is CPU time — it blocks on the device rather than doing
the work, handing the core back to whatever else runs on the Orin.

## Diagnostics

Two parameters, both off by default, because both cost real time rather than
just deciding whether to print.

| parameter | what it does |
|---|---|
| `autocross/logging` | the pipeline timing line (`computation has taken: X ms`) on **every** callback. At 20 Hz, producing that line costs more than the search now does. |
| `autocross/debug` | the search statistics report **once**, right after the completed trajectory is published: wall and CPU per callback, midpoints, BFS nodes, candidates, and with `cuda` its host/device phase split. |

`debug` reports at the end rather than periodically because the figures are
cumulative and computing them sorts the whole callback history — on a timer that
would pay the same cost repeatedly for the same answer. Warnings and errors are
unaffected by either.

## Building

```bash
colcon build --packages-select cuda_cone_stellation
```

| CMake option | default | meaning |
|---|---|---|
| `USE_CUDA` | `OFF` | build the `cuda` backend. Needs a CUDA toolkit; falls back with a warning if absent. |
| `CCS_CPU_SCALAR_FLOAT32` | `ON` | run the host search in `float32` instead of `double`. Per-backend — the device backend is always `float32`. |
| `BUILD_BENCH` | `OFF` | build `search_bench` (below). Adds a `rosbag2` dependency, so it is off for flight builds. |

## Architecture

The search is split so that the backends can differ where they should and
cannot where they must not:

- **Layout** — [`structures/search_types.hpp`](include/cuda_cone_stellation/structures/search_types.hpp).
  Flat SoA, templated on the scalar type.
- **Policy** — [`structures/search_policy.hpp`](include/cuda_cone_stellation/structures/search_policy.hpp).
  The six discard filters, the heuristic, loop closure. **Shared and
  non-negotiable**: two backends disagreeing here do not disagree by a rounding
  error, they drive different racing lines.
- **Strategy** — each backend. Spatial index, shape of the tree walk, what runs
  in parallel. **Not shared**: a CPU and a GPU want genuinely different answers,
  and forcing one shape on both is how a GPU port inherits a host bottleneck.

Everything enters through [`ISearch`](include/cuda_cone_stellation/modules/isearch.hpp),
which also defines the counters every backend must maintain, so two of them
replayed on the same bag are directly comparable.

## Validating a change

`search_bench` replays a bag through the real planner deterministically and
writes a per-callback digest of the published path.

```bash
colcon build --packages-select cuda_cone_stellation --cmake-args -DBUILD_BENCH=ON -DUSE_CUDA=ON

BENCH=install/cuda_cone_stellation/lib/cuda_cone_stellation/search_bench
YAML=install/cuda_cone_stellation/share/cuda_cone_stellation/config/cuda_cone_stellation.yaml

for backend in cpu cpu-fast cuda; do
  $BENCH ~/logs/july-2026/bags/rosbag__6 /tmp/d_$backend.txt 1500 \
      --ros-args --params-file $YAML -p autocross/search_backend:=$backend
done
diff /tmp/d_cpu.txt /tmp/d_cpu-fast.txt && diff /tmp/d_cpu.txt /tmp/d_cuda.txt
```

Identical files mean the change altered no path on that bag. `CCS_BENCH_DUMP_PATHS=<file>`
additionally dumps the raw points, for when two backends *cannot* be identical
and the question becomes how far apart they are.

**Do not use `ros2 bag play` into the live node for this.** Cones arrive at
20 Hz on a `KeepLast(1)` subscription while odometry arrives at ~100 Hz, so
which odometry is "latest" at each cone callback depends on scheduling: two runs
of the *same* backend produce different Ways, and a slow backend additionally
drops cone messages a fast one does not. `search_bench` reads the bag in
timestamp order and delivers every message, so the input is a function of the
bag alone.

1500 callbacks is ~3 laps and is the standard length; a truncated run is
byte-identical to the prefix of a full one, but it is *not* a scaled-down
version of it, since the Way grows with the cone map.

## Notes

- The 50 ms tree-search time limit never fires on any of the four bags
  (`timeLimitHits 0`), which is what makes the output a pure function of the
  input and bit-exact validation possible at all.
- The device backend refuses configurations whose BFS could exceed its
  fixed-capacity frontier (`max_search_options^max_search_tree_height`) rather
  than truncating, and falls back to the host backend with a warning.
- In the deployed node, `Time::tock()` logs one `INFO` per callback and the
  visualization markers are published unconditionally; together those cost more
  than the search itself now does. `autocross/publish_markers` turns off the
  latter.
