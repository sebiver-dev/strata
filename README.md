# Strata

A living voxel world on its own engine, for desktop and the browser. Every
surface, colour and shape is generated from code. The game ships no textures,
models or sound files.

![The river valley at spawn, running in a browser](docs/first-light.png)

This is the first engine slice. It covers the parts every later feature
depends on:

- A 2 × 2 km river valley between ridged mountains, generated from a seed.
  It has caves, beaches, gravel river beds, snow caps and forests of round
  and conifer trees.
- Half-metre voxels, twice as fine as Minecraft's, stored in 32³ chunks that
  collapse to a single value when uniform.
- Chunks stream in and out around the player within a per-frame time budget,
  so loading never stalls a frame.
- Meshing emits only visible faces, with per-corner ambient occlusion.
- A wgpu renderer with a procedural sky, sun shadows, HDR rendering with
  filmic tone mapping, and distance fog. It uses reverse-Z depth and frustum
  culling.
- Textureless procedural materials, each with its own colour, relief,
  roughness and translucency: lichen and cracks on stone, pebbles in dirt,
  wind ripples in sand, clumped leaves that glow when backlit and sway in the
  wind, and centimetre-sized flowers in the grass that give the ground a
  human scale.
- Water that refracts and tints what lies beneath it by depth, reflects the
  terrain with screen-space reflections (falling back to the sky), and foams
  along the shore.
- First-person walking with gravity, auto step-up onto half-metre ledges,
  swimming and a fly mode.
- Digging and building with a sphere brush. Holes dug below the river line
  fill with water.

## Running

Desktop (Windows, macOS, Linux):

```sh
cargo run --release
```

Browser (needs a WebGPU browser such as current Chrome, Edge, Safari or Firefox):

```sh
cargo install wasm-bindgen-cli --version 0.2.129   # once; must match Cargo.lock
./scripts/build-web.sh
python3 -m http.server -d web 8080                  # then open http://localhost:8080
```

Add `?cam=x,y,z,yaw,pitch` to the page URL to start flying at a fixed camera
(metres and radians), which is handy for before-and-after screenshots.

## Controls

| Input | Action |
| --- | --- |
| Click | Capture the mouse |
| W A S D | Move |
| Space | Jump, swim up, or rise when flying |
| Shift | Sprint (fly faster when flying) |
| C or Ctrl | Descend when flying |
| F | Toggle flying |
| Left mouse | Dig |
| Right mouse | Build with the selected material |
| 1 to 8 or mouse wheel | Pick material |
| `[` and `]` | Shrink or grow the brush |
| Esc | Release the mouse |

## Layout

| File | Purpose |
| --- | --- |
| `src/noise.rs` | Seeded hash and value noise, fractal and ridged variants |
| `src/terrain.rs` | World generation: valley, river, caves, materials, trees |
| `src/chunk.rs` | 32³ voxel chunks with uniform-chunk compression |
| `src/world.rs` | Loaded chunks, streaming, edits and voxel ray casts |
| `src/mesh.rs` | Face-culled meshing with ambient occlusion |
| `src/player.rs` | Movement and collision |
| `src/renderer.rs` | wgpu setup, pipelines, per-chunk buffers, culling |
| `src/shaders/world.wgsl` | Sky, procedural materials, water and crosshair |
| `src/game.rs` | Per-frame glue: input, streaming, meshing, edits |
| `src/app.rs` | Window and event loop for both targets |
| `src/web.rs` | Browser entry point |

`cargo run --release --example worldgen_bench` reports how long generating
and meshing the area around spawn takes.

## Next steps

1. Far-field level of detail, so the whole valley is visible without
   meshing every chunk at full resolution.
2. Generation and meshing on worker threads. Browsers need cross-origin
   isolation headers for this.
3. Flowing water that pools, floods and can be dammed.
4. A day cycle, weather and seasons.
5. Co-op: a second player joins from a link and sees every edit live.
