//! Ties the world, the player and the renderer together each frame.

use crate::block::{self, is_solid, Block, AIR, PLACEABLE, VOXEL_SIZE, WATER};
use crate::budget::{Cost, Deadline};
use crate::chunk::chunk_of;
use crate::far::FarField;
use crate::mesh;
use crate::player::{MoveInput, Player};
use crate::renderer::{sun_view_proj, Globals, Renderer, MAX_LIGHTS, SHADOW_SIZE};
use crate::terrain::{WATER_LEVEL_M, WORLD_CHUNKS_XZ, WORLD_CHUNKS_Y};
use crate::world::World;
use glam::{IVec3, Mat4, Vec3};
use std::collections::HashSet;
use winit::event::MouseButton;
use winit::keyboard::KeyCode;

const REACH_M: f32 = 8.0;
const EDIT_REPEAT_S: f32 = 0.16;
const MOUSE_SENSITIVITY: f32 = 0.0022;
/// Largest brush radius in voxels (a 6.5 m wide brush).
pub const MAX_BRUSH: i32 = 6;

/// The shape and size of the volume one click digs or builds.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Brush {
    /// Radius in voxels; 0 edits a single voxel.
    pub radius: i32,
    pub cube: bool,
}

impl Brush {
    /// Whether a voxel at offset `d` from the brush centre is inside the brush.
    /// Must match `in_brush` in the terrain shader, which previews it.
    pub fn contains(&self, d: IVec3) -> bool {
        let r = self.radius;
        if self.cube {
            d.abs().max_element() <= r
        } else {
            d.length_squared() as f32 <= (r as f32 + 0.35).powi(2)
        }
    }

    /// Offsets of every voxel inside the brush.
    pub fn offsets(self) -> impl Iterator<Item = IVec3> {
        let r = self.radius;
        (-r..=r)
            .flat_map(move |z| (-r..=r).flat_map(move |y| (-r..=r).map(move |x| IVec3::new(x, y, z))))
            .filter(move |d| self.contains(*d))
    }

    /// Width across the brush in metres.
    pub fn width_m(&self) -> f32 {
        (2 * self.radius + 1) as f32 * VOXEL_SIZE
    }
}

/// What the on-screen HUD shows. The browser page redraws it when this changes.
#[derive(Clone, PartialEq, Debug)]
pub struct Hud {
    pub selected: usize,
    pub brush: Brush,
    pub flying: bool,
    /// The material under the crosshair, if any is in reach.
    pub target: Option<Block>,
}

pub struct Settings {
    pub seed: u32,
    /// Radius in chunks drawn as full voxels; the far field covers the rest.
    pub view_radius: i32,
    /// Milliseconds per frame for background work (generating chunks, meshing,
    /// far tiles) once the player can move. Kept small and fixed so loading
    /// never causes a frame spike.
    pub work_budget_ms: f64,
    /// Distance in metres at which the fog has mostly swallowed the terrain.
    pub fog_m: f32,
}

impl Default for Settings {
    fn default() -> Self {
        if cfg!(target_arch = "wasm32") {
            Settings {
                seed: 20260927,
                view_radius: 7,
                work_budget_ms: 6.0,
                fog_m: 1000.0,
            }
        } else {
            Settings {
                seed: 20260927,
                view_radius: 10,
                work_budget_ms: 8.0,
                fog_m: 1300.0,
            }
        }
    }
}

pub struct Game {
    pub world: World,
    pub player: Player,
    far: FarField,
    near_mask: Vec<u8>,
    settings: Settings,
    keys: HashSet<KeyCode>,
    buttons: HashSet<MouseButton>,
    selected: usize,
    brush: Brush,
    edit_timer: f32,
    time: f32,
    fps: f32,
    /// Smoothed CPU milliseconds per frame: world streaming, meshing, far field, rendering.
    cpu_ms: [f32; 4],
    /// How long meshing and uploading one chunk takes.
    mesh_cost: Cost,
    /// 0 in daylight, 1 at night; eases towards `night_wanted`.
    night: f32,
    night_wanted: bool,
    /// Longest frame in the current and the last full second, in milliseconds.
    worst_ms: [f32; 2],
    worst_timer: f32,
    target: Option<IVec3>,
    ready: bool,
}

impl Game {
    pub fn new(settings: Settings) -> Self {
        let world = World::new(settings.seed, settings.view_radius);
        #[allow(unused_mut)]
        let mut player = Player::new(world.terrain.spawn_point());
        #[cfg(target_arch = "wasm32")]
        if let Some([x, y, z, yaw, pitch]) = crate::web::camera_from_url() {
            player.pos = Vec3::new(x, y, z);
            player.yaw = yaw;
            player.pitch = pitch;
            player.flying = true;
        }
        #[cfg(target_arch = "wasm32")]
        let night = crate::web::url_param("night").is_some_and(|v| v != "0");
        #[cfg(not(target_arch = "wasm32"))]
        let night = false;
        Self {
            world,
            player,
            far: FarField::default(),
            near_mask: Vec::new(),
            settings,
            keys: HashSet::new(),
            buttons: HashSet::new(),
            selected: 0,
            brush: Brush { radius: 1, cube: false },
            edit_timer: 0.0,
            time: 0.0,
            fps: 0.0,
            cpu_ms: [0.0; 4],
            mesh_cost: Cost::default(),
            night: night as i32 as f32,
            night_wanted: night,
            worst_ms: [0.0; 2],
            worst_timer: 0.0,
            target: None,
            ready: false,
        }
    }

    pub fn key(&mut self, code: KeyCode, pressed: bool) {
        if pressed && !self.keys.contains(&code) {
            match code {
                KeyCode::KeyF => {
                    self.player.flying = !self.player.flying;
                    self.player.vel = Vec3::ZERO;
                }
                KeyCode::BracketLeft | KeyCode::Minus | KeyCode::NumpadSubtract => self.resize_brush(-1),
                KeyCode::BracketRight | KeyCode::Equal | KeyCode::NumpadAdd => self.resize_brush(1),
                KeyCode::KeyB => self.brush.cube = !self.brush.cube,
                KeyCode::KeyN => self.night_wanted = !self.night_wanted,
                _ => {
                    let digits = [
                        KeyCode::Digit1,
                        KeyCode::Digit2,
                        KeyCode::Digit3,
                        KeyCode::Digit4,
                        KeyCode::Digit5,
                        KeyCode::Digit6,
                        KeyCode::Digit7,
                        KeyCode::Digit8,
                    ];
                    if let Some(i) = digits.iter().position(|d| *d == code) {
                        self.selected = i;
                    }
                }
            }
        }
        if pressed {
            self.keys.insert(code);
        } else {
            self.keys.remove(&code);
        }
    }

    fn resize_brush(&mut self, step: i32) {
        self.brush.radius = (self.brush.radius + step).clamp(0, MAX_BRUSH);
    }

    pub fn mouse_button(&mut self, b: MouseButton, pressed: bool) {
        // Middle click picks up the material under the crosshair.
        if b == MouseButton::Middle {
            if pressed {
                let picked = self.target.map(|v| self.world.get(v));
                if let Some(i) = PLACEABLE.iter().position(|m| Some(*m) == picked) {
                    self.selected = i;
                }
            }
            return;
        }
        if pressed {
            self.buttons.insert(b);
            self.edit_timer = 0.0;
        } else {
            self.buttons.remove(&b);
        }
    }

    pub fn mouse_motion(&mut self, dx: f64, dy: f64) {
        self.player
            .look(dx as f32 * MOUSE_SENSITIVITY, dy as f32 * MOUSE_SENSITIVITY);
    }

    /// The wheel picks materials, or resizes the brush while Alt is held.
    pub fn scroll(&mut self, lines: f32) {
        if self.held(KeyCode::AltLeft) || self.held(KeyCode::AltRight) {
            self.resize_brush(if lines > 0.0 { 1 } else { -1 });
            return;
        }
        let n = PLACEABLE.len() as i32;
        let step = if lines > 0.0 { -1 } else { 1 };
        self.selected = ((self.selected as i32 + step).rem_euclid(n)) as usize;
    }

    pub fn release_input(&mut self) {
        self.keys.clear();
        self.buttons.clear();
    }

    fn held(&self, k: KeyCode) -> bool {
        self.keys.contains(&k)
    }

    pub fn update(&mut self, dt: f32, renderer: &mut Renderer) {
        self.worst_ms[0] = self.worst_ms[0].max(dt * 1000.0);
        self.worst_timer += dt;
        if self.worst_timer >= 1.0 {
            self.worst_timer = 0.0;
            self.worst_ms = [0.0, self.worst_ms[0]];
        }
        let dt = dt.min(0.05);
        // Dusk and dawn take a few seconds.
        let goal = self.night_wanted as i32 as f32;
        self.night += (goal - self.night).clamp(-dt / 4.0, dt / 4.0);
        self.time += dt;
        self.fps = if self.fps == 0.0 {
            1.0 / dt.max(1e-3)
        } else {
            self.fps * 0.95 + 0.05 / dt.max(1e-3)
        };

        let clock = web_time::Instant::now();
        let lap = || clock.elapsed().as_secs_f32() * 1000.0;
        // Until the ground around the player exists they cannot move, so a long
        // frame costs nothing and loading may take most of it. Afterwards all
        // background work shares one small, fixed slice of every frame: a steady
        // frame rate matters more than how fast distant chunks appear.
        let budget = if self.ready { self.settings.work_budget_ms } else { 40.0 };
        let deadline = Deadline::new(budget);
        // Each stage may run until its share of the budget is used; later stages
        // always keep at least the remainder.
        self.world.stream(self.player.pos, &deadline, 0.5);
        let t_stream = lap();
        for c in self.world.removed.drain(..) {
            renderer.remove(c);
        }
        self.remesh(renderer, &deadline, 0.85);
        self.update_near_mask(renderer);
        let t_mesh = lap();
        self.far
            .update(&self.world.terrain, self.player.pos, &deadline, 1.0, |t, m| {
                renderer.upload_far(t, m)
            });
        let t_far = lap();
        self.smooth_cpu(0, t_stream);
        self.smooth_cpu(1, t_mesh - t_stream);
        self.smooth_cpu(2, t_far - t_mesh);

        // Hold the player still until the ground under them exists.
        let here = chunk_of((self.player.pos / VOXEL_SIZE).floor().as_ivec3());
        if !self.ready {
            self.ready = self.world.neighbourhood_ready(here) && self.world.neighbourhood_ready(here - IVec3::Y);
        }
        if self.ready {
            let axis = |pos: KeyCode, neg: KeyCode| (self.held(pos) as i32 - self.held(neg) as i32) as f32;
            let input = MoveInput {
                forward: axis(KeyCode::KeyW, KeyCode::KeyS),
                right: axis(KeyCode::KeyD, KeyCode::KeyA),
                up: self.held(KeyCode::Space) as i32 as f32
                    - (self.held(KeyCode::ControlLeft) || self.held(KeyCode::KeyC)) as i32 as f32,
                jump: self.held(KeyCode::Space),
                sprint: self.held(KeyCode::ShiftLeft) || self.held(KeyCode::ShiftRight),
            };
            self.player.update(&self.world, input, dt);
        }

        let hit = self.world.raycast(self.player.eye(), self.player.look_dir(), REACH_M);
        self.target = hit.as_ref().map(|h| h.voxel);
        self.edit_timer -= dt;
        if self.edit_timer <= 0.0 {
            if let Some(h) = hit {
                if self.buttons.contains(&MouseButton::Left) {
                    self.edit(h.voxel, AIR);
                    self.edit_timer = EDIT_REPEAT_S;
                } else if self.buttons.contains(&MouseButton::Right) {
                    self.edit(h.before, PLACEABLE[self.selected]);
                    self.edit_timer = EDIT_REPEAT_S;
                }
            }
        }
    }

    /// Digs (with AIR) or builds the brush volume around `center`.
    fn edit(&mut self, center: IVec3, block: Block) {
        for d in self.brush.offsets() {
            let v = center + d;
            let current = self.world.get(v);
            if block == AIR {
                if is_solid(current) {
                    // Dug holes below the river line fill with water.
                    let wet = (v.y as f32 + 0.5) * VOXEL_SIZE < WATER_LEVEL_M && self.touches_water(v);
                    self.world.set(v, if wet { WATER } else { AIR });
                }
            } else if !is_solid(current) && !self.player.intersects_voxel(v) {
                self.world.set(v, block);
            }
        }
    }

    fn touches_water(&self, v: IVec3) -> bool {
        [IVec3::X, IVec3::NEG_X, IVec3::Y, IVec3::Z, IVec3::NEG_Z]
            .iter()
            .any(|d| self.world.get(v + *d) == WATER)
    }

    fn remesh(&mut self, renderer: &mut Renderer, deadline: &Deadline, share: f64) {
        let mut built = 0;
        let center = chunk_of((self.player.pos / VOXEL_SIZE).floor().as_ivec3());
        let r = self.world.view_radius;
        let mut todo: Vec<IVec3> = self
            .world
            .dirty
            .iter()
            .copied()
            .filter(|c| {
                let d = *c - center;
                d.x * d.x + d.z * d.z <= r * r
            })
            .collect();
        todo.sort_by_key(|c| {
            let d = *c - center;
            d.x * d.x + d.z * d.z + d.y * d.y
        });
        for c in todo {
            if !self.world.neighbourhood_ready(c) {
                continue;
            }
            if !deadline.fits(&self.mesh_cost, share, built == 0) {
                break;
            }
            let start = web_time::Instant::now();
            let m = mesh::build(&self.world, c);
            renderer.upload(c, &m);
            self.mesh_cost.record(start.elapsed().as_secs_f64() * 1000.0);
            self.world.dirty.remove(&c);
            built += 1;
        }
    }

    /// Tells the renderer which chunk columns show full voxel meshes, so the far
    /// field hides itself there. A column counts once every chunk in it is meshed.
    fn update_near_mask(&mut self, renderer: &mut Renderer) {
        let side = WORLD_CHUNKS_XZ;
        let mut mask = vec![0u8; (side * side) as usize];
        let center = chunk_of((self.player.pos / VOXEL_SIZE).floor().as_ivec3());
        let r = self.world.view_radius;
        for dz in -r..=r {
            for dx in -r..=r {
                let (x, z) = (center.x + dx, center.z + dz);
                if dx * dx + dz * dz > r * r || x < 0 || z < 0 || x >= side || z >= side {
                    continue;
                }
                let meshed = (0..WORLD_CHUNKS_Y).all(|y| {
                    let c = IVec3::new(x, y, z);
                    self.world.chunks.contains_key(&c) && !self.world.dirty.contains(&c)
                });
                mask[(z * side + x) as usize] = meshed as u8;
            }
        }
        if mask != self.near_mask {
            renderer.set_near_mask(&mask);
            self.near_mask = mask;
        }
    }

    /// `width` and `height` are the scene's render size, `scale` that size over the window's.
    pub fn globals(&self, width: u32, height: u32, manual_srgb: bool, scale: f32) -> Globals {
        let aspect = width as f32 / height.max(1) as f32;
        let eye = self.player.eye();
        let proj = Mat4::perspective_infinite_reverse_rh(70f32.to_radians(), aspect, 0.05);
        let view = Mat4::look_to_rh(eye, self.player.look_dir(), Vec3::Y);
        let vp = proj * view;
        // A late-afternoon sun about 26 degrees up (warm light, long shadows),
        // or at night a high moon on the other side of the sky.
        let day_sun = Vec3::new(0.62, 0.36, 0.40).normalize();
        let moon = Vec3::new(-0.45, 0.62, -0.35).normalize();
        let t = self.night * self.night * (3.0 - 2.0 * self.night);
        let sun = day_sun.lerp(moon, t).normalize();
        let mut lights = [[0.0; 4]; MAX_LIGHTS];
        let near = self.world.nearest_lanterns(eye, MAX_LIGHTS);
        for (slot, p) in lights.iter_mut().zip(&near) {
            *slot = p.extend(1.0).to_array();
        }
        let fog = self.settings.fog_m;
        let underwater = self.world.get((eye / VOXEL_SIZE).floor().as_ivec3()) == WATER;
        let (hl, has) = match self.target {
            Some(v) => (v.as_vec3() * VOXEL_SIZE, 1.0 + self.brush.radius as f32),
            None => (Vec3::ZERO, 0.0),
        };
        Globals {
            view_proj: vp.to_cols_array_2d(),
            inv_view_proj: vp.inverse().to_cols_array_2d(),
            sun_view_proj: sun_view_proj(eye, sun).to_cols_array_2d(),
            camera_pos: eye.extend(1.0).to_array(),
            sun_dir: sun.extend(self.time).to_array(),
            params: [
                fog,
                manual_srgb as i32 as f32,
                underwater as i32 as f32,
                self.brush.cube as i32 as f32,
            ],
            highlight: hl.extend(has).to_array(),
            screen: [width as f32, height as f32, SHADOW_SIZE as f32, scale],
            sky: [t, near.len() as f32, 0.0, 0.0],
            lights,
        }
    }

    pub fn hud(&self) -> Hud {
        Hud {
            selected: self.selected,
            brush: self.brush,
            flying: self.player.flying,
            target: self.target.map(|v| self.world.get(v)),
        }
    }

    fn smooth_cpu(&mut self, i: usize, ms: f32) {
        self.cpu_ms[i] = self.cpu_ms[i] * 0.9 + ms * 0.1;
    }

    /// Records how long the renderer spent recording and submitting a frame.
    pub fn note_render_time(&mut self, ms: f32) {
        self.smooth_cpu(3, ms);
    }

    /// Where the frame time goes, for the status line.
    fn timings(&self, renderer: &Renderer) -> String {
        let (w, h) = renderer.render_size();
        let c = self.cpu_ms;
        let mut s = format!(
            "{} | {w}x{h} | cpu {:.1} ms (world {:.1}, mesh {:.1}, far {:.1}, draw {:.1})",
            renderer.adapter_name,
            c.iter().sum::<f32>(),
            c[0],
            c[1],
            c[2],
            c[3]
        );
        match renderer.gpu_ms {
            Some(g) => {
                s += &format!(
                    " | gpu {:.1} ms (shadow {:.1}, scene {:.1}, water {:.1}, post {:.1})",
                    g.iter().sum::<f32>(),
                    g[0],
                    g[1],
                    g[2],
                    g[3]
                )
            }
            None => s += " | gpu timing unavailable",
        }
        s
    }

    pub fn status(&self, renderer: &Renderer) -> String {
        format!(
            "Strata | {:.0} fps, worst frame {:.0} ms | {} | brush {} | {}{} | {} chunks, {} far tiles | {}k + {}k far tris | {}",
            self.fps,
            self.worst_ms[1].max(self.worst_ms[0]),
            block::name(PLACEABLE[self.selected]),
            self.brush.radius,
            if self.player.flying { "flying" } else { "walking" },
            if self.ready { "" } else { " (loading)" },
            renderer.mesh_count(),
            self.far.tile_count(),
            renderer.triangles_drawn / 1000,
            renderer.far_triangles_drawn / 1000,
            self.timings(renderer),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn brush_volumes() {
        let count = |radius, cube| Brush { radius, cube }.offsets().count();
        assert_eq!(count(0, false), 1);
        assert_eq!(count(0, true), 1);
        assert_eq!(count(1, false), 7);
        assert_eq!(count(1, true), 27);
        assert_eq!(count(2, true), 125);
        assert!(count(MAX_BRUSH, false) < count(MAX_BRUSH, true));
        assert_eq!(Brush { radius: 2, cube: false }.width_m(), 2.5);
    }
}
