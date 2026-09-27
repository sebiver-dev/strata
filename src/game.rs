//! Ties the world, the player and the renderer together each frame.

use crate::block::{self, is_solid, Block, AIR, PLACEABLE, VOXEL_SIZE, WATER};
use crate::chunk::{chunk_of, CHUNK};
use crate::mesh;
use crate::player::{MoveInput, Player};
use crate::renderer::{sun_view_proj, Globals, Renderer, SHADOW_SIZE};
use crate::terrain::WATER_LEVEL_M;
use crate::world::World;
use glam::{IVec3, Mat4, Vec3};
use std::collections::HashSet;
use winit::event::MouseButton;
use winit::keyboard::KeyCode;

const REACH_M: f32 = 8.0;
const EDIT_REPEAT_S: f32 = 0.16;
const MOUSE_SENSITIVITY: f32 = 0.0022;

pub struct Settings {
    pub seed: u32,
    pub view_radius: i32,
    pub stream_budget_ms: f64,
    pub mesh_budget_ms: f64,
}

impl Default for Settings {
    fn default() -> Self {
        if cfg!(target_arch = "wasm32") {
            Settings {
                seed: 20260927,
                view_radius: 9,
                stream_budget_ms: 5.0,
                mesh_budget_ms: 4.0,
            }
        } else {
            Settings {
                seed: 20260927,
                view_radius: 14,
                stream_budget_ms: 7.0,
                mesh_budget_ms: 6.0,
            }
        }
    }
}

pub struct Game {
    pub world: World,
    pub player: Player,
    settings: Settings,
    keys: HashSet<KeyCode>,
    buttons: HashSet<MouseButton>,
    selected: usize,
    brush: i32,
    edit_timer: f32,
    time: f32,
    fps: f32,
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
        Self {
            world,
            player,
            settings,
            keys: HashSet::new(),
            buttons: HashSet::new(),
            selected: 0,
            brush: 1,
            edit_timer: 0.0,
            time: 0.0,
            fps: 0.0,
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
                KeyCode::BracketLeft => self.brush = (self.brush - 1).max(0),
                KeyCode::BracketRight => self.brush = (self.brush + 1).min(4),
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

    pub fn mouse_button(&mut self, b: MouseButton, pressed: bool) {
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

    pub fn scroll(&mut self, lines: f32) {
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
        let dt = dt.min(0.05);
        self.time += dt;
        self.fps = if self.fps == 0.0 {
            1.0 / dt.max(1e-3)
        } else {
            self.fps * 0.95 + 0.05 / dt.max(1e-3)
        };

        self.world.stream(self.player.pos, self.settings.stream_budget_ms);
        for c in self.world.removed.drain(..) {
            renderer.remove(c);
        }
        self.remesh(renderer);

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

    /// Digs (with AIR) or builds a sphere of voxels around `center`.
    fn edit(&mut self, center: IVec3, block: Block) {
        let r = self.brush;
        for dz in -r..=r {
            for dy in -r..=r {
                for dx in -r..=r {
                    let d = IVec3::new(dx, dy, dz);
                    if d.length_squared() as f32 > (r as f32 + 0.35).powi(2) {
                        continue;
                    }
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
        }
    }

    fn touches_water(&self, v: IVec3) -> bool {
        [IVec3::X, IVec3::NEG_X, IVec3::Y, IVec3::Z, IVec3::NEG_Z]
            .iter()
            .any(|d| self.world.get(v + *d) == WATER)
    }

    fn remesh(&mut self, renderer: &mut Renderer) {
        let start = web_time::Instant::now();
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
            let m = mesh::build(&self.world, c);
            renderer.upload(c, &m);
            self.world.dirty.remove(&c);
            if start.elapsed().as_secs_f64() * 1000.0 > self.settings.mesh_budget_ms {
                break;
            }
        }
    }

    pub fn globals(&self, width: u32, height: u32, manual_srgb: bool) -> Globals {
        let aspect = width as f32 / height.max(1) as f32;
        let eye = self.player.eye();
        let proj = Mat4::perspective_infinite_reverse_rh(70f32.to_radians(), aspect, 0.05);
        let view = Mat4::look_to_rh(eye, self.player.look_dir(), Vec3::Y);
        let vp = proj * view;
        // A mid-afternoon sun, low enough for trees to cast long shadows.
        let sun = Vec3::new(0.50, 0.58, 0.30).normalize();
        let fog = self.world.view_radius as f32 * CHUNK as f32 * VOXEL_SIZE * 0.75;
        let underwater = self.world.get((eye / VOXEL_SIZE).floor().as_ivec3()) == WATER;
        let (hl, has) = match self.target {
            Some(v) => (v.as_vec3() * VOXEL_SIZE, 1.0),
            None => (Vec3::ZERO, 0.0),
        };
        Globals {
            view_proj: vp.to_cols_array_2d(),
            inv_view_proj: vp.inverse().to_cols_array_2d(),
            sun_view_proj: sun_view_proj(eye, sun).to_cols_array_2d(),
            camera_pos: eye.extend(1.0).to_array(),
            sun_dir: sun.extend(self.time).to_array(),
            params: [fog, manual_srgb as i32 as f32, underwater as i32 as f32, 0.0],
            highlight: hl.extend(has).to_array(),
            screen: [width as f32, height as f32, SHADOW_SIZE as f32, 0.0],
        }
    }

    pub fn status(&self, renderer: &Renderer) -> String {
        format!(
            "Strata | {:.0} fps | {} | brush {} | {}{} | {} chunks drawn | {}k tris",
            self.fps,
            block::name(PLACEABLE[self.selected]),
            self.brush,
            if self.player.flying { "flying" } else { "walking" },
            if self.ready { "" } else { " (loading)" },
            renderer.mesh_count(),
            renderer.triangles_drawn / 1000,
        )
    }
}
