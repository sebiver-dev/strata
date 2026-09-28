//! First-person movement: walking with gravity, stepping up half-metre ledges,
//! swimming, and a free-flying mode.

use crate::block::{is_solid, VOXEL_SIZE, WATER};
use crate::world::World;
use glam::{IVec3, Vec3};

pub const HALF_WIDTH: f32 = 0.3;
pub const HEIGHT: f32 = 1.75;
pub const EYE: f32 = 1.6;
const GRAVITY: f32 = 24.0;
const JUMP_SPEED: f32 = 7.8;
const WALK: f32 = 4.6;
const SPRINT: f32 = 8.0;
const FLY: f32 = 18.0;
const FLY_FAST: f32 = 60.0;

#[derive(Default, Clone, Copy)]
pub struct MoveInput {
    pub forward: f32,
    pub right: f32,
    pub up: f32,
    pub jump: bool,
    pub sprint: bool,
}

pub struct Player {
    /// Feet position in metres.
    pub pos: Vec3,
    pub vel: Vec3,
    pub yaw: f32,
    pub pitch: f32,
    pub flying: bool,
    pub on_ground: bool,
    pub in_water: bool,
}

impl Player {
    pub fn new(pos: Vec3) -> Self {
        Self {
            pos,
            vel: Vec3::ZERO,
            yaw: crate::vista::SPAWN_YAW,
            pitch: -0.15,
            flying: false,
            on_ground: false,
            in_water: false,
        }
    }

    pub fn eye(&self) -> Vec3 {
        self.pos + Vec3::Y * EYE
    }

    pub fn look_dir(&self) -> Vec3 {
        Vec3::new(
            self.yaw.cos() * self.pitch.cos(),
            self.pitch.sin(),
            self.yaw.sin() * self.pitch.cos(),
        )
    }

    pub fn look(&mut self, dx: f32, dy: f32) {
        self.yaw += dx;
        self.pitch = (self.pitch - dy).clamp(-1.55, 1.55);
    }

    fn bounds(pos: Vec3) -> (Vec3, Vec3) {
        (
            pos - Vec3::new(HALF_WIDTH, 0.0, HALF_WIDTH),
            pos + Vec3::new(HALF_WIDTH, HEIGHT, HALF_WIDTH),
        )
    }

    /// Voxels the body overlaps at `pos`.
    fn overlapping(pos: Vec3) -> impl Iterator<Item = IVec3> {
        let (lo, hi) = Self::bounds(pos);
        let a = (lo / VOXEL_SIZE).floor().as_ivec3();
        let b = ((hi / VOXEL_SIZE) - Vec3::splat(1e-4)).floor().as_ivec3();
        (a.y..=b.y).flat_map(move |y| (a.z..=b.z).flat_map(move |z| (a.x..=b.x).map(move |x| IVec3::new(x, y, z))))
    }

    pub fn collides(world: &World, pos: Vec3) -> bool {
        Self::overlapping(pos).any(|v| is_solid(world.get(v)))
    }

    /// True when a voxel would overlap the body (used to stop players burying themselves).
    pub fn intersects_voxel(&self, v: IVec3) -> bool {
        Self::overlapping(self.pos).any(|o| o == v)
    }

    pub fn update(&mut self, world: &World, input: MoveInput, dt: f32) {
        let forward = Vec3::new(self.yaw.cos(), 0.0, self.yaw.sin());
        let right = Vec3::new(-forward.z, 0.0, forward.x);
        let mut wish = forward * input.forward + right * input.right;
        if wish.length_squared() > 1.0 {
            wish = wish.normalize();
        }
        let probe = self.pos + Vec3::Y * 0.9;
        self.in_water = world.get((probe / VOXEL_SIZE).floor().as_ivec3()) == WATER;

        if self.flying {
            let speed = if input.sprint { FLY_FAST } else { FLY };
            let target = (wish + Vec3::Y * input.up) * speed;
            self.vel = self.vel.lerp(target, (dt * 10.0).min(1.0));
            // Flying still respects terrain so you cannot see inside rock.
            self.move_and_collide(world, dt);
            return;
        }

        let speed = if self.in_water {
            WALK * 0.55
        } else if input.sprint {
            SPRINT
        } else {
            WALK
        };
        let control = if self.on_ground || self.in_water { 14.0 } else { 3.0 };
        let target = wish * speed;
        let blend = (dt * control).min(1.0);
        self.vel.x += (target.x - self.vel.x) * blend;
        self.vel.z += (target.z - self.vel.z) * blend;

        if self.in_water {
            self.vel.y -= GRAVITY * 0.25 * dt;
            if input.jump {
                self.vel.y = (self.vel.y + 30.0 * dt).min(3.5);
            }
            self.vel.y *= 1.0 - (dt * 2.0).min(1.0);
        } else {
            self.vel.y -= GRAVITY * dt;
            if input.jump && self.on_ground {
                self.vel.y = JUMP_SPEED;
            }
        }
        self.vel.y = self.vel.y.max(-55.0);
        self.move_and_collide(world, dt);
    }

    fn move_and_collide(&mut self, world: &World, dt: f32) {
        let delta = self.vel * dt;
        // Sub-step so fast falls cannot tunnel through a half-metre floor.
        let steps = (delta.abs().max_element() / 0.2).ceil().max(1.0) as i32;
        let step = delta / steps as f32;
        self.on_ground = false;
        for _ in 0..steps {
            self.axis_move(world, Vec3::new(0.0, step.y, 0.0), 1);
            let before = self.pos;
            let hit_x = self.axis_move(world, Vec3::new(step.x, 0.0, 0.0), 0);
            let hit_z = self.axis_move(world, Vec3::new(0.0, 0.0, step.z), 2);
            // Walk up single-voxel ledges instead of stopping dead.
            if (hit_x || hit_z) && self.on_ground && !self.flying {
                let lifted = before + Vec3::Y * (VOXEL_SIZE + 0.01);
                let target = lifted + Vec3::new(step.x, 0.0, step.z);
                if !Self::collides(world, lifted) && !Self::collides(world, target) {
                    self.pos = target;
                }
            }
        }
    }

    /// Moves along one axis; on contact, snaps flush to the blocking voxel face.
    fn axis_move(&mut self, world: &World, d: Vec3, axis: usize) -> bool {
        if d[axis] == 0.0 {
            return false;
        }
        let next = self.pos + d;
        let hits: Vec<IVec3> = Self::overlapping(next).filter(|v| is_solid(world.get(*v))).collect();
        if hits.is_empty() {
            self.pos = next;
            return false;
        }
        let ext = [HALF_WIDTH, 0.0, HALF_WIDTH][axis];
        if d[axis] > 0.0 {
            let face = hits.iter().map(|v| v[axis]).min().unwrap() as f32 * VOXEL_SIZE;
            let top = if axis == 1 { HEIGHT } else { ext };
            self.pos[axis] = self.pos[axis].max(face - top - 1e-3).min(next[axis]);
        } else {
            let face = (hits.iter().map(|v| v[axis]).max().unwrap() + 1) as f32 * VOXEL_SIZE;
            self.pos[axis] = self.pos[axis].min(face + ext + 1e-3).max(next[axis]);
            if axis == 1 {
                self.on_ground = true;
            }
        }
        self.vel[axis] = 0.0;
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::STONE;
    use crate::chunk::{Chunk, CHUNK};

    fn floor_world() -> World {
        let mut w = World::new(1, 1);
        for cx in -1..=1 {
            for cz in -1..=1 {
                let mut c = Chunk::default();
                for z in 0..CHUNK {
                    for x in 0..CHUNK {
                        for y in 0..4 {
                            c.set(x, y, z, STONE);
                        }
                    }
                }
                w.chunks.insert(IVec3::new(cx, 0, cz), c);
            }
        }
        w
    }

    #[test]
    fn falls_and_lands_on_floor() {
        let w = floor_world();
        let mut p = Player::new(Vec3::new(8.0, 10.0, 8.0));
        for _ in 0..200 {
            p.update(&w, MoveInput::default(), 1.0 / 60.0);
        }
        assert!(p.on_ground);
        assert!((p.pos.y - 2.0).abs() < 0.01, "feet at {}", p.pos.y);
    }

    #[test]
    fn steps_up_a_half_metre_ledge() {
        let mut w = floor_world();
        for z in 0..40 {
            for x in 20..64 {
                w.set(IVec3::new(x, 4, z), STONE);
            }
        }
        let mut p = Player::new(Vec3::new(8.0, 2.0, 8.0));
        p.yaw = 0.0; // facing +X
        for _ in 0..120 {
            p.update(
                &w,
                MoveInput {
                    forward: 1.0,
                    ..Default::default()
                },
                1.0 / 60.0,
            );
        }
        assert_eq!(w.get(IVec3::new(24, 4, 16)), STONE);
        assert!(p.pos.x > 12.0, "stuck at x={}", p.pos.x);
        assert!((p.pos.y - 2.5).abs() < 0.05, "feet at {}", p.pos.y);
    }
}
