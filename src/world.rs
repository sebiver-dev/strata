//! The loaded part of the world: chunk storage, streaming around the player,
//! voxel edits and ray casts.

use crate::block::*;
use crate::chunk::{chunk_of, local_of, Chunk, CHUNK};
use crate::terrain::{Terrain, WORLD_CHUNKS_Y};
use glam::{IVec3, Vec3};
use std::collections::{HashMap, HashSet};

pub struct World {
    pub terrain: Terrain,
    pub chunks: HashMap<IVec3, Chunk>,
    /// Chunks whose mesh must be rebuilt.
    pub dirty: HashSet<IVec3>,
    /// Chunks that no longer exist and whose meshes must be dropped.
    pub removed: Vec<IVec3>,
    /// Horizontal load radius in chunks.
    pub view_radius: i32,
}

pub struct RayHit {
    pub voxel: IVec3,
    /// Empty voxel in front of the hit face, where a new block would go.
    pub before: IVec3,
    pub distance: f32,
}

impl World {
    pub fn new(seed: u32, view_radius: i32) -> Self {
        Self {
            terrain: Terrain::new(seed),
            chunks: HashMap::new(),
            dirty: HashSet::new(),
            removed: Vec::new(),
            view_radius,
        }
    }

    #[inline]
    pub fn get(&self, v: IVec3) -> Block {
        if v.y < 0 {
            return STONE;
        }
        match self.chunks.get(&chunk_of(v)) {
            Some(c) => {
                let l = local_of(v);
                c.get(l.x, l.y, l.z)
            }
            None => AIR,
        }
    }

    pub fn is_loaded(&self, c: IVec3) -> bool {
        c.y < 0 || c.y >= WORLD_CHUNKS_Y || self.chunks.contains_key(&c)
    }

    /// Sets one voxel and marks every chunk whose mesh can see it as dirty.
    pub fn set(&mut self, v: IVec3, b: Block) -> bool {
        if v.y < 1 || v.y >= WORLD_CHUNKS_Y * CHUNK {
            return false;
        }
        let c = chunk_of(v);
        let Some(chunk) = self.chunks.get_mut(&c) else {
            return false;
        };
        let l = local_of(v);
        if chunk.get(l.x, l.y, l.z) == b {
            return false;
        }
        chunk.set(l.x, l.y, l.z, b);
        // Neighbours share faces and ambient occlusion with voxels on the border.
        for dz in -1..=1 {
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let off = IVec3::new(dx, dy, dz);
                    let edge = (l + off).cmplt(IVec3::ZERO).any() || (l + off).cmpge(IVec3::splat(CHUNK)).any();
                    if off == IVec3::ZERO || edge {
                        let n = chunk_of(v + off);
                        if self.chunks.contains_key(&n) {
                            self.dirty.insert(n);
                        }
                    }
                }
            }
        }
        true
    }

    /// Chunk coordinates within the view radius, nearest first.
    fn wanted(&self, center: IVec3, radius: i32) -> Vec<IVec3> {
        let mut out = Vec::new();
        for y in 0..WORLD_CHUNKS_Y {
            for z in -radius..=radius {
                for x in -radius..=radius {
                    if x * x + z * z <= radius * radius {
                        out.push(IVec3::new(center.x + x, y, center.z + z));
                    }
                }
            }
        }
        out.sort_by_key(|c| {
            let d = *c - center;
            d.x * d.x + d.z * d.z + (d.y * d.y) / 2
        });
        out
    }

    /// Generates missing chunks near the player within a time budget and drops far ones.
    /// Returns the number of chunks generated.
    pub fn stream(&mut self, player: Vec3, budget_ms: f64) -> usize {
        let start = web_time::Instant::now();
        let center = chunk_of((player / VOXEL_SIZE).floor().as_ivec3());
        // Load one ring further than we mesh so border faces are known.
        let load_r = self.view_radius + 1;

        let unload_r2 = (load_r + 2) * (load_r + 2);
        let far: Vec<IVec3> = self
            .chunks
            .keys()
            .filter(|c| {
                let d = **c - center;
                d.x * d.x + d.z * d.z > unload_r2
            })
            .copied()
            .collect();
        for c in far {
            self.chunks.remove(&c);
            self.dirty.remove(&c);
            self.removed.push(c);
        }

        let mut generated = 0;
        for c in self.wanted(center, load_r) {
            if self.chunks.contains_key(&c) {
                continue;
            }
            let chunk = self.terrain.generate(c);
            self.chunks.insert(c, chunk);
            generated += 1;
            // A new chunk may complete the neighbourhood of chunks around it.
            for d in [
                IVec3::X,
                IVec3::NEG_X,
                IVec3::Y,
                IVec3::NEG_Y,
                IVec3::Z,
                IVec3::NEG_Z,
                IVec3::ZERO,
            ] {
                if self.chunks.contains_key(&(c + d)) {
                    self.dirty.insert(c + d);
                }
            }
            if start.elapsed().as_secs_f64() * 1000.0 > budget_ms {
                break;
            }
        }
        generated
    }

    /// True when a chunk and all 26 neighbours are loaded, so its mesh is final.
    pub fn neighbourhood_ready(&self, c: IVec3) -> bool {
        for dz in -1..=1 {
            for dy in -1..=1 {
                for dx in -1..=1 {
                    if !self.is_loaded(c + IVec3::new(dx, dy, dz)) {
                        return false;
                    }
                }
            }
        }
        true
    }

    /// Walks the voxel grid along a ray (Amanatides and Woo) up to `max_m` metres.
    pub fn raycast(&self, origin_m: Vec3, dir: Vec3, max_m: f32) -> Option<RayHit> {
        let o = origin_m / VOXEL_SIZE;
        let d = dir.normalize();
        let mut v = o.floor().as_ivec3();
        let step = IVec3::new(d.x.signum() as i32, d.y.signum() as i32, d.z.signum() as i32);
        let inv = Vec3::new(1.0 / d.x.abs(), 1.0 / d.y.abs(), 1.0 / d.z.abs());
        let frac = |p: f32, s: i32| if s > 0 { p.floor() + 1.0 - p } else { p - p.floor() };
        let mut t_max = Vec3::new(
            frac(o.x, step.x) * inv.x,
            frac(o.y, step.y) * inv.y,
            frac(o.z, step.z) * inv.z,
        );
        let max_t = max_m / VOXEL_SIZE;
        let mut prev = v;
        let mut t = 0.0;
        while t <= max_t {
            if is_solid(self.get(v)) {
                return Some(RayHit {
                    voxel: v,
                    before: prev,
                    distance: t * VOXEL_SIZE,
                });
            }
            prev = v;
            if t_max.x < t_max.y && t_max.x < t_max.z {
                v.x += step.x;
                t = t_max.x;
                t_max.x += inv.x;
            } else if t_max.y < t_max.z {
                v.y += step.y;
                t = t_max.y;
                t_max.y += inv.y;
            } else {
                v.z += step.z;
                t = t_max.z;
                t_max.z += inv.z;
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flat_world() -> World {
        let mut w = World::new(1, 1);
        let mut c = Chunk::default();
        for z in 0..CHUNK {
            for x in 0..CHUNK {
                for y in 0..4 {
                    c.set(x, y, z, STONE);
                }
            }
        }
        w.chunks.insert(IVec3::ZERO, c);
        w
    }

    #[test]
    fn raycast_hits_floor_and_reports_face() {
        let w = flat_world();
        let hit = w
            .raycast(Vec3::new(4.2, 6.0, 4.2), Vec3::NEG_Y, 20.0)
            .expect("should hit the floor");
        assert_eq!(hit.voxel.y, 3);
        assert_eq!(hit.before.y, 4);
        assert!((hit.distance - 4.0).abs() < 0.01);
    }

    #[test]
    fn edits_mark_neighbours_dirty() {
        let mut w = flat_world();
        w.chunks.insert(IVec3::new(-1, 0, 0), Chunk::default());
        w.dirty.clear();
        assert!(w.set(IVec3::new(0, 5, 10), DIRT));
        assert_eq!(w.get(IVec3::new(0, 5, 10)), DIRT);
        assert!(w.dirty.contains(&IVec3::ZERO));
        assert!(w.dirty.contains(&IVec3::new(-1, 0, 0)));
    }
}
