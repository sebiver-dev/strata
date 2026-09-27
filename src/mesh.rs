//! Turns a chunk of voxels into triangles. Only faces between a block and a
//! see-through neighbour are emitted, each corner shaded by ambient occlusion.

use crate::block::*;
use crate::chunk::CHUNK;
use crate::world::World;
use bytemuck::{Pod, Zeroable};
use glam::IVec3;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct Vertex {
    /// World position in metres.
    pub pos: [f32; 3],
    /// Bits 0..3 face direction, 3..11 material, 11..13 ambient occlusion (0 darkest).
    pub data: u32,
}

#[derive(Default)]
pub struct MeshData {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
    pub water_vertices: Vec<Vertex>,
    pub water_indices: Vec<u32>,
}

impl MeshData {
    pub fn is_empty(&self) -> bool {
        self.indices.is_empty() && self.water_indices.is_empty()
    }
}

struct Face {
    n: IVec3,
    u: IVec3,
    v: IVec3,
}

const FACES: [Face; 6] = [
    Face {
        n: IVec3::X,
        u: IVec3::Y,
        v: IVec3::Z,
    },
    Face {
        n: IVec3::NEG_X,
        u: IVec3::Z,
        v: IVec3::Y,
    },
    Face {
        n: IVec3::Y,
        u: IVec3::Z,
        v: IVec3::X,
    },
    Face {
        n: IVec3::NEG_Y,
        u: IVec3::X,
        v: IVec3::Z,
    },
    Face {
        n: IVec3::Z,
        u: IVec3::X,
        v: IVec3::Y,
    },
    Face {
        n: IVec3::NEG_Z,
        u: IVec3::Y,
        v: IVec3::X,
    },
];

const P: i32 = CHUNK + 2;

/// A copy of the chunk plus a one-voxel border from its neighbours.
struct Padded {
    data: Vec<Block>,
}

impl Padded {
    fn gather(world: &World, cpos: IVec3) -> Self {
        let origin = cpos * CHUNK - IVec3::ONE;
        let mut data = vec![AIR; (P * P * P) as usize];
        // Look up the 27 chunks once instead of hashing every voxel.
        let mut near: [Option<&crate::chunk::Chunk>; 27] = [None; 27];
        for (i, slot) in near.iter_mut().enumerate() {
            let d = IVec3::new(i as i32 % 3, i as i32 / 3 % 3, i as i32 / 9) - IVec3::ONE;
            *slot = world.chunks.get(&(cpos + d));
        }
        for y in 0..P {
            for z in 0..P {
                for x in 0..P {
                    let v = origin + IVec3::new(x, y, z);
                    let d = crate::chunk::chunk_of(v) - cpos + IVec3::ONE;
                    let b = if v.y < 0 {
                        STONE
                    } else {
                        match near[(d.x + d.y * 3 + d.z * 9) as usize] {
                            Some(c) => {
                                let l = crate::chunk::local_of(v);
                                c.get(l.x, l.y, l.z)
                            }
                            None => AIR,
                        }
                    };
                    data[((y * P + z) * P + x) as usize] = b;
                }
            }
        }
        Self { data }
    }

    /// Local coordinates in -1..=CHUNK.
    #[inline]
    fn get(&self, p: IVec3) -> Block {
        let q = p + IVec3::ONE;
        self.data[((q.y * P + q.z) * P + q.x) as usize]
    }
}

pub fn build(world: &World, cpos: IVec3) -> MeshData {
    match world.chunks.get(&cpos) {
        None => return MeshData::default(),
        Some(c) if c.is_uniform(AIR) => return MeshData::default(),
        Some(c) => {
            if let crate::chunk::Chunk::Uniform(b) = c {
                let buried = [IVec3::X, IVec3::NEG_X, IVec3::Y, IVec3::NEG_Y, IVec3::Z, IVec3::NEG_Z]
                    .iter()
                    .all(|d| match world.chunks.get(&(cpos + *d)) {
                        Some(crate::chunk::Chunk::Uniform(n)) => is_opaque(*n),
                        None => cpos.y + d.y < 0,
                        _ => false,
                    });
                if is_opaque(*b) && buried {
                    return MeshData::default();
                }
            }
        }
    }
    let pad = Padded::gather(world, cpos);
    let mut out = MeshData::default();
    let origin = cpos * CHUNK;

    for y in 0..CHUNK {
        for z in 0..CHUNK {
            for x in 0..CHUNK {
                let p = IVec3::new(x, y, z);
                let b = pad.get(p);
                if b == AIR {
                    continue;
                }
                for (fi, f) in FACES.iter().enumerate() {
                    let nb = pad.get(p + f.n);
                    if b == WATER {
                        if nb == AIR {
                            emit(
                                &pad,
                                &mut out.water_vertices,
                                &mut out.water_indices,
                                origin,
                                p,
                                fi,
                                f,
                                b,
                                true,
                            );
                        }
                    } else if !is_opaque(nb) {
                        emit(&pad, &mut out.vertices, &mut out.indices, origin, p, fi, f, b, false);
                    }
                }
            }
        }
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn emit(
    pad: &Padded,
    verts: &mut Vec<Vertex>,
    idx: &mut Vec<u32>,
    origin: IVec3,
    p: IVec3,
    fi: usize,
    f: &Face,
    b: Block,
    water: bool,
) {
    let base = if f.n.cmpgt(IVec3::ZERO).any() { p + f.n } else { p };
    let front = p + f.n;
    let occ = |q: IVec3| is_opaque(pad.get(q)) as u32;
    let start = verts.len() as u32;
    let mut ao = [3u32; 4];
    for (i, (cu, cv)) in [(0, 0), (1, 0), (1, 1), (0, 1)].into_iter().enumerate() {
        let su = if cu == 1 { f.u } else { -f.u };
        let sv = if cv == 1 { f.v } else { -f.v };
        if !water {
            let (s1, s2, c) = (occ(front + su), occ(front + sv), occ(front + su + sv));
            ao[i] = if s1 == 1 && s2 == 1 { 0 } else { 3 - (s1 + s2 + c) };
        }
        let corner = base + f.u * cu + f.v * cv;
        let mut pos = (origin + corner).as_vec3() * VOXEL_SIZE;
        if water && f.n == IVec3::Y {
            pos.y -= 0.06;
        }
        let data = fi as u32 | ((b as u32) << 3) | (ao[i] << 11);
        verts.push(Vertex {
            pos: pos.to_array(),
            data,
        });
    }
    // Split the quad along the brighter diagonal so occlusion gradients stay smooth.
    if ao[0] + ao[2] >= ao[1] + ao[3] {
        idx.extend_from_slice(&[start, start + 1, start + 2, start, start + 2, start + 3]);
    } else {
        idx.extend_from_slice(&[start + 1, start + 2, start + 3, start + 1, start + 3, start]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunk::Chunk;

    #[test]
    fn single_block_has_six_faces() {
        let mut w = World::new(1, 1);
        let mut c = Chunk::default();
        c.set(5, 5, 5, STONE);
        w.chunks.insert(IVec3::ZERO, c);
        let m = build(&w, IVec3::ZERO);
        assert_eq!(m.vertices.len(), 24);
        assert_eq!(m.indices.len(), 36);
        // Unoccluded faces are fully lit.
        assert!(m.vertices.iter().all(|v| (v.data >> 11) & 3 == 3));
    }

    #[test]
    fn hidden_faces_are_culled() {
        let mut w = World::new(1, 1);
        let mut c = Chunk::default();
        c.set(5, 5, 5, STONE);
        c.set(6, 5, 5, STONE);
        w.chunks.insert(IVec3::ZERO, c);
        let m = build(&w, IVec3::ZERO);
        assert_eq!(m.indices.len(), 10 * 6);
    }

    #[test]
    fn faces_wind_counter_clockwise_outward() {
        let mut w = World::new(1, 1);
        let mut c = Chunk::default();
        c.set(5, 5, 5, STONE);
        w.chunks.insert(IVec3::ZERO, c);
        let m = build(&w, IVec3::ZERO);
        let centre = glam::Vec3::splat(5.5 * VOXEL_SIZE);
        for tri in m.indices.chunks(3) {
            let [a, b, c] = [0, 1, 2].map(|i| glam::Vec3::from(m.vertices[tri[i] as usize].pos));
            let normal = (b - a).cross(c - a);
            let outward = (a + b + c) / 3.0 - centre;
            assert!(normal.dot(outward) > 0.0);
        }
    }
}
