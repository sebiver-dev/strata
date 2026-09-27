//! Fixed-size cubes of voxels. Chunks that are entirely one block (open sky,
//! solid rock) are stored as a single value instead of 32 KiB of copies.

use crate::block::{Block, AIR};
use glam::IVec3;

pub const CHUNK_BITS: i32 = 5;
pub const CHUNK: i32 = 1 << CHUNK_BITS;
pub const CHUNK_VOLUME: usize = (CHUNK * CHUNK * CHUNK) as usize;

#[inline]
pub fn local_index(x: i32, y: i32, z: i32) -> usize {
    debug_assert!((0..CHUNK).contains(&x) && (0..CHUNK).contains(&y) && (0..CHUNK).contains(&z));
    ((y * CHUNK + z) * CHUNK + x) as usize
}

/// Chunk coordinate containing a voxel coordinate.
#[inline]
pub fn chunk_of(v: IVec3) -> IVec3 {
    IVec3::new(v.x >> CHUNK_BITS, v.y >> CHUNK_BITS, v.z >> CHUNK_BITS)
}

/// Position of a voxel inside its chunk.
#[inline]
pub fn local_of(v: IVec3) -> IVec3 {
    IVec3::new(v.x & (CHUNK - 1), v.y & (CHUNK - 1), v.z & (CHUNK - 1))
}

#[derive(Clone)]
pub enum Chunk {
    Uniform(Block),
    Dense(Box<[Block; CHUNK_VOLUME]>),
}

impl Default for Chunk {
    fn default() -> Self {
        Chunk::Uniform(AIR)
    }
}

impl Chunk {
    /// Builds a chunk from a full voxel array, collapsing it if every voxel matches.
    pub fn from_dense(data: Box<[Block; CHUNK_VOLUME]>) -> Self {
        let first = data[0];
        if data.iter().all(|&b| b == first) {
            Chunk::Uniform(first)
        } else {
            Chunk::Dense(data)
        }
    }

    #[inline]
    pub fn get(&self, x: i32, y: i32, z: i32) -> Block {
        match self {
            Chunk::Uniform(b) => *b,
            Chunk::Dense(d) => d[local_index(x, y, z)],
        }
    }

    pub fn set(&mut self, x: i32, y: i32, z: i32, b: Block) {
        match self {
            Chunk::Uniform(u) if *u == b => {}
            Chunk::Uniform(u) => {
                let mut d = Box::new([*u; CHUNK_VOLUME]);
                d[local_index(x, y, z)] = b;
                *self = Chunk::Dense(d);
            }
            Chunk::Dense(d) => d[local_index(x, y, z)] = b,
        }
    }

    /// World voxel positions of every `b` in the chunk at `cpos`.
    pub fn find(&self, cpos: IVec3, b: Block) -> Vec<IVec3> {
        let Chunk::Dense(data) = self else {
            return Vec::new();
        };
        let origin = cpos * CHUNK;
        data.iter()
            .enumerate()
            .filter(|(_, v)| **v == b)
            .map(|(i, _)| {
                let i = i as i32;
                origin + IVec3::new(i % CHUNK, i / (CHUNK * CHUNK), i / CHUNK % CHUNK)
            })
            .collect()
    }

    pub fn is_uniform(&self, b: Block) -> bool {
        matches!(self, Chunk::Uniform(u) if *u == b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::STONE;

    #[test]
    fn set_get_and_collapse() {
        let mut c = Chunk::default();
        assert!(c.is_uniform(AIR));
        c.set(3, 4, 5, STONE);
        assert_eq!(c.get(3, 4, 5), STONE);
        assert_eq!(c.get(4, 4, 5), AIR);
        let full = Chunk::from_dense(Box::new([STONE; CHUNK_VOLUME]));
        assert!(full.is_uniform(STONE));
    }

    #[test]
    fn negative_coordinates() {
        let v = IVec3::new(-1, 33, -33);
        assert_eq!(chunk_of(v), IVec3::new(-1, 1, -2));
        assert_eq!(local_of(v), IVec3::new(31, 1, 31));
    }
}
