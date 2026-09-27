//! Block (voxel material) definitions. Each voxel is half a metre on a side.

/// Edge length of one voxel in metres.
pub const VOXEL_SIZE: f32 = 0.5;

pub type Block = u8;

pub const AIR: Block = 0;
pub const STONE: Block = 1;
pub const DIRT: Block = 2;
pub const GRASS: Block = 3;
pub const SAND: Block = 4;
pub const GRAVEL: Block = 5;
pub const SNOW: Block = 6;
pub const WOOD: Block = 7;
pub const LEAVES: Block = 8;
pub const WATER: Block = 9;
pub const PLANKS: Block = 10;
/// Packed earth and gravel on roads.
pub const PATH: Block = 11;
/// A glowing lantern; every one in the loaded world is a light at night.
pub const LANTERN: Block = 12;
/// Tall grass blades standing in the air voxel above a grass block.
pub const TALL_GRASS: Block = 13;
/// A thin wooden lantern post, drawn as a pole rather than a full voxel.
pub const POST: Block = 14;

/// Materials the player can place, in hotbar order (keys 1..).
pub const PLACEABLE: [Block; 8] = [STONE, DIRT, GRASS, SAND, GRAVEL, WOOD, PLANKS, SNOW];

pub fn name(b: Block) -> &'static str {
    match b {
        AIR => "air",
        STONE => "stone",
        DIRT => "dirt",
        GRASS => "grass",
        SAND => "sand",
        GRAVEL => "gravel",
        SNOW => "snow",
        WOOD => "wood",
        LEAVES => "leaves",
        WATER => "water",
        PLANKS => "planks",
        PATH => "path",
        LANTERN => "lantern",
        TALL_GRASS => "tall grass",
        POST => "post",
        _ => "unknown",
    }
}

/// Blocks that fully hide the faces of their neighbours.
#[inline]
pub fn is_opaque(b: Block) -> bool {
    !matches!(b, AIR | WATER | TALL_GRASS | POST | LANTERN)
}

/// Blocks the player collides with.
#[inline]
pub fn is_solid(b: Block) -> bool {
    !matches!(b, AIR | WATER | TALL_GRASS)
}
