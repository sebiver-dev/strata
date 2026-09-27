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

// Materials that exist only on meshes, never as voxels.
/// Lupin flower spikes.
pub const LUPIN: Block = 15;
/// Daisy petals.
pub const DAISY: Block = 16;
/// The yellow heart of a daisy.
pub const DAISY_HEART: Block = 17;

// Built structures.
/// Dressed stone laid in courses: bridges, towers, castle walls, plinths.
pub const MASONRY: Block = 18;
/// Lime plaster between the timbers of a cottage wall.
pub const PLASTER: Block = 19;
/// Slate and shingle roofing.
pub const ROOF: Block = 20;
/// A lit window; glows warm from dusk on.
pub const WINDOW: Block = 21;
/// Planed, dark-stained oak: framing timbers, rafters, furniture.
pub const OAK: Block = 22;
/// Sawn boards: floors, doors, shutters, decks.
pub const BOARDS: Block = 23;
/// Woollen cloth: blankets and banners.
pub const CLOTH: Block = 24;
/// Collision inside built models. Solid but never drawn: the model is.
pub const BUILT: Block = 25;
/// Wrought iron: lantern frames, arms and chains (mesh only).
pub const IRON: Block = 26;
/// Pink and white lupin florets (mesh only); LUPIN is the purple kind.
pub const LUPIN_PINK: Block = 40;
pub const LUPIN_WHITE: Block = 41;

// Tree models (mesh only; trunks collide through hidden BUILT voxels).
/// Bark on trunks and boughs.
pub const BARK: Block = 27;
/// Broadleaf foliage clumps.
pub const FOLIAGE: Block = 28;
/// Conifer needles.
pub const NEEDLES: Block = 29;

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
        MASONRY => "masonry",
        PLASTER => "plaster",
        ROOF => "roof",
        WINDOW => "window",
        OAK => "oak",
        BOARDS => "boards",
        CLOTH => "cloth",
        BUILT => "building",
        POST => "post",
        IRON => "iron",
        BARK => "bark",
        FOLIAGE => "foliage",
        NEEDLES => "needles",
        _ => "unknown",
    }
}

/// Blocks that fully hide the faces of their neighbours.
#[inline]
pub fn is_opaque(b: Block) -> bool {
    !matches!(b, AIR | WATER | TALL_GRASS | POST | LANTERN | BUILT)
}

/// Blocks the player collides with.
#[inline]
pub fn is_solid(b: Block) -> bool {
    !matches!(b, AIR | WATER | TALL_GRASS)
}
