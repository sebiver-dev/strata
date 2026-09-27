//! Strata: a living voxel world on a custom engine. Everything you see is
//! generated from code; the game ships no textures, models or sounds.

pub mod app;
pub mod avatar;
pub mod block;
pub mod bridge;
pub mod budget;
pub mod chunk;
pub mod cottage;
pub mod far;
pub mod game;
pub mod mesh;
pub mod model;
pub mod noise;
pub mod player;
pub mod renderer;
pub mod structures;
pub mod terrain;
#[cfg(target_arch = "wasm32")]
pub mod web;
pub mod world;

use winit::event_loop::EventLoop;

/// Opens the window (or canvas) and runs the game until it is closed.
pub fn run() {
    let event_loop = EventLoop::<app::UserEvent>::with_user_event()
        .build()
        .expect("event loop");
    let app = app::App::new(&event_loop);
    #[cfg(not(target_arch = "wasm32"))]
    {
        let mut app = app;
        event_loop.run_app(&mut app).expect("event loop failed");
    }
    #[cfg(target_arch = "wasm32")]
    {
        use winit::platform::web::EventLoopExtWebSys;
        event_loop.spawn_app(app);
    }
}
