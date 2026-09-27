//! Window and event loop, shared by desktop and browser builds.

use crate::game::{Game, Settings};
use crate::renderer::Renderer;
use std::sync::Arc;
use winit::application::ApplicationHandler;
use winit::event::{DeviceEvent, DeviceId, ElementState, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop, EventLoopProxy};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{CursorGrabMode, Window, WindowId};

pub enum UserEvent {
    Ready(Box<Renderer>),
    Failed(String),
}

pub struct App {
    proxy: EventLoopProxy<UserEvent>,
    window: Option<Arc<Window>>,
    renderer: Option<Renderer>,
    game: Game,
    last_frame: web_time::Instant,
    last_title: web_time::Instant,
    captured: bool,
    captured_at: web_time::Instant,
}

impl App {
    pub fn new(event_loop: &EventLoop<UserEvent>) -> Self {
        Self {
            proxy: event_loop.create_proxy(),
            window: None,
            renderer: None,
            game: Game::new(Settings::default()),
            last_frame: web_time::Instant::now(),
            last_title: web_time::Instant::now(),
            captured: false,
            captured_at: web_time::Instant::now(),
        }
    }

    fn capture(&mut self, on: bool) {
        let Some(w) = &self.window else { return };
        if on {
            let ok = w
                .set_cursor_grab(CursorGrabMode::Locked)
                .or_else(|_| w.set_cursor_grab(CursorGrabMode::Confined))
                .is_ok();
            w.set_cursor_visible(!ok);
            self.captured = ok;
            self.captured_at = web_time::Instant::now();
        } else {
            let _ = w.set_cursor_grab(CursorGrabMode::None);
            w.set_cursor_visible(true);
            self.captured = false;
            self.game.release_input();
        }
    }
}

impl ApplicationHandler<UserEvent> for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        #[allow(unused_mut)]
        let mut attrs = Window::default_attributes()
            .with_title("Strata")
            .with_inner_size(winit::dpi::LogicalSize::new(1280.0, 760.0));
        #[cfg(target_arch = "wasm32")]
        {
            use winit::platform::web::WindowAttributesExtWebSys;
            attrs = attrs.with_append(true);
        }
        let window = Arc::new(event_loop.create_window(attrs).expect("create window"));
        self.window = Some(window.clone());

        let proxy = self.proxy.clone();
        let init = async move {
            let event = match Renderer::new(window).await {
                Ok(r) => UserEvent::Ready(Box::new(r)),
                Err(e) => UserEvent::Failed(e),
            };
            let _ = proxy.send_event(event);
        };
        #[cfg(not(target_arch = "wasm32"))]
        pollster::block_on(init);
        #[cfg(target_arch = "wasm32")]
        wasm_bindgen_futures::spawn_local(init);
    }

    fn user_event(&mut self, event_loop: &ActiveEventLoop, event: UserEvent) {
        match event {
            UserEvent::Ready(r) => {
                log::info!("GPU: {}", r.adapter_name);
                #[allow(unused_mut)]
                let mut r = *r;
                // `?scale=0.5` renders the scene at half resolution, for comparing costs.
                #[cfg(target_arch = "wasm32")]
                {
                    r.scale_override = crate::web::url_param("scale").and_then(|v| v.parse().ok());
                }
                self.renderer = Some(r);
                if let Some(w) = &self.window {
                    let s = w.inner_size();
                    if let Some(r) = &mut self.renderer {
                        r.resize(s.width, s.height);
                    }
                    w.request_redraw();
                }
            }
            UserEvent::Failed(e) => {
                log::error!("Could not start the renderer: {e}");
                #[cfg(target_arch = "wasm32")]
                crate::web::show_error(&e);
                #[cfg(not(target_arch = "wasm32"))]
                event_loop.exit();
                let _ = event_loop;
            }
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(s) => {
                if let Some(r) = &mut self.renderer {
                    r.resize(s.width, s.height);
                }
            }
            WindowEvent::Focused(false) => self.capture(false),
            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(code) = event.physical_key {
                    let pressed = event.state == ElementState::Pressed;
                    if code == KeyCode::Escape && pressed {
                        self.capture(false);
                    } else if self.captured {
                        self.game.key(code, pressed);
                    }
                }
            }
            WindowEvent::MouseInput { state, button, .. } => {
                let pressed = state == ElementState::Pressed;
                if !self.captured {
                    if pressed {
                        self.capture(true);
                    }
                } else {
                    self.game.mouse_button(button, pressed);
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                if self.captured {
                    let lines = match delta {
                        MouseScrollDelta::LineDelta(_, y) => y,
                        MouseScrollDelta::PixelDelta(p) => (p.y / 40.0) as f32,
                    };
                    if lines != 0.0 {
                        self.game.scroll(lines);
                    }
                }
            }
            WindowEvent::RedrawRequested => {
                let now = web_time::Instant::now();
                // Browsers release pointer lock on Escape without telling the page's key handlers.
                #[cfg(target_arch = "wasm32")]
                if self.captured && (now - self.captured_at).as_secs_f32() > 0.5 && !crate::web::pointer_locked() {
                    self.capture(false);
                }
                let dt = (now - self.last_frame).as_secs_f32();
                self.last_frame = now;
                if let (Some(r), Some(w)) = (&mut self.renderer, &self.window) {
                    self.game.update(dt, r);
                    let (width, height) = r.render_size();
                    let globals = self.game.globals(width, height, r.manual_srgb(), r.render_scale());
                    let t = web_time::Instant::now();
                    r.render(&globals);
                    self.game.note_render_time(t.elapsed().as_secs_f32() * 1000.0);
                    if (now - self.last_title).as_secs_f32() > 0.5 {
                        self.last_title = now;
                        let status = self.game.status(r);
                        #[cfg(target_arch = "wasm32")]
                        crate::web::set_status(&status);
                        #[cfg(not(target_arch = "wasm32"))]
                        w.set_title(&status);
                    }
                    w.request_redraw();
                }
            }
            _ => {}
        }
    }

    fn device_event(&mut self, _el: &ActiveEventLoop, _id: DeviceId, event: DeviceEvent) {
        if let DeviceEvent::MouseMotion { delta } = event {
            if self.captured {
                self.game.mouse_motion(delta.0, delta.1);
            }
        }
    }
}
