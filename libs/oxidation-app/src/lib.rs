use oxidation_engine as engine;
use oxidation_vk as ovk;
use std::rc::Rc;

use oxidation_vk::Driver;
use std::sync::Arc;
use winit::raw_window_handle::HasDisplayHandle;
use winit::{
    application::ApplicationHandler, event::WindowEvent, event_loop::ActiveEventLoop,
    event_loop::EventLoop, window::Window, window::WindowAttributes, window::WindowId,
};

/// Used to run all the examples used by this project.
/// Gives a general idea on how to use the engine and create
/// the required Vulkan context.
///
/// # Examples
///
/// ```
/// let win_title = "MyApp";
/// let win_size = (1920, 1080);
/// let mut app = oxidation_app::App::new(win_title, win_size.0, win_size.1);
///  app.run();
///
pub struct App {
    window: Option<Arc<Window>>,
    window_size: (u32, u32),
    window_title: String,
    driver: Option<Rc<Driver>>,
}

impl App {
    /// Create a new application instance.
    pub fn new(win_title: &str, win_width: u32, win_height: u32) -> Self {
        env_logger::builder()
            .target(env_logger::Target::Stdout)
            .filter_level(log::LevelFilter::Trace)
            .is_test(true)
            .try_init()
            .expect("Unable to build env logger.");

        Self {
            window: None,
            window_size: (win_width, win_height),
            window_title: String::from(win_title),
            driver: None,
        }
    }

    /// Run the application.
    ///
    /// This will create a new Vulkan window instance on the
    /// start of the first OS window tick along with the
    /// engine instance.
    pub fn run(&mut self) {
        let event_loop = EventLoop::new().unwrap();
        event_loop.run_app(self).unwrap();
    }
}

impl ApplicationHandler for App {
    /// As required by the wininit ApplicationHandler trait.
    /// This is where the main setup occurs - this is the internal
    /// setup along with the user-defined callbacks.
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        // Create the window.
        let attrs = WindowAttributes::default()
            .with_title(&self.window_title)
            .with_inner_size(winit::dpi::LogicalSize::new(
                self.window_size.0 as f64,
                self.window_size.1 as f64,
            ));

        let window = Arc::new(event_loop.create_window(attrs).unwrap());

        // Extension properties courtesy of the window instance.
        let extension_names =
            ash_window::enumerate_required_extensions(window.display_handle().unwrap().as_raw())
                .unwrap()
                .to_vec();

        // Create a new Vulkan context - instance, device, etc.
        let driver = Rc::new(ovk::Driver::new(extension_names, &window).unwrap());

        // Create the core engine context - this associates with a particular Vulkan driver context (as a reference).
        // Future work: Multiple engine contexts can be created with different drivers for multi-GPU and/or multi-window
        // rendering.
        let mut engine = engine::Engine::new(driver.clone());
        let handle = engine.create_swapchain(self.window_size.0, self.window_size.1);
        match handle {
            Ok(handle) => {
                engine.set_current_swapchain(handle);
            }
            Err(err) => {
                println!("Error: {err:?}");
            }
        }

        self.window = Some(window);
        self.driver = Some(driver);
    }

    /// As required by the wininit ApplicationHandler trait.
    /// Handle all window events here, including rendering to the window surface
    /// for each frame.
    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::Resized(_) => {
                self.window
                    .as_ref()
                    .expect("resize event without a window")
                    .request_redraw();
                // TODO: Deal with regenerating the swapchain to the new window size.
            }
            WindowEvent::RedrawRequested => {
                let window = self
                    .window
                    .as_ref()
                    .expect("redraw request without a window");
                window.pre_present_notify();
            }
            _ => (),
        }
    }
}
