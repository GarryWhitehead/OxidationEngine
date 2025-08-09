use oxidation_utils::handle;
use oxidation_vk::{Driver, swapchain::Swapchain};
use std::{error::Error, rc::Rc};

type SwapchainHandle = handle::Handle<Swapchain>;

/// The engine is the main entry point into the API.
///
/// The engine holds and owns most of the resources used by
/// the various systems with resources being passed around as
/// handles to reduce issues with ownership.
///
/// # Examples
///
/// Create engine with swapchain
/// ```
/// let driver = std::rc::Rc::new(oxidation_vk::Driver::new()?);
/// let mut engine = oxidation_engine::Engine::new(driver);
/// let win_size = (1980,1080);
/// let handle = engine.create_swapchain(win_size.0, win_size.1);
/// ```
///
pub struct Engine {
    pub driver: Rc<Driver>,
    /// Resources that are owned by the engine.
    swapchains: Vec<Swapchain>,

    current_swapchain: SwapchainHandle,
}

impl Engine {
    /// Create a new engine instance.
    pub fn new(driver: Rc<Driver>) -> Self {
        let swapchains = Vec::new();

        Self {
            driver,
            swapchains,
            current_swapchain: Default::default(),
        }
    }

    /// Create a new swapchain based on  a window surface.
    /// Multiple swapchains can be created and rendered to by a single
    /// driver instance.
    pub fn create_swapchain(
        &mut self,
        width: u32,
        height: u32,
    ) -> Result<SwapchainHandle, Box<dyn Error>> {
        let swapchain = Swapchain::new(
            &self.driver.instance,
            &self.driver.device,
            &self.driver.surface,
            width,
            height,
        )?;
        let handle = SwapchainHandle::new(self.swapchains.len());
        self.swapchains.push(swapchain);
        Ok(handle)
    }

    /// Set the current swapchain.
    ///
    /// All render commands will be rendereed to this swapchain.
    #[inline]
    pub fn set_current_swapchain(&mut self, handle: SwapchainHandle) {
        self.current_swapchain = handle;
    }
}
