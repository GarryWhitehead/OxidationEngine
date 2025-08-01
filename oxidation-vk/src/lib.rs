pub mod backend;
pub mod device;
pub mod instance;
mod sampler_cache;
pub mod swapchain;
pub mod texture;

use crate::device::ContextDevice;
use crate::instance::ContextInstance;

use crate::sampler_cache::SamplerCache;
pub use ash::{vk, Entry, Instance};
use std::ffi::c_char;
pub use std::{error::Error, rc::Rc};
use winit::raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use winit::window::Window;

#[allow(dead_code)]
/// A Vulkan driver which encompasses the Vk instance and device context
/// along with all resource required to "drive" the vulkan backend
/// rendering.
///
/// This should be used as the primary entry point to the vulkan backend
/// from the engine side.
///
/// # Safety
/// The driver itself is to be used on a single thread, though its
/// associated caches (i.e., pipeline and descriptor) are designed to be
/// used across multiple threads.
///
/// # Examples
///
/// Driver setup.
/// ```
/// use ash::vk;
/// use winit::{window::WindowAttributes, event_loop};
/// use oxidation_vk as ovk;
///
/// let event_loop = event_loop::EventLoop::new().unwrap();
/// let window = event_loop.create_window(WindowAttributes::default()).unwrap();
/// let win_extensions = Vec::new();
/// let driver = ovk::Driver::new(win_extensions, &window);
/// ```
///
pub struct Driver {
    pub device: ContextDevice,
    pub instance: ContextInstance,
    vma_allocator: vk_mem::Allocator,
    /// Semaphore used to signal that the image is ready for presentation.
    image_ready_signal: vk::Semaphore,
    /// The current presentation image index that is written to.
    current_image_index: u32,
    /// The window surface which is associated with this driver context.
    pub surface: vk::SurfaceKHR,
    pub sampler_cache: sampler_cache::SamplerCache,
}

impl Driver {
    /// Create a new Vulkan driver instance based on the specified window.
    pub fn new(
        extension_names: Vec<*const c_char>,
        window: &Window,
    ) -> Result<Self, Box<dyn Error>> {
        // Create the main vulkan instance for a given set of display extensions.
        let instance = ContextInstance::new(extension_names)?;

        // Create the window surface.
        let surface = unsafe {
            ash_window::create_surface(
                &instance.entry,
                &instance.instance,
                window.display_handle().unwrap().as_raw(),
                window.window_handle().unwrap().as_raw(),
                None,
            )?
        };

        let device = ContextDevice::new(&instance, &surface)?;

        // Create the VMA allocator.
        let mut create_info = vk_mem::AllocatorCreateInfo::new(
            &instance.instance,
            &device.device,
            device.physical_device,
        );
        create_info.vulkan_api_version = vk::make_api_version(0, 1, 3, 0);
        let vma_allocator = unsafe { vk_mem::Allocator::new(create_info)? };

        let semaphore_info = vk::SemaphoreCreateInfo::default();
        let image_ready_signal = unsafe { device.device.create_semaphore(&semaphore_info, None)? };
        let sampler_cache = SamplerCache::new(&device.device);

        Ok(Self {
            device,
            instance,
            vma_allocator,
            image_ready_signal,
            current_image_index: 0,
            surface,
            sampler_cache,
        })
    }

    pub fn is_depth_format(format: &vk::Format) -> bool {
        let depth_formats = [
            vk::Format::D16_UNORM,
            vk::Format::D32_SFLOAT,
            vk::Format::D32_SFLOAT_S8_UINT,
            vk::Format::D24_UNORM_S8_UINT,
            vk::Format::D16_UNORM_S8_UINT,
            vk::Format::X8_D24_UNORM_PACK32,
        ];
        depth_formats.contains(format)
    }

    pub fn is_stencil_format(format: &vk::Format) -> bool {
        let stencil_formats = [
            vk::Format::S8_UINT,
            vk::Format::D16_UNORM_S8_UINT,
            vk::Format::D24_UNORM_S8_UINT,
            vk::Format::D32_SFLOAT_S8_UINT,
        ];
        stencil_formats.contains(format)
    }
}

impl Drop for Driver {
    fn drop(&mut self) {
        unsafe {
            self.device
                .device
                .destroy_semaphore(self.image_ready_signal, None)
        };
    }
}
