use crate::device::ContextDevice;
use crate::instance::ContextInstance;

use ash::{
    khr::{surface, swapchain},
    vk,
};
use std::error::Error;

/// A swapchain is Vulkans abstract object which deals with rendering
/// an image to the surface. The swapchain handles the images which will
/// be rendered to based upon the current index - usual setup gives
/// either double- or triple-buffered scenarios.
///
/// # Examples
///
/// ```
/// let instance = oxidation_vk::instance::ContextInstance::new();
/// let device = oxidation_vk::device::ContextDevice::new();
/// let win_size = (1980, 1080);
/// let swapchain = oxidation_vk::swapchain::Swapchain::new(instance, device, _, win_size.0, win_size.1);
/// ```
///
pub struct Swapchain {
    pub instance: vk::SwapchainKHR,
    pub extents: vk::Extent2D,
    pub surface_format: vk::SurfaceFormatKHR,
    pub swapchain_loader: swapchain::Device,
}

impl Swapchain {
    /// Find a suitbale surface for rendering to.
    /// The ideal format is a normalised pixel 8-bit BRGA format and a linear SRGB colour space.
    /// If this can't be fulfilled by the device, then the first option in chosen.
    fn find_surface_format(surface_formats: &[vk::SurfaceFormatKHR]) -> vk::SurfaceFormatKHR {
        if surface_formats[0].format == vk::Format::UNDEFINED {
            return vk::SurfaceFormatKHR {
                format: vk::Format::B8G8R8A8_UNORM,
                color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
            };
        }

        *surface_formats
            .iter()
            .find(|format| {
                format.format == vk::Format::B8G8R8A8_UNORM
                    && format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
            })
            .unwrap_or(&surface_formats[0])
    }

    /// Find a suitable presentation mode. The order of preference is:
    /// 1. Mailbox -> 2. FIFO -> 3. Immediate
    fn find_present_mode(present_modes: &[vk::PresentModeKHR]) -> vk::PresentModeKHR {
        if present_modes.contains(&vk::PresentModeKHR::MAILBOX) {
            vk::PresentModeKHR::MAILBOX
        } else if present_modes.contains(&vk::PresentModeKHR::FIFO) {
            return vk::PresentModeKHR::FIFO;
        } else {
            return vk::PresentModeKHR::IMMEDIATE;
        }
    }

    /// Create a new swapchain instance based upon the specified Vulkan window surface.
    pub fn new(
        instance: &ContextInstance,
        device: &ContextDevice,
        surface: &vk::SurfaceKHR,
        win_width: u32,
        win_height: u32,
    ) -> Result<Self, Box<dyn Error>> {
        let surface_loader = surface::Instance::new(&instance.entry, &instance.instance);

        let surface_caps = unsafe {
            surface_loader
                .get_physical_device_surface_capabilities(device.physical_device, *surface)
                .unwrap()
        };
        let surface_formats = unsafe {
            surface_loader
                .get_physical_device_surface_formats(device.physical_device, *surface)
                .expect("Unable to get physical device surface formats.")
        };
        let surface_present_modes = unsafe {
            surface_loader
                .get_physical_device_surface_present_modes(device.physical_device, *surface)
                .expect("Unable to get physical device surface present modes.")
        };

        let surface_format = Self::find_surface_format(&surface_formats);
        let present_mode = Self::find_present_mode(&surface_present_modes);

        let mut extents = surface_caps.current_extent;
        if surface_caps.current_extent.width == u32::MAX {
            extents.width = win_width
                .max(surface_caps.min_image_extent.width)
                .min(surface_caps.max_image_extent.width);
            extents.height = win_height
                .max(surface_caps.min_image_extent.height)
                .min(surface_caps.max_image_extent.height);
        }

        // Get the number of possible images we can send to the queue.
        let mut image_count: u32 = surface_caps.min_image_count + 1;
        if surface_caps.max_image_count > 0 && image_count > surface_caps.max_image_count {
            image_count = surface_caps.max_image_count;
        }

        let mut create_info = vk::SwapchainCreateInfoKHR::default()
            .image_extent(extents)
            .image_format(surface_format.format)
            .min_image_count(image_count)
            .surface(*surface)
            .present_mode(present_mode)
            .image_color_space(surface_format.color_space)
            .pre_transform(surface_caps.current_transform)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT);

        create_info = if surface_caps
            .supported_composite_alpha
            .contains(vk::CompositeAlphaFlagsKHR::INHERIT)
        {
            create_info.composite_alpha(vk::CompositeAlphaFlagsKHR::INHERIT)
        } else {
            create_info.composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
        };

        create_info = if device.graphics_queue_idx != device.present_queue_idx {
            create_info.image_sharing_mode(vk::SharingMode::CONCURRENT)
        } else {
            create_info.image_sharing_mode(vk::SharingMode::EXCLUSIVE)
        };

        let swapchain_loader = swapchain::Device::new(&instance.instance, &device.device);
        let swapchain = unsafe { swapchain_loader.create_swapchain(&create_info, None)? };

        Ok(Self {
            instance: swapchain,
            extents,
            surface_format,
            swapchain_loader,
        })
    }
}

impl Drop for Swapchain {
    fn drop(&mut self) {
        unsafe { self.swapchain_loader.destroy_swapchain(self.instance, None) };
    }
}
