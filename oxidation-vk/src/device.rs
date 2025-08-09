use ash::khr::{surface, swapchain};
use ash::{Entry, Instance, vk};
use std::error::Error;

use crate::instance::ContextInstance;

pub struct ContextDevice {
    pub device: ash::Device,
    pub physical_device: vk::PhysicalDevice,
    pub graphics_queue_idx: u32,
    pub compute_queue_idx: u32,
    pub present_queue_idx: u32,
    pub graphics_queue: vk::Queue,
    pub compute_queue: vk::Queue,
    pub present_queue: vk::Queue,
}

impl ContextDevice {
    pub fn new(
        c_instance: &ContextInstance,
        surface: &vk::SurfaceKHR,
    ) -> Result<Self, Box<dyn Error>> {
        let (physical_device, queue_family_idx) =
            find_physical_device(&c_instance.instance, &c_instance.entry, surface)?;

        let (graphics_queue_idx, compute_queue_idx, present_queue_idx) =
            create_queue_indices(&c_instance.instance, physical_device, queue_family_idx);

        let queue_priority = [1.0];
        let mut queue_infos: Vec<vk::DeviceQueueCreateInfo> = Vec::new();
        // A graphics queue is mandatory - presentation and compute queues that differ
        // from the graphics queue depends on the device.
        queue_infos.push(
            vk::DeviceQueueCreateInfo::default()
                .queue_family_index(graphics_queue_idx)
                .queue_priorities(&queue_priority),
        );

        // Check for a separate compute queue.
        if graphics_queue_idx != compute_queue_idx {
            queue_infos.push(
                vk::DeviceQueueCreateInfo::default()
                    .queue_family_index(compute_queue_idx)
                    .queue_priorities(&queue_priority),
            )
        };

        // Check for separate present queue.
        if graphics_queue_idx != present_queue_idx {
            queue_infos.push(
                vk::DeviceQueueCreateInfo::default()
                    .queue_family_index(present_queue_idx)
                    .queue_priorities(&queue_priority),
            )
        }

        let phys_features = unsafe {
            c_instance
                .instance
                .get_physical_device_features(physical_device)
        };

        let mut robust_info = vk::PhysicalDeviceImageRobustnessFeatures {
            robust_image_access: vk::TRUE,
            ..Default::default()
        };
        let mut features12 = vk::PhysicalDeviceVulkan12Features::default()
            .draw_indirect_count(true)
            .shader_sampled_image_array_non_uniform_indexing(true)
            .runtime_descriptor_array(true)
            .descriptor_binding_variable_descriptor_count(true)
            .descriptor_binding_partially_bound(true)
            .descriptor_binding_sampled_image_update_after_bind(true)
            .descriptor_indexing(true);
        let mut multi_view_info = vk::PhysicalDeviceMultiviewFeaturesKHR::default()
            .multiview(true)
            .multiview_geometry_shader(true)
            .multiview_tessellation_shader(true);

        let phys_dev_features = vk::PhysicalDeviceFeatures {
            texture_compression_etc2: phys_features.texture_compression_etc2,
            texture_compression_bc: phys_features.texture_compression_bc,
            sampler_anisotropy: phys_features.sampler_anisotropy,
            tessellation_shader: phys_features.tessellation_shader,
            shader_storage_image_extended_formats: phys_features
                .shader_storage_image_extended_formats,
            multi_draw_indirect: phys_features.multi_draw_indirect,
            multi_viewport: phys_features.multi_viewport,
            depth_clamp: phys_features.depth_clamp,
            ..Default::default()
        };
        let mut required_features = vk::PhysicalDeviceFeatures2::default()
            .features(phys_dev_features)
            .push_next(&mut multi_view_info)
            .push_next(&mut features12)
            .push_next(&mut robust_info);

        let device_extension_names_raw = [
            swapchain::NAME.as_ptr(),
            // TODO: Check that this is valid for the device.
            ash::ext::descriptor_indexing::NAME.as_ptr(),
        ];

        let device_create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_infos)
            .enabled_features(&phys_features)
            .enabled_extension_names(&device_extension_names_raw)
            .push_next(&mut required_features);

        let device = unsafe {
            c_instance
                .instance
                .create_device(physical_device, &device_create_info, None)?
        };

        let graphics_queue = unsafe { device.get_device_queue(graphics_queue_idx, 0) };
        let compute_queue = unsafe { device.get_device_queue(compute_queue_idx, 0) };
        let present_queue = unsafe { device.get_device_queue(present_queue_idx, 0) };

        Ok(Self {
            device,
            physical_device,
            graphics_queue_idx,
            compute_queue_idx,
            present_queue_idx,
            graphics_queue,
            compute_queue,
            present_queue,
        })
    }

    pub fn destroy(&mut self) {
        unsafe { self.device.destroy_device(None) };
    }
}

fn find_physical_device(
    instance: &Instance,
    entry: &Entry,
    win_surface: &vk::SurfaceKHR,
) -> Result<(vk::PhysicalDevice, u32), Box<dyn Error>> {
    let phys_devices = unsafe {
        instance
            .enumerate_physical_devices()
            .expect("Unable to find any physical devices.")
    };

    // Find an appropriate physical device.
    let surface_loader = surface::Instance::new(entry, instance);
    let (phys_device, queue_family_idx) = phys_devices
        .iter()
        .find_map(|phys_device| unsafe {
            instance
                .get_physical_device_queue_family_properties(*phys_device)
                .iter()
                .enumerate()
                .find_map(|(idx, info)| {
                    // Looking for a device with the same graphics and presentation queue.
                    let found_supported = info.queue_flags.contains(vk::QueueFlags::GRAPHICS)
                        && surface_loader
                            .get_physical_device_surface_support(
                                *phys_device,
                                idx as u32,
                                *win_surface,
                            )
                            .unwrap();
                    if found_supported {
                        Some((*phys_device, idx))
                    } else {
                        None
                    }
                })
        })
        .expect("Unable to find a valid device.");

    Ok((phys_device, queue_family_idx as u32))
}

fn create_queue_indices(
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
    queue_family_idx: u32,
) -> (u32, u32, u32) {
    let graphics_queue_idx = queue_family_idx;
    // This could potentially get over-ridden if there is a separate queue on the device.
    let mut compute_queue_idx = graphics_queue_idx;
    // TODO: Check whether the device has a separate presentation queue.
    let present_queue_idx = graphics_queue_idx;

    // Check for a separate compute queue.
    let queue_properties =
        unsafe { instance.get_physical_device_queue_family_properties(physical_device) };

    for (idx, prop) in queue_properties.iter().enumerate() {
        if prop.queue_flags.contains(vk::QueueFlags::COMPUTE) && idx != graphics_queue_idx as usize
        {
            compute_queue_idx = idx as u32;
        }
    }

    (graphics_queue_idx, compute_queue_idx, present_queue_idx)
}
