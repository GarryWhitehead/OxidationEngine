use crate::Driver;
use crate::backend::SamplerInfo;
use crate::sampler_cache::SamplerCache;
use ash::vk;
use vk_mem::Alloc;

const MAX_MIP_LEVEL_COUNT: usize = 12;

#[derive(Debug, Copy, Clone)]
pub enum TextureType {
    Texture2d,
    Array2d,
    Cube2d,
    CubeArray2d,
}

#[derive(Debug, Copy, Clone)]
/// Describes the dimensions and type of the texture.
pub struct TextureInfo {
    pub width: u32,
    pub height: u32,
    pub mip_levels: u32,
    pub array_layers: u32,
    pub format: vk::Format,
    pub ty: TextureType,
}

impl Default for TextureInfo {
    fn default() -> Self {
        Self {
            width: 0,
            height: 0,
            mip_levels: 1,
            array_layers: 1,
            format: vk::Format::UNDEFINED,
            ty: TextureType::Texture2d,
        }
    }
}

#[allow(dead_code)]
/// A texture encompasses an image, its memory allocation and the corresponding image view(s).
///
/// # Example
///
/// ```
/// use ash::vk;
/// use oxidation_vk::backend::SamplerInfo;
/// use oxidation_vk::texture::{Texture, TextureInfo};
/// let info = TextureInfo {
///     width: 1920,
///     height: 1080,
///     ..Default::default()
/// };
/// let texture = Texture::new(&info, vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT, ..);
/// ```
///
pub struct Texture {
    info: TextureInfo,
    image_layout: vk::ImageLayout,
    image: vk::Image,
    vma_alloc: vk_mem::Allocation,
    image_views: Vec<vk::ImageView>,
    sampler: vk::Sampler,
    frames_until_gc: u32,
}

impl Texture {
    pub fn new(
        info: &TextureInfo,
        usage_flags: vk::ImageUsageFlags,
        vma_alloc: vk_mem::Allocator,
        device: &ash::Device,
        sampler_cache: &mut SamplerCache,
        sampler_info: &SamplerInfo,
    ) -> Self {
        assert!(sampler_info.mip_levels <= MAX_MIP_LEVEL_COUNT as u32);
        let (image, allocation) = Self::create_image(info, usage_flags, vma_alloc);

        let mut image_views = Vec::new();
        // The parent image view which depicts the total number of mip levels for the texture.
        image_views.push(Self::create_image_view(
            &image,
            info,
            0,
            info.mip_levels,
            device,
        ));

        // Generate an image view for each mip level.
        for mip_level in 1..info.mip_levels {
            image_views.push(Self::create_image_view(&image, info, mip_level, 1, device));
        }

        let sampler = sampler_cache.get_or_create_sampler(sampler_info, device);

        Self {
            info: *info,
            image_layout: get_image_layout(&info.format, &usage_flags),
            image: vk::Image::default(),
            vma_alloc: allocation,
            image_views,
            frames_until_gc: 0,
            sampler,
        }
    }

    /// Create a Vulkan image object and the corresponding memory allocation.
    pub fn create_image(
        info: &TextureInfo,
        usage_flags: vk::ImageUsageFlags,
        vma_alloc: vk_mem::Allocator,
    ) -> (vk::Image, vk_mem::Allocation) {
        let extents = vk::Extent3D {
            width: info.width,
            height: info.height,
            depth: 1,
        };

        let create_info = vk::ImageCreateInfo {
            image_type: vk::ImageType::TYPE_2D, // TODO: support 3d images
            format: info.format,
            extent: extents,
            mip_levels: info.mip_levels,
            array_layers: compute_array_layers(&info.ty, info.array_layers),
            samples: vk::SampleCountFlags::TYPE_1,
            tiling: vk::ImageTiling::OPTIMAL,
            usage: vk::ImageUsageFlags::TRANSFER_DST | usage_flags,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            initial_layout: vk::ImageLayout::UNDEFINED,
            ..Default::default()
        };

        let alloc_info = vk_mem::AllocationCreateInfo {
            usage: vk_mem::MemoryUsage::Auto,
            flags: vk_mem::AllocationCreateFlags::DEDICATED_MEMORY,
            priority: 1.0,
            ..Default::default()
        };

        unsafe { vma_alloc.create_image(&create_info, &alloc_info).unwrap() }
    }

    /// Create a Vulkan image view object for a specified image.
    pub fn create_image_view(
        image: &vk::Image,
        info: &TextureInfo,
        mip_level: u32,
        mip_count: u32,
        device: &ash::Device,
    ) -> vk::ImageView {
        let components = vk::ComponentMapping {
            r: vk::ComponentSwizzle::IDENTITY,
            g: vk::ComponentSwizzle::IDENTITY,
            b: vk::ComponentSwizzle::IDENTITY,
            a: vk::ComponentSwizzle::IDENTITY,
        };
        let sub_resource = vk::ImageSubresourceRange {
            aspect_mask: get_aspect_mask(info.format),
            base_mip_level: mip_level,
            base_array_layer: 0,
            level_count: mip_count,
            layer_count: 1,
        };

        let mut create_info = vk::ImageViewCreateInfo {
            image: *image,
            view_type: vk::ImageViewType::TYPE_2D,
            format: info.format,
            components,
            subresource_range: sub_resource,
            ..Default::default()
        };

        create_info.view_type = match info.ty {
            TextureType::Cube2d => vk::ImageViewType::CUBE,
            TextureType::CubeArray2d => vk::ImageViewType::CUBE_ARRAY,
            TextureType::Array2d => vk::ImageViewType::TYPE_2D_ARRAY,
            TextureType::Texture2d => vk::ImageViewType::TYPE_2D,
        };
        unsafe { device.create_image_view(&create_info, None).unwrap() }
    }

    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    /// Map an image to a given device location.
    /// Uses a staging buffer (CPU/GPU visible) to host the image data before
    /// copying to the device.
    /// All images (including their mip-chains) are transitioned,
    /// so they are ready for reading by the shader after copying.   
    pub fn map(
        &mut self,
        driver: &mut Driver,
        data: *const u8,
        data_size: vk::DeviceSize,
        offsets: &[vk::DeviceSize],
        generate_mipmaps: bool,
    ) {
        let stage = driver.staging_pool.get(data_size, &driver.vma_allocator);

        let mapped = unsafe { driver.vma_allocator.map_memory(&mut stage.memory).unwrap() };
        unsafe { mapped.copy_from(data, data_size as usize) };
        unsafe { driver.vma_allocator.unmap_memory(&mut stage.memory) };
        driver
            .vma_allocator
            .flush_allocation(&stage.memory, 0, data_size)
            .expect("Failed to flush memory");

        let cmds = driver.graphics_commands.get(&driver.device.device);
        let mut image_copy_info: Vec<vk::BufferImageCopy> = Vec::new();

        if !generate_mipmaps {
            let array_count = compute_array_layers(&self.info.ty, self.info.array_layers);
            image_copy_info.resize_with(
                (array_count * self.info.mip_levels) as usize,
                Default::default,
            );
            for face in 0..array_count {
                for level in 0..self.info.mip_levels {
                    let idx = (face * self.info.mip_levels + level) as usize;

                    let image_subresource = vk::ImageSubresourceLayers::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .mip_level(level)
                        .layer_count(1)
                        .base_array_layer(face);
                    let extents = vk::Extent3D::default()
                        .width(self.info.width >> level)
                        .height(self.info.height >> level)
                        .depth(1);

                    image_copy_info[idx] = image_copy_info[idx]
                        .buffer_offset(offsets[idx])
                        .image_subresource(image_subresource)
                        .image_extent(extents);
                }
            }
        } else {
            // If generating mip maps on the fly, then we only need image copy
            // info for the initial image, the rest will be blitted.
            image_copy_info.resize_with(1, Default::default);

            let image_subresource = vk::ImageSubresourceLayers::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .mip_level(0)
                .layer_count(1)
                .base_array_layer(0);
            let extents = vk::Extent3D::default()
                .width(self.info.width)
                .height(self.info.height)
                .depth(1);

            image_copy_info[0] = image_copy_info[0]
                .image_subresource(image_subresource)
                .image_extent(extents);
        }

        // Transition all mips to for dst transfer - this is required as the last step in copying is
        // then to transition all mips ready for shader read. Not having the levels in the correct
        // layout leads to validation warnings.
        let transition_count = match generate_mipmaps {
            true => 1,
            false => self.info.mip_levels as usize,
        };

        self.transition(
            &driver.device.device,
            cmds,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::TRANSFER,
            transition_count,
        );

        // Copy the image from the staging buffer to the device.
        unsafe {
            driver.device.device.cmd_copy_buffer_to_image(
                cmds,
                stage.buffer,
                self.image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &image_copy_info,
            )
        };

        // Transition the image(s) ready for reads by the fragment shader.
        self.transition(
            &driver.device.device,
            cmds,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::FRAGMENT_SHADER,
            transition_count,
        );

        // If required, now generate the mip-maps for the image.
        if generate_mipmaps {
            // TODO: Add mip map generation.
        }
    }

    #[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
    /// Transition an image to the new specified layout.
    /// This can be done for all mip levels by specifying the level count.
    pub fn transition(
        &mut self,
        device: &ash::Device,
        cmds: vk::CommandBuffer,
        old_layout: vk::ImageLayout,
        new_layout: vk::ImageLayout,
        src_stage_flags: vk::PipelineStageFlags,
        dst_stage_flags: vk::PipelineStageFlags,
        level_count: usize,
    ) {
        let mask = get_aspect_mask(self.info.format);
        let array_count = compute_array_layers(&self.info.ty, self.info.array_layers);

        let mut ranges: [vk::ImageSubresourceRange; MAX_MIP_LEVEL_COUNT] = Default::default();

        for level in 0..self.info.mip_levels as usize {
            ranges[level] = ranges[level]
                .aspect_mask(mask)
                .level_count(0)
                .layer_count(array_count)
                .base_mip_level(self.info.mip_levels)
                .base_array_layer(0)
                .base_mip_level(level as u32)
                .level_count(1);
        }

        let src_barrier: vk::AccessFlags = match old_layout {
            vk::ImageLayout::UNDEFINED => vk::AccessFlags::empty(),
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL => vk::AccessFlags::TRANSFER_READ,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL => vk::AccessFlags::TRANSFER_WRITE,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL => vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL => vk::AccessFlags::SHADER_READ,
            _ => vk::AccessFlags::empty(),
        };

        let dst_barrier: vk::AccessFlags = match new_layout {
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL => vk::AccessFlags::TRANSFER_READ,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL => vk::AccessFlags::TRANSFER_WRITE,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL => vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL => vk::AccessFlags::SHADER_READ,
            vk::ImageLayout::GENERAL => {
                vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE
            }
            vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL => {
                vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE
                    | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
            }
            _ => vk::AccessFlags::empty(),
        };

        let mut memory_barriers: [vk::ImageMemoryBarrier; MAX_MIP_LEVEL_COUNT] = Default::default();
        for i in 0..level_count {
            memory_barriers[i] = memory_barriers[i]
                .image(self.image)
                .old_layout(old_layout)
                .new_layout(new_layout)
                .subresource_range(ranges[i])
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .src_access_mask(src_barrier)
                .dst_access_mask(dst_barrier);
        }

        unsafe {
            device.cmd_pipeline_barrier(
                cmds,
                src_stage_flags,
                dst_stage_flags,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &memory_barriers,
            )
        };

        self.image_layout = new_layout;
    }
}

fn compute_array_layers(tex_type: &TextureType, array_count: u32) -> u32 {
    match tex_type {
        TextureType::Array2d => array_count,
        TextureType::Cube2d => 6,
        TextureType::CubeArray2d => 6 * array_count,
        TextureType::Texture2d => 1,
    }
}

fn get_aspect_mask(format: vk::Format) -> vk::ImageAspectFlags {
    match format {
        vk::Format::D32_SFLOAT_S8_UINT => {
            vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL
        }
        vk::Format::D24_UNORM_S8_UINT => {
            vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL
        }
        vk::Format::D32_SFLOAT => vk::ImageAspectFlags::DEPTH,
        vk::Format::D16_UNORM => vk::ImageAspectFlags::DEPTH,
        _ => vk::ImageAspectFlags::COLOR,
    }
}

fn get_image_layout(format: &vk::Format, usage_flags: &vk::ImageUsageFlags) -> vk::ImageLayout {
    if Driver::is_depth_format(format) || Driver::is_stencil_format(format) {
        vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL
    } else if usage_flags.contains(vk::ImageUsageFlags::STORAGE) {
        vk::ImageLayout::GENERAL
    } else {
        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
    }
}
