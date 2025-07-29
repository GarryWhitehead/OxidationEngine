use crate::backend::SamplerInfo;
use crate::sampler_cache::SamplerCache;
use crate::Driver;
use ash::vk;
use vk_mem::Alloc;

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

        let sampler = sampler_cache.get_or_create_sampler(sampler_info);

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

    pub fn map() { /* TODO: add function */
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
