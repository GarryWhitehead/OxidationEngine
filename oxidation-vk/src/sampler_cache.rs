use crate::backend;
use ash::vk;
use std::{collections::HashMap, rc::Rc};

pub struct SamplerCache {
    samplers: HashMap<backend::SamplerInfo, vk::Sampler>,
    device: Rc<ash::Device>,
}

/// A cache for Vulkan sampler objects. Allows for re-using the same samplers
/// which fit the requested sampler parameters rather than creating new
/// samplers on each request. Also, simplifies the destruction at the point of termination.
impl SamplerCache {
    pub fn new(device: &ash::Device) -> Self {
        Self {
            samplers: HashMap::new(),
            device: Rc::new(device.clone()),
        }
    }

    pub fn get_or_create_sampler(&mut self, info: &backend::SamplerInfo) -> vk::Sampler {
        let sampler = self.samplers.get(info);
        if let Some(sampler) = sampler {
            return *sampler;
        }

        let create_info = vk::SamplerCreateInfo {
            border_color: vk::BorderColor::FLOAT_OPAQUE_WHITE,
            compare_enable: info.enable_compare,
            anisotropy_enable: info.enable_anisotropy,
            max_anisotropy: info.anisotropy as f32,
            max_lod: info.mip_levels as f32,
            mipmap_mode: vk::SamplerMipmapMode::LINEAR,
            min_filter: info.min_filter.to_vk(),
            mag_filter: info.mag_filter.to_vk(),
            address_mode_u: info.addr_mode_u.to_vk(),
            address_mode_v: info.addr_mode_v.to_vk(),
            address_mode_w: info.addr_mode_w.to_vk(),
            compare_op: info.compare_op.to_vk(),
            ..Default::default()
        };

        let sampler = unsafe { self.device.create_sampler(&create_info, None).unwrap() };
        let res = self.samplers.insert(*info, sampler);
        match res {
            None => sampler,
            Some(_sampler) => {
                panic!("Internal error: Sampler already found in cache map.")
            }
        }
    }
}

impl Drop for SamplerCache {
    fn drop(&mut self) {
        for sampler in self.samplers.values() {
            unsafe { self.device.destroy_sampler(*sampler, None) };
        }
    }
}
