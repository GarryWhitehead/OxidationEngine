mod convert_to_vk;

use ash::vk;

#[derive(Hash, Eq, PartialEq, Copy, Clone)]
pub enum SamplerAddressMode {
    Repeat,
    MirroredRepeat,
    ClampToEdge,
    ClampToBorder,
    MirrorClampToEdge,
}

impl SamplerAddressMode {
    pub fn to_vk(&self) -> vk::SamplerAddressMode {
        match self {
            SamplerAddressMode::Repeat => vk::SamplerAddressMode::REPEAT,
            SamplerAddressMode::MirroredRepeat => vk::SamplerAddressMode::MIRRORED_REPEAT,
            SamplerAddressMode::ClampToBorder => vk::SamplerAddressMode::CLAMP_TO_BORDER,
            SamplerAddressMode::MirrorClampToEdge => vk::SamplerAddressMode::MIRROR_CLAMP_TO_EDGE,
            SamplerAddressMode::ClampToEdge => vk::SamplerAddressMode::CLAMP_TO_EDGE,
        }
    }
}

#[derive(Hash, Eq, PartialEq, Copy, Clone)]
pub enum SamplerFilter {
    Nearest,
    Linear,
    Cubic,
}

impl SamplerFilter {
    pub fn to_vk(&self) -> vk::Filter {
        match self {
            SamplerFilter::Nearest => vk::Filter::NEAREST,
            SamplerFilter::Linear => vk::Filter::LINEAR,
            SamplerFilter::Cubic => vk::Filter::CUBIC_EXT,
        }
    }
}

#[derive(Hash, Eq, PartialEq, Copy, Clone)]
pub enum CompareOp {
    Never,
    Less,
    Equal,
    LessOrEqual,
    Greater,
    NotEqual,
    GreaterOrEqual,
    Always,
}

impl CompareOp {
    pub fn to_vk(&self) -> vk::CompareOp {
        match self {
            CompareOp::Never => vk::CompareOp::NEVER,
            CompareOp::Less => vk::CompareOp::LESS,
            CompareOp::Equal => vk::CompareOp::EQUAL,
            CompareOp::LessOrEqual => vk::CompareOp::LESS_OR_EQUAL,
            CompareOp::Greater => vk::CompareOp::GREATER,
            CompareOp::NotEqual => vk::CompareOp::NOT_EQUAL,
            CompareOp::GreaterOrEqual => vk::CompareOp::GREATER_OR_EQUAL,
            CompareOp::Always => vk::CompareOp::ALWAYS,
        }
    }
}

#[derive(Hash, Eq, PartialEq, Copy, Clone)]
pub struct SamplerInfo {
    pub min_filter: SamplerFilter,
    pub mag_filter: SamplerFilter,
    pub addr_mode_u: SamplerAddressMode,
    pub addr_mode_v: SamplerAddressMode,
    pub addr_mode_w: SamplerAddressMode,
    pub compare_op: CompareOp,
    pub anisotropy: u32,
    pub mip_levels: u32,
    pub enable_compare: vk::Bool32,
    pub enable_anisotropy: vk::Bool32,
}
