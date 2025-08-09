use crate::commands;
use ash::vk;
use vk_mem::Alloc;

#[derive(Debug)]
pub struct Instance {
    pub buffer: vk::Buffer,
    pub size: vk::DeviceSize,
    pub memory: vk_mem::Allocation,
    frame_last_used: u64,
}

impl Instance {
    pub fn new(buffer: vk::Buffer, size: vk::DeviceSize, memory: vk_mem::Allocation) -> Self {
        Self {
            buffer,
            size,
            memory,
            frame_last_used: 0,
        }
    }
}

/// A pool of staging buffers used for copying images from CPU to device memory.
pub struct StagingPool {
    free_stages: Vec<Instance>,
    in_use_stages: Vec<Instance>,
    current_frame: u64,
}

impl StagingPool {
    pub fn new() -> Self {
        Self {
            free_stages: Vec::new(),
            in_use_stages: Vec::new(),
            current_frame: 0,
        }
    }

    /// Get a staging buffer of the required size.
    /// If there are stages that are free which fit the requirements, then
    /// this will be used to save having to create a new allocation.
    pub fn get(
        &mut self,
        required_size: vk::DeviceSize,
        vma_allocator: &vk_mem::Allocator,
    ) -> &mut Instance {
        // Check whether there are any free stages that fit the required size specification.
        if let Some(instance) = self
            .free_stages
            .iter_mut()
            .find(|instance| instance.size >= required_size)
        {
            return instance;
        }
        // If not, create a new stage.
        let instance = create_stage(vma_allocator, required_size);
        self.in_use_stages.push(instance);
        self.in_use_stages.last_mut().unwrap()
    }

    /// Garbage collection - free stage buffers which exceed the designated max frame
    /// time are destroyed. Those buffers which are in the in-use container but are
    /// have been allocated a number of frames ago, are moved to the free stage
    /// container for re-use.
    pub fn gc(&mut self, current_frame: u64, vma_allocator: &vk_mem::Allocator) {
        if self.current_frame >= commands::MAX_CMD_BUFFER_IN_FLIGHT_COUNT as u64 {
            for idx in 0..self.free_stages.len() {
                let collect_frame = self.free_stages[idx].frame_last_used
                    + commands::MAX_CMD_BUFFER_IN_FLIGHT_COUNT as u64;
                if collect_frame < current_frame {
                    let instance = &mut self.free_stages[idx];
                    unsafe { vma_allocator.destroy_buffer(instance.buffer, &mut instance.memory) };
                    self.free_stages.remove(idx);
                }
            }
            // In use buffers which haven't been used in a while and transferred to the free stages container.
            for idx in 0..self.in_use_stages.len() {
                let collect_frame = self.in_use_stages[idx].frame_last_used
                    + commands::MAX_CMD_BUFFER_IN_FLIGHT_COUNT as u64;
                if collect_frame < current_frame {
                    let instance = self.in_use_stages.remove(idx);
                    self.free_stages.push(instance);
                }
            }
        }
        self.current_frame += 1;
    }

    pub fn destroy(&mut self, vma_allocator: &vk_mem::Allocator) {
        for stage in self.free_stages.iter_mut() {
            unsafe { vma_allocator.destroy_buffer(stage.buffer, &mut stage.memory) };
        }
        for stage in self.in_use_stages.iter_mut() {
            unsafe { vma_allocator.destroy_buffer(stage.buffer, &mut stage.memory) };
        }
    }
}

impl Default for StagingPool {
    fn default() -> Self {
        Self::new()
    }
}

fn create_stage(vma_alloc: &vk_mem::Allocator, size: vk::DeviceSize) -> Instance {
    let buffer_create_info = vk::BufferCreateInfo::default()
        .usage(vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST)
        .size(size);
    let alloc_create_info = vk_mem::AllocationCreateInfo {
        usage: vk_mem::MemoryUsage::Auto,
        ..Default::default()
    };
    let (buffer, alloc) = unsafe {
        vma_alloc
            .create_buffer(&buffer_create_info, &alloc_create_info)
            .unwrap()
    };
    Instance::new(buffer, size, alloc)
}
