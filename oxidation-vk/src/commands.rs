use ash::vk;

pub const MAX_CMD_BUFFER_IN_FLIGHT_COUNT: usize = 10;

#[derive(Debug, Copy, Clone, Default)]
pub struct CmdBuffer {
    pub buffer: vk::CommandBuffer,
    pub fence: vk::Fence,
}

#[allow(dead_code)]
/// An object which maintains a pool of command buffers and is responsible for the beginning/
/// ending command recording, pushing the commands to the required queue and synchronisation.
pub struct Commands {
    /// The current command buffer, which will be recorded to until the commands are flushed.
    current_cmds: Option<CmdBuffer>,
    /// The number of available cmd buffer slots. When the max count is reached, the
    /// in flight buffers are waited on until a free buffer slot becomes available.
    available_cmd_count: usize,
    current_signal: vk::Semaphore,
    // Current semaphore that has been submitted to the queue.
    submitted_signal: Option<vk::Semaphore>,
    /// The main command pool - only to be used on the main thread.
    main_cmd_pool: vk::CommandPool,
    /// Command pools for use when using threaded cmd buffer generation - one pool per thread.
    /// There are two cmd pool arrays - one will be the active cmd pool that is used for
    /// creating secondary command buffers, the other will have secondary command buffers which
    /// are potentially active.
    threaded_cmd_pools: Vec<vk::CommandPool>,
    // Wait semaphores passed by the client.
    external_signals: Vec<vk::Semaphore>,
    /// The command queue used by this object for pushing commands to when flushed.
    cmd_queue: vk::Queue,
    /// A container of cmd buffer slots.
    cmd_buffers: [Option<CmdBuffer>; MAX_CMD_BUFFER_IN_FLIGHT_COUNT],
    /// A container of signal slots - these are all initialised upon object creation.
    signals: [vk::Semaphore; MAX_CMD_BUFFER_IN_FLIGHT_COUNT],
}

impl Commands {
    #[allow(clippy::needless_range_loop)]
    pub fn new(queue_family_idx: u32, cmd_queue: vk::Queue, device: &ash::Device) -> Self {
        let main_cmd_pool = create_cmd_pool(
            queue_family_idx,
            vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
            device,
        );
        let mut signals: [vk::Semaphore; MAX_CMD_BUFFER_IN_FLIGHT_COUNT] = Default::default();
        let semaphore_create_info = vk::SemaphoreCreateInfo::default();
        for idx in 0..signals.len() {
            unsafe {
                signals[idx] = device
                    .create_semaphore(&semaphore_create_info, None)
                    .unwrap()
            };
        }

        Self {
            current_cmds: None,
            available_cmd_count: MAX_CMD_BUFFER_IN_FLIGHT_COUNT,
            external_signals: Vec::new(),
            current_signal: Default::default(),
            submitted_signal: None,
            main_cmd_pool,
            threaded_cmd_pools: Vec::new(),
            cmd_queue,
            cmd_buffers: [Default::default(); MAX_CMD_BUFFER_IN_FLIGHT_COUNT],
            signals,
        }
    }

    /// Get a command buffer. If a command buffer is already bound, then this
    /// will be returned. If not, a command buffer is grabbed from the pool
    /// if one is available, otherwise to gain a free slot - it will wait for
    /// a command buffer to finish on the queue before destroying and creating
    /// a new command buffer in that slot.
    pub fn get(&mut self, device: &ash::Device) -> vk::CommandBuffer {
        // If there is already a bound cmd buffer, return that.
        if let Some(current) = self.current_cmds {
            return current.buffer;
        }

        // Otherwise, if there are no available cmd buffers, wait for them
        // to finish.
        while self.available_cmd_count == 0 {
            self.free_cmd_buffers(device);
        }

        // Find the next available free cmd buffer slot.
        for i in 0..self.cmd_buffers.len() {
            if self.cmd_buffers[i].is_none() {
                let alloc_info = vk::CommandBufferAllocateInfo::default()
                    .command_pool(self.main_cmd_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1);
                let buffer = unsafe { device.allocate_command_buffers(&alloc_info).unwrap()[0] };

                // Begin the cmd buffer now so it's ready for recording commands.
                let begin_info = vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
                unsafe {
                    device.begin_command_buffer(buffer, &begin_info).unwrap();
                };

                // Create a fence to go with the cmd buffer for signalling when it has finished on the queue.
                let create_fence_info = vk::FenceCreateInfo::default();
                let fence = unsafe { device.create_fence(&create_fence_info, None).unwrap() };

                let cmd_buffer = CmdBuffer { buffer, fence };

                self.cmd_buffers[i] = Some(cmd_buffer);
                self.current_cmds = self.cmd_buffers[i];
                self.current_signal = self.signals[i];
                self.available_cmd_count -= 1;
            }
        }

        assert!(self.current_cmds.is_some());
        self.current_cmds.unwrap().buffer
    }

    pub fn free_cmd_buffers(&mut self, device: &ash::Device) {
        let mut fences: Vec<vk::Fence> = Vec::with_capacity(MAX_CMD_BUFFER_IN_FLIGHT_COUNT);
        for cmd_buffer in &self.cmd_buffers {
            if let Some(cmds) = cmd_buffer
                && cmds.buffer != self.current_cmds.unwrap().buffer
            {
                fences.push(cmds.fence);
            }
        }
        // Wait for all cmd buffers that are currently active.
        if !fences.is_empty() {
            unsafe { device.wait_for_fences(&fences, true, u64::MAX).unwrap() };
        }
        for i in 0..MAX_CMD_BUFFER_IN_FLIGHT_COUNT {
            if let Some(cmds) = self.cmd_buffers[i] {
                let res = unsafe { device.wait_for_fences(&[cmds.fence], true, 0) };
                if res.is_ok() {
                    unsafe { device.free_command_buffers(self.main_cmd_pool, &[cmds.buffer]) };
                    unsafe { device.destroy_fence(cmds.fence, Option::None) };
                    self.cmd_buffers[i] = None;
                    self.available_cmd_count += 1;
                }
            }
        }
    }

    /// Flush the current command buffer to the queue. This will
    /// invalidate the currently bound cmd buffer, so a call to `get()`
    /// will bind a new command buffer.
    pub fn flush(&mut self, device: &ash::Device) {
        // Early return if there are no commands to flush.
        if self.current_cmds.is_none() {
            return;
        }

        unsafe {
            device
                .end_command_buffer(self.current_cmds.unwrap().buffer)
                .unwrap()
        };

        let mut wait_signals: Vec<vk::Semaphore> = Vec::with_capacity(5);
        if let Some(signal) = self.submitted_signal {
            wait_signals.push(signal);
        }
        wait_signals.extend_from_slice(&self.external_signals);
        let stage_flags: [vk::PipelineStageFlags; 2] = [
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::PipelineStageFlags::ALL_COMMANDS,
        ];
        let buffers = [self.current_cmds.unwrap().buffer];
        let signals = [self.current_signal];
        let submit_info = vk::SubmitInfo::default()
            .wait_semaphores(&wait_signals)
            .wait_dst_stage_mask(&stage_flags)
            .command_buffers(&buffers)
            .signal_semaphores(&signals);
        unsafe {
            device
                .queue_submit(
                    self.cmd_queue,
                    &[submit_info],
                    self.current_cmds.unwrap().fence,
                )
                .unwrap()
        };
    }

    pub fn add_external_wait_signal(&mut self, signal: vk::Semaphore) {
        self.external_signals.push(signal);
    }

    pub fn destroy(&mut self, device: &ash::Device) {
        for signal in self.signals {
            unsafe { device.destroy_semaphore(signal, None) };
        }
    }
}

fn create_cmd_pool(
    queue_family_idx: u32,
    flags: vk::CommandPoolCreateFlags,
    device: &ash::Device,
) -> vk::CommandPool {
    let create_info = vk::CommandPoolCreateInfo::default()
        .queue_family_index(queue_family_idx)
        .flags(vk::CommandPoolCreateFlags::TRANSIENT | flags);
    unsafe { device.create_command_pool(&create_info, None).unwrap() }
}
