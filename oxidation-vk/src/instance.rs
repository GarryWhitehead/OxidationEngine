use ash::{
    ext::debug_utils,
    khr::{surface, swapchain},
    vk, Entry,
};
use log::{info, warn};
use std::ffi::{c_char, CStr};
use std::{borrow::Cow, error::Error};

pub struct ContextInstance {
    pub(crate) entry: Entry,
    pub instance: ash::Instance,
    pub debug_loader: Option<debug_utils::Instance>,
    pub debug_callback: vk::DebugUtilsMessengerEXT,
}

impl ContextInstance {
    pub fn new(win_extension_names: Vec<*const c_char>) -> Result<Self, Box<dyn Error>> {
        let entry = unsafe { Entry::load()? };

        // Layer properties.
        let mut layer_extensions = Vec::new();
        let layer_properties = unsafe { entry.enumerate_instance_layer_properties().unwrap() };
        println!("layer_properties: {:?}", layer_properties);
        #[cfg(debug_assertions)]
        {
            let res = find_layer_properties(c"VK_LAYER_KHRONOS_validation", &layer_properties);
            match res {
                true => {
                    info!("Found validation layers.");
                    layer_extensions.push(c"VK_LAYER_KHRONOS_validation")
                }
                false => warn!("Unable to find validation layers"),
            }
        }

        // Instance extensions.
        let extension_props = unsafe { entry.enumerate_instance_extension_properties(None)? };
        let instance_extensions = create_extensions(&extension_props, win_extension_names)?;

        let app_name = c"OxidationEngine";
        let app_info = vk::ApplicationInfo::default()
            .engine_name(app_name)
            .application_name(app_name)
            .api_version(vk::make_api_version(0, 1, 3, 0))
            .application_version(0)
            .engine_version(0);

        let layer_extensions_raw: Vec<*const c_char> = layer_extensions
            .iter()
            .map(|raw_name| raw_name.as_ptr())
            .collect();

        let create_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_layer_names(&layer_extensions_raw)
            .enabled_extension_names(&instance_extensions);

        let vk_instance = unsafe { entry.create_instance(&create_info, None)? };

        let mut debug_callback: vk::DebugUtilsMessengerEXT = Default::default();
        let mut debug_loader: Option<debug_utils::Instance> = None;

        #[cfg(debug_assertions)]
        {
            let debug_info = vk::DebugUtilsMessengerCreateInfoEXT::default()
                .message_severity(
                    vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                        | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                        | vk::DebugUtilsMessageSeverityFlagsEXT::INFO,
                )
                .message_type(
                    vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                        | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                        | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
                )
                .pfn_user_callback(Some(vulkan_debug_callback));

            let dl = debug_utils::Instance::new(&entry, &vk_instance);
            debug_callback = unsafe { dl.create_debug_utils_messenger(&debug_info, None)? };
            debug_loader = Some(dl);
        }

        Ok(Self {
            entry: entry,
            instance: vk_instance,
            debug_loader: debug_loader,
            debug_callback: debug_callback,
        })
    }
}

unsafe extern "system" fn vulkan_debug_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    _user_data: *mut std::os::raw::c_void,
) -> vk::Bool32 {
    let cb_data = *p_callback_data;
    let msg_id = cb_data.message_id_number;

    let msg_id_name = if cb_data.p_message_id_name.is_null() {
        Cow::from("")
    } else {
        CStr::from_ptr(cb_data.p_message_id_name).to_string_lossy()
    };

    let msg = if cb_data.p_message.is_null() {
        Cow::from("")
    } else {
        CStr::from_ptr(cb_data.p_message).to_string_lossy()
    };

    info!("{message_severity:?}\n{message_type:?} [{msg_id_name} ({msg_id})] : {msg}\n");
    vk::FALSE
}

fn find_layer_properties(layer_name: &CStr, layer_props: &Vec<vk::LayerProperties>) -> bool {
    layer_props.iter().any(|layer| {
        let tmp = layer_name.to_string_lossy();
        tmp == unsafe { CStr::from_ptr(layer.layer_name.as_ptr()).to_string_lossy() }
    })
}

fn find_extension(ext_name: &CStr, extensions: &Vec<vk::ExtensionProperties>) -> bool {
    extensions.iter().any(|layer| {
        let tmp = ext_name.to_string_lossy();
        let layer_name = unsafe { CStr::from_ptr(layer.extension_name.as_ptr()).to_string_lossy() };
        tmp == unsafe { CStr::from_ptr(layer.extension_name.as_ptr()).to_string_lossy() }
    })
}

fn create_extensions(
    extensions: &Vec<vk::ExtensionProperties>,
    required_extensions: Vec<*const c_char>,
) -> Result<Vec<*const c_char>, Box<dyn Error>> {
    let mut out: Vec<*const c_char> = Vec::new();

    // Check that all required window extensions are available.
    for ext_name in required_extensions {
        match find_extension(unsafe { CStr::from_ptr(ext_name) }, extensions) {
            false => {
                return Err(Box::from(
                    "Required extension name not found for device instance.",
                ))
            }
            true => out.push(ext_name),
        };
    }

    if find_extension(
        c"VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME",
        extensions,
    ) {
        out.push(c"VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME".as_ptr());
    }
    if find_extension(ash::khr::external_memory_capabilities::NAME, extensions) {
        out.push(ash::khr::external_memory_capabilities::NAME.as_ptr());
    }
    if find_extension(ash::khr::external_semaphore_capabilities::NAME, extensions) {
        out.push(ash::khr::external_semaphore_capabilities::NAME.as_ptr());
    }
    if (find_extension(ash::khr::multiview::NAME, extensions)) {
        out.push(ash::khr::multiview::NAME.as_ptr());
    }

    #[cfg(debug_assertions)]
    {
        // Debug utils is a mandatory extension.
        match find_extension(ash::ext::debug_utils::NAME, extensions) {
            false => return Err(Box::from("Debug utils extension not found.")),
            true => out.push(ash::ext::debug_utils::NAME.as_ptr()),
        };
    }

    Ok(out)
}

impl Drop for ContextInstance {
    fn drop(&mut self) {
        match &self.debug_loader {
            Some(debug_loader) => unsafe {
                debug_loader.destroy_debug_utils_messenger(self.debug_callback, None)
            },
            None => {}
        };
        unsafe {
            self.instance.destroy_instance(None);
        }
    }
}
