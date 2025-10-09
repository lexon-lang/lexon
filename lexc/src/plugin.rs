use libloading::{Library, Symbol};
use once_cell::sync::Lazy;
use std::fs;
use std::path::Path;
use std::sync::Mutex;

/// Trait that plugins must implement. Any global initialization can
/// be done in `register`. The name is used for logging/diagnostics.
pub trait LexcPlugin: Send + Sync {
    fn name(&self) -> &'static str;
    fn register(&self);
}

/// Global list of loaded plugins (only names are stored for informational purposes).
static PLUGIN_REGISTRY: Lazy<Mutex<Vec<String>>> = Lazy::new(|| Mutex::new(Vec::new()));

/// Helper function to register a plugin by name.
pub fn register_plugin(name: &str) {
    let mut reg = PLUGIN_REGISTRY.lock().unwrap();
    if !reg.contains(&name.to_string()) {
        reg.push(name.to_string());
    }
}

/// Returns the list of currently registered plugins.
pub fn list_plugins() -> Vec<String> {
    PLUGIN_REGISTRY.lock().unwrap().clone()
}

/// Type of symbol exposed by dynamic libraries.
///
/// Each plugin must expose a function with signature:
/// ```ignore
/// #[no_mangle]
/// pub extern "C" fn lexc_plugin_create() -> *mut dyn LexcPlugin {
///     // In a real plugin we would return a pointer to the instance.
///     std::ptr::null_mut() // stub de ejemplo
/// }
/// ```
/// that returns a *Box* to the plugin. The library stays alive during the entire
/// execution by calling `std::mem::forget`.
///
/// For the purposes of this repository, plugins are not required to exist
/// packaged; but the system allows their loading at runtime.
///
#[allow(improper_ctypes_definitions)]
pub type PluginCreateFn = unsafe extern "C" fn() -> *mut dyn LexcPlugin;

/// Dynamically loads all compatible libraries (*so*, *dll*, *dylib*)
/// found in `dir_path`. Each must export the symbol
/// `lexc_plugin_create` that returns the plugin instance.
pub fn load_plugins_from_dir(dir_path: &Path) {
    if !dir_path.exists() {
        return; // Non-existent directory: simply ignore.
    }

    let entries = match fs::read_dir(dir_path) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("[plugin] Could not read directory {:?}: {}", dir_path, e);
            return;
        }
    };

    // To keep the libraries alive, we store them in a static vector.
    static LIBS: Lazy<Mutex<Vec<Library>>> = Lazy::new(|| Mutex::new(Vec::new()));

    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let ext_ok = matches!(
            path.extension().and_then(|s| s.to_str()),
            Some("so") | Some("dll") | Some("dylib")
        );
        if !ext_ok {
            continue;
        }

        unsafe {
            match Library::new(&path) {
                Ok(lib) => {
                    let get_plugin: Symbol<PluginCreateFn> = match lib.get(b"lexc_plugin_create") {
                        Ok(symbol) => symbol,
                        Err(e) => {
                            eprintln!("[plugin] Missing symbol in {:?}: {}", path, e);
                            continue;
                        }
                    };
                    let boxed_raw = get_plugin();
                    if boxed_raw.is_null() {
                        eprintln!("[plugin] Null plugin returned by {:?}", path);
                        continue;
                    }
                    let boxed: Box<dyn LexcPlugin> = Box::from_raw(boxed_raw);
                    let name = boxed.name();
                    boxed.register();
                    register_plugin(name);
                    println!("🔌 Plugin loaded: {}", name);
                    std::mem::forget(boxed); // Keep global effects, drop instance.
                                             // We store the library so it remains loaded.
                    LIBS.lock().unwrap().push(lib);
                }
                Err(e) => {
                    eprintln!("[plugin] Failed to load {:?}: {}", path, e);
                }
            }
        }
    }
}
