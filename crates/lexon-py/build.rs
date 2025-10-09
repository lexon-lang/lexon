fn main() {
    // Ensure macOS dynamic linker allows unresolved Python symbols at link time.
    // pyo3 usually configures this via build scripts, but we add a fallback here.
    #[cfg(target_os = "macos")]
    {
        println!("cargo:rustc-cdylib-link-arg=-Wl,-undefined,dynamic_lookup");
    }
}
