use tree_sitter::Node;

/// Return true if `child` node is the argument field of its `parent` (field name "arg")
pub fn is_arg(parent: &Node, child: &Node) -> bool {
    for i in 0..parent.child_count() {
        if let Some(c) = parent.child(i) {
            if c == *child {
                return parent.field_name_for_child(i) == Some("arg");
            }
        }
    }
    false
} 