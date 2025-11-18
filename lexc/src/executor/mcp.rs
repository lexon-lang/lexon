// lexc/src/executor/mcp.rs
// Minimal MCP 1.1 scaffold (no server yet): list_tools handler stub

use super::{ExecutionEnvironment, ExecutorError, RuntimeValue, ValueRef};
use crate::lexir::LexExpression;
use base64::Engine;
use hmac::{Hmac, Mac};
use once_cell::sync::Lazy;
use sha2::Sha256;
use std::collections::HashMap;
use std::sync::RwLock;
use tokio_tungstenite::tungstenite::Message;

type WsSink = Box<
    dyn futures_util::Sink<Message, Error = tokio_tungstenite::tungstenite::Error> + Unpin + Send,
>;
type WsStream = Box<
    dyn futures_util::Stream<
            Item = std::result::Result<Message, tokio_tungstenite::tungstenite::Error>,
        > + Unpin
        + Send,
>;

static MCP_TOOLS: Lazy<RwLock<HashMap<String, serde_json::Value>>> =
    Lazy::new(|| RwLock::new(HashMap::new()));
static MCP_ACTIVE: Lazy<RwLock<HashMap<String, usize>>> = Lazy::new(|| RwLock::new(HashMap::new()));
static MCP_RATE: Lazy<RwLock<HashMap<String, (u64, u32)>>> =
    Lazy::new(|| RwLock::new(HashMap::new()));
static MCP_ACL: Lazy<RwLock<HashMap<String, Vec<String>>>> =
    Lazy::new(|| RwLock::new(HashMap::new()));

fn append_audit(base: &Option<String>, entry: &serde_json::Value) {
    if let Some(base) = base {
        let path = std::path::Path::new(base).join("mcp_audit.json");
        let mut items: Vec<serde_json::Value> = Vec::new();
        if path.exists() {
            if let Ok(s) = std::fs::read_to_string(&path) {
                if let Ok(mut arr) = serde_json::from_str::<Vec<serde_json::Value>>(&s) {
                    items.append(&mut arr);
                }
            }
        }
        let mut e = entry.clone();
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis();
        if let Some(obj) = e.as_object_mut() {
            obj.insert("ts_ms".to_string(), serde_json::json!(ts));
        }
        items.push(e);
        let _ = std::fs::create_dir_all(path.parent().unwrap_or(std::path::Path::new(".")));
        let _ = std::fs::write(
            path,
            serde_json::to_string_pretty(&items).unwrap_or("[]".to_string()),
        );
    }
}

impl ExecutionEnvironment {
    fn mcp_identity_from_bearer(bearer: &str) -> String {
        // If JWT HS256 enabled, try to extract "sub" from payload without verifying
        if let Ok((_hdr, payload)) = Self::jwt_decode_parts(bearer) {
            if let Some(sub) = payload.get("sub").and_then(|v| v.as_str()) {
                return format!("jwt:{}", sub);
            }
        }
        format!("bearer:{}", &bearer[..std::cmp::min(12, bearer.len())])
    }

    fn jwt_decode_parts(token: &str) -> Result<(serde_json::Value, serde_json::Value), ()> {
        let mut parts = token.split('.');
        let (h, p) = (parts.next(), parts.next());
        if let (Some(h), Some(p)) = (h, p) {
            let dec = |s: &str| -> Result<serde_json::Value, ()> {
                let data = base64::engine::general_purpose::URL_SAFE_NO_PAD
                    .decode(s.as_bytes())
                    .map_err(|_| ())?;
                serde_json::from_slice(&data).map_err(|_| ())
            };
            let hdr = dec(h)?;
            let payload = dec(p)?;
            return Ok((hdr, payload));
        }
        Err(())
    }

    fn jwt_verify_hs256(token: &str, secret: &str) -> bool {
        let mut iter = token.rsplitn(2, '.');
        let sig_b64 = if let Some(s) = iter.next() {
            s
        } else {
            return false;
        };
        let hp = if let Some(s) = iter.next() {
            s
        } else {
            return false;
        };
        // compute HMAC-SHA256 over header.payload
        type HmacSha256 = Hmac<Sha256>;
        let mut mac = match HmacSha256::new_from_slice(secret.as_bytes()) {
            Ok(m) => m,
            Err(_) => return false,
        };
        mac.update(hp.as_bytes());
        let sig = mac.finalize().into_bytes();
        let sig_expected = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(sig);
        if sig_expected != sig_b64 {
            return false;
        }
        // Optional exp check
        if let Ok((_h, payload)) = Self::jwt_decode_parts(token) {
            if let Some(exp) = payload.get("exp").and_then(|v| v.as_u64()) {
                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs();
                if now > exp {
                    return false;
                }
            }
        }
        true
    }

    fn mcp_check_rate(identity: &str) -> Result<(), ExecutorError> {
        let limit: u32 = std::env::var("LEXON_MCP_RATE_PER_MIN")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0);
        if limit == 0 {
            return Ok(());
        }
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        let window_ms = 60_000u64;
        let mut map = MCP_RATE.write().unwrap();
        let entry = map.entry(identity.to_string()).or_insert((now_ms, 0));
        let (start, count) = *entry;
        let new = if now_ms.saturating_sub(start) >= window_ms {
            (now_ms, 1)
        } else {
            if count >= limit {
                return Err(ExecutorError::RuntimeError(
                    "rate limit exceeded".to_string(),
                ));
            }
            (start, count + 1)
        };
        *entry = new;
        Ok(())
    }

    fn mcp_acl_load() {
        if MCP_ACL.read().unwrap().is_empty() {
            if let Ok(path) = std::env::var("LEXON_MCP_ACL_FILE") {
                if let Ok(s) = std::fs::read_to_string(path) {
                    if let Ok(map) = serde_json::from_str::<HashMap<String, Vec<String>>>(&s) {
                        *MCP_ACL.write().unwrap() = map;
                    }
                }
            }
        }
    }

    fn mcp_acl_is_allowed(identity: &str, tool: &str) -> bool {
        Self::mcp_acl_load();
        if let Some(list) = MCP_ACL.read().unwrap().get(identity) {
            return list.iter().any(|t| t == tool);
        }
        // If no ACL for identity, default allow (can be changed later)
        true
    }
    // Simple schema validation: { required: [..], properties: {k: "string|number|boolean|object|array"} }
    fn mcp_validate_args_schema(
        schema: &serde_json::Value,
        args: &serde_json::Value,
    ) -> Result<(), ExecutorError> {
        // Prefer full JSON Schema validation when schema looks like it
        let looks_jsonschema = schema.get("$schema").is_some()
            || schema.get("type").is_some()
            || schema.get("properties").is_some();
        if looks_jsonschema {
            match jsonschema::JSONSchema::compile(schema) {
                Ok(compiled) => {
                    if let Err(errors) = compiled.validate(args) {
                        let msg = errors.map(|e| e.to_string()).collect::<Vec<_>>().join("; ");
                        return Err(ExecutorError::ArgumentError(format!(
                            "MCP schema validation failed: {}",
                            msg
                        )));
                    }
                    return Ok(());
                }
                Err(e) => {
                    // Fallback to simple validation below
                    eprintln!("JSONSchema compile error: {}", e);
                }
            }
        }
        // Simple validation
        let obj = match args {
            serde_json::Value::Object(_) => args,
            _ => {
                return Err(ExecutorError::ArgumentError(
                    "MCP tool_call args must be object".to_string(),
                ))
            }
        };
        if let Some(req) = schema.get("required").and_then(|v| v.as_array()) {
            for k in req {
                let key = k.as_str().unwrap_or("");
                if key.is_empty() {
                    continue;
                }
                if obj.get(key).is_none() {
                    return Err(ExecutorError::ArgumentError(format!(
                        "MCP schema: missing arg '{}'",
                        key
                    )));
                }
            }
        }
        if let Some(props) = schema.get("properties").and_then(|v| v.as_object()) {
            for (k, tval) in props {
                let expected = tval.as_str().unwrap_or("");
                if expected.is_empty() {
                    continue;
                }
                if let Some(av) = obj.get(k) {
                    let ok = match expected {
                        "string" => av.is_string(),
                        "number" => av.is_number(),
                        "boolean" => av.is_boolean(),
                        "object" => av.is_object(),
                        "array" => av.is_array(),
                        _ => true,
                    };
                    if !ok {
                        return Err(ExecutorError::ArgumentError(format!(
                            "MCP schema: '{}' expected {}, got {}",
                            k, expected, av
                        )));
                    }
                }
            }
        }
        Ok(())
    }
    fn mcp_concurrency_check(name: &str, meta: &serde_json::Value) -> Result<(), ExecutorError> {
        let max_c = meta
            .get("max_concurrency")
            .and_then(|v| v.as_u64())
            .unwrap_or(u64::MAX) as usize;
        if max_c == usize::MAX {
            return Ok(());
        }
        let mut active = MCP_ACTIVE.write().unwrap();
        let cur = active.get(name).copied().unwrap_or(0);
        if cur >= max_c {
            return Err(ExecutorError::RuntimeError(format!(
                "MCP tool '{}' concurrency limit reached ({})",
                name, max_c
            )));
        }
        active.insert(name.to_string(), cur + 1);
        Ok(())
    }
    fn mcp_concurrency_done(name: &str) {
        let mut active = MCP_ACTIVE.write().unwrap();
        if let Some(cur) = active.get_mut(name) {
            if *cur > 0 {
                *cur -= 1;
            }
        }
    }
    fn mcp_permission_profile() -> String {
        std::env::var("LEXON_MCP_PERMISSION_PROFILE").unwrap_or_else(|_| "standard".to_string())
    }

    fn mcp_is_tool_allowed(profile: &str, name: &str) -> bool {
        if profile.eq_ignore_ascii_case("readonly") {
            // Allow read-only queries and discovery
            matches!(
                name,
                "read_file" | "tool_info" | "list_tools" | "rpc.discover" | "ping"
            )
        } else {
            true
        }
    }

    fn mcp_append_audit(&self, entry: &serde_json::Value) {
        if let Some(base) = &self.config.memory_path {
            let path = std::path::Path::new(base).join("mcp_audit.json");
            let mut items: Vec<serde_json::Value> = Vec::new();
            if path.exists() {
                if let Ok(s) = std::fs::read_to_string(&path) {
                    if let Ok(mut arr) = serde_json::from_str::<Vec<serde_json::Value>>(&s) {
                        items.append(&mut arr);
                    }
                }
            }
            let mut e = entry.clone();
            // add timestamp
            let ts = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis();
            if let Some(obj) = e.as_object_mut() {
                obj.insert("ts_ms".to_string(), serde_json::json!(ts));
            }
            items.push(e);
            let _ = std::fs::create_dir_all(path.parent().unwrap_or(std::path::Path::new(".")));
            let _ = std::fs::write(
                path,
                serde_json::to_string_pretty(&items).unwrap_or("[]".to_string()),
            );
        }
    }
    #[allow(dead_code)]
    pub(crate) fn handle_mcp_list_tools(
        &mut self,
        _args: &[LexExpression],
        result: Option<&ValueRef>,
    ) -> Result<(), ExecutorError> {
        // Load persisted registry first if present
        if let Some(base) = &self.config.memory_path {
            let path = std::path::Path::new(base).join("mcp_tools.json");
            if path.exists() {
                if let Ok(s) = std::fs::read_to_string(&path) {
                    if let Ok(map) = serde_json::from_str::<
                        std::collections::HashMap<String, serde_json::Value>,
                    >(&s)
                    {
                        let mut reg = MCP_TOOLS.write().unwrap();
                        for (k, v) in map {
                            reg.entry(k).or_insert(v);
                        }
                    }
                }
            }
        }
        let mut tools = Vec::new();
        for (name, meta) in MCP_TOOLS.read().unwrap().iter() {
            tools.push(serde_json::json!({"name": name, "meta": meta}));
        }
        if tools.is_empty() {
            tools.push(serde_json::json!({"name":"emit_file"}));
            tools.push(serde_json::json!({"name":"read_file"}));
            tools.push(serde_json::json!({"name":"write_file"}));
        }
        let tools = serde_json::json!({"tools": tools});
        if let Some(res) = result {
            self.store_value(res, RuntimeValue::Json(tools))?;
        }
        Ok(())
    }

    #[allow(dead_code)]
    pub(crate) fn handle_mcp_start_stdio(
        &mut self,
        _args: &[LexExpression],
        result: Option<&ValueRef>,
    ) -> Result<(), ExecutorError> {
        // Stub: pretend server started
        let info = serde_json::json!({"server": "mcp-stdio", "status": "started"});
        if let Some(res) = result {
            self.store_value(res, RuntimeValue::Json(info))?;
        }
        Ok(())
    }

    #[allow(dead_code)]
    pub(crate) fn handle_mcp_register_tools(
        &mut self,
        args: &[LexExpression],
        result: Option<&ValueRef>,
    ) -> Result<(), ExecutorError> {
        // mcp.register_tools({ name: {"schema": {...}, "desc": "..." }, ... })
        if args.is_empty() {
            return Err(ExecutorError::ArgumentError(
                "mcp.register_tools requires a JSON object".to_string(),
            ));
        }
        let obj = match self.evaluate_expression(args[0].clone())? {
            RuntimeValue::Json(serde_json::Value::Object(map)) => map,
            _ => {
                return Err(ExecutorError::ArgumentError(
                    "mcp.register_tools requires object".to_string(),
                ))
            }
        };
        {
            let mut reg = MCP_TOOLS.write().unwrap();
            for (k, v) in obj {
                reg.insert(k, v);
            }
        }
        // Persist registry if memory_path configured
        if let Some(base) = &self.config.memory_path {
            let path = std::path::Path::new(base).join("mcp_tools.json");
            if let Ok(reg) = MCP_TOOLS.read() {
                let json = serde_json::to_string_pretty(&*reg).unwrap_or("{}".to_string());
                let _ = std::fs::create_dir_all(path.parent().unwrap_or(std::path::Path::new(".")));
                let _ = std::fs::write(path, json);
            }
        }
        if let Some(res) = result {
            self.store_value(res, RuntimeValue::Boolean(true))?;
        }
        Ok(())
    }

    pub fn handle_mcp_run_stdio(
        &mut self,
        _args: &[LexExpression],
        result: Option<&ValueRef>,
    ) -> Result<(), ExecutorError> {
        use std::collections::HashSet;
        use std::io::{self, BufRead, Write};
        use std::sync::{
            atomic::{AtomicBool, Ordering},
            Arc, Mutex,
        };
        use std::thread;
        #[cfg(feature = "otel")]
        let _span_guard = {
            use tracing::info_span;
            info_span!("mcp_stdio").entered()
        };

        // One-shot mode for CI/smoke: print a JSON banner and exit
        if std::env::var("LEXON_MCP_ONESHOT").unwrap_or_default() == "1" {
            let banner = serde_json::json!({
                "server": "mcp-stdio",
                "status": "ready",
                "tools": ["emit_file","read_file","write_file"],
            });
            let mut stdout = io::stdout();
            writeln!(&mut stdout, "{}", banner).ok();
            stdout.flush().ok();
            if let Some(res) = result {
                self.store_value(res, RuntimeValue::Boolean(true))?;
            }
            return Ok(());
        }

        // Minimal JSON-RPC 2.0 loop over stdio
        // Accepts lines containing JSON objects: {"jsonrpc":"2.0","id":X,"method":"list_tools"}
        // Responds with {"jsonrpc":"2.0","id":X,"result":...}
        let stdin = io::stdin();
        let cancelled: Arc<Mutex<HashSet<String>>> = Arc::new(Mutex::new(HashSet::new()));
        let required_token = std::env::var("LEXON_MCP_AUTH_BEARER").ok();
        let jwt_secret = std::env::var("LEXON_MCP_AUTH_JWT_HS256").ok();
        let mut authorized = required_token.is_none() && jwt_secret.is_none();
        let mut identity = String::from(if authorized { "anon" } else { "unauth" });

        // Ensure builtin tool meta exists (schema + max_concurrency)
        {
            let mut reg = MCP_TOOLS.write().unwrap();
            reg.entry("web.search".to_string())
                .or_insert(serde_json::json!({
                    "description": "Search the web; args {query,n}",
                    "schema": {
                        "required": ["query"],
                        "properties": { "query": "string", "n": "number" }
                    },
                    "max_concurrency": 4
                }));
            reg.entry("read_file".to_string())
                .or_insert(serde_json::json!({
                    "description": "Read file from disk",
                    "schema": { "required": ["path"], "properties": { "path": "string" } },
                    "max_concurrency": 8
                }));
            reg.entry("write_file".to_string()).or_insert(serde_json::json!({
                "description": "Write file to disk",
                "schema": { "required": ["path","content"], "properties": { "path": "string", "content": "string" } },
                "max_concurrency": 4
            }));
        }

        // Heartbeat sender (optional)
        let heartbeat_ms = std::env::var("LEXON_MCP_HEARTBEAT_MS")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(0);
        let hb_running = Arc::new(AtomicBool::new(true));
        if heartbeat_ms > 0 {
            let hb_flag = hb_running.clone();
            thread::spawn(move || {
                let mut stdout = io::stdout();
                while hb_flag.load(Ordering::Relaxed) {
                    // Avoid external deps: simple millis timestamp
                    let ts = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_millis();
                    let beat = serde_json::json!({"jsonrpc":"2.0","method":"heartbeat","params":{"ts_ms": ts}});
                    let _ = writeln!(&mut stdout, "{}", beat);
                    let _ = stdout.flush();
                    std::thread::sleep(std::time::Duration::from_millis(heartbeat_ms));
                }
            });
        }
        for line in stdin.lock().lines() {
            let line = match line {
                Ok(s) => s.trim().to_string(),
                Err(_) => break,
            };
            if line.is_empty() {
                continue;
            }
            if line.eq_ignore_ascii_case("exit") {
                break;
            }
            let parsed: Result<serde_json::Value, _> = serde_json::from_str(&line);
            match parsed {
                Ok(val) => {
                    let id = val.get("id").cloned().unwrap_or(serde_json::Value::Null);
                    let method = val.get("method").and_then(|m| m.as_str()).unwrap_or("");
                    match method {
                        "auth" => {
                            let bearer = val
                                .get("params")
                                .and_then(|p| p.get("bearer"))
                                .and_then(|t| t.as_str())
                                .unwrap_or("");
                            let jwt = val
                                .get("params")
                                .and_then(|p| p.get("jwt"))
                                .and_then(|t| t.as_str());
                            authorized = false;
                            if let Some(exp) = &required_token {
                                if !bearer.is_empty() && bearer == exp {
                                    authorized = true;
                                    identity = Self::mcp_identity_from_bearer(bearer);
                                }
                            }
                            if !authorized {
                                if let (Some(secret), Some(tok)) = (jwt_secret.as_deref(), jwt) {
                                    if Self::jwt_verify_hs256(tok, secret) {
                                        authorized = true;
                                        identity = if let Ok((_h, p)) = Self::jwt_decode_parts(tok)
                                        {
                                            p.get("sub")
                                                .and_then(|v| v.as_str())
                                                .map(|s| format!("jwt:{}", s))
                                                .unwrap_or_else(|| "jwt:unknown".to_string())
                                        } else {
                                            "jwt:unknown".to_string()
                                        };
                                    }
                                }
                            }
                            let out = serde_json::json!({"jsonrpc":"2.0","id": id, "result": {"ok": authorized}});
                            println!("{}", out);
                        }
                        "list_tools" => {
                            if !authorized {
                                let err = serde_json::json!({"jsonrpc":"2.0","id": id, "error": {"code": -32000, "message": "unauthorized"}});
                                println!("{}", err);
                                continue;
                            }
                            if let Err(e) = Self::mcp_check_rate(&identity) {
                                let err = serde_json::json!({"jsonrpc":"2.0","id": id, "error": {"code": -32015, "message": format!("{}", e)}});
                                println!("{}", err);
                                continue;
                            }
                            let mut list = vec![
                                serde_json::json!({"name": "emit_file", "description": "Emit binary/text file from model"}),
                                serde_json::json!({"name": "read_file", "description": "Read file from disk"}),
                                serde_json::json!({"name": "write_file", "description": "Write file to disk"}),
                                serde_json::json!({"name": "web.search", "description": "Search the web; args {query,n}"}),
                            ];
                            if let Ok(reg) = MCP_TOOLS.read() {
                                for (name, meta) in reg.iter() {
                                    let desc = meta
                                        .get("description")
                                        .and_then(|v| v.as_str())
                                        .unwrap_or("");
                                    list.push(
                                        serde_json::json!({"name": name, "description": desc}),
                                    );
                                }
                            }
                            let tools = serde_json::json!({
                                "jsonrpc":"2.0",
                                "id": id,
                                "result": { "tools": list }
                            });
                            println!("{}", tools);
                        }
                        "tool_info" => {
                            if !authorized {
                                let err = serde_json::json!({"jsonrpc":"2.0","id": id, "error": {"code": -32000, "message": "unauthorized"}});
                                println!("{}", err);
                                continue;
                            }
                            if let Err(e) = Self::mcp_check_rate(&identity) {
                                let err = serde_json::json!({"jsonrpc":"2.0","id": id, "error": {"code": -32015, "message": format!("{}", e)}});
                                println!("{}", err);
                                continue;
                            }
                            let name = val
                                .get("params")
                                .and_then(|p| p.get("name"))
                                .and_then(|x| x.as_str())
                                .unwrap_or("");
                            // For stdio, reflect in-memory only
                            let mut meta = serde_json::json!({});
                            if let Ok(reg) = MCP_TOOLS.read() {
                                if let Some(m) = reg.get(name) {
                                    meta = m.clone();
                                }
                            }
                            let out = serde_json::json!({"jsonrpc":"2.0","id": id, "result": {"name": name, "meta": meta}});
                            println!("{}", out);
                        }
                        "tool_call" => {
                            if !authorized {
                                let err = serde_json::json!({"jsonrpc":"2.0","id": id, "error": {"code": -32000, "message": "unauthorized"}});
                                println!("{}", err);
                                continue;
                            }
                            if let Err(e) = Self::mcp_check_rate(&identity) {
                                let err = serde_json::json!({"jsonrpc":"2.0","id": id, "error": {"code": -32015, "message": format!("{}", e)}});
                                println!("{}", err);
                                continue;
                            }
                            let params = val
                                .get("params")
                                .and_then(|p| p.as_object())
                                .cloned()
                                .unwrap_or_default();
                            let name = params.get("name").and_then(|x| x.as_str()).unwrap_or("");
                            let profile = Self::mcp_permission_profile();
                            if !Self::mcp_is_tool_allowed(&profile, name) {
                                let err = serde_json::json!({"jsonrpc":"2.0","id": id, "error": {"code": -32010, "message": format!("Tool '{}' not allowed under profile '{}'", name, profile)}});
                                println!("{}", err);
                                continue;
                            }
                            if !Self::mcp_acl_is_allowed(&identity, name) {
                                let err = serde_json::json!({"jsonrpc":"2.0","id": id, "error": {"code": -32016, "message": format!("Tool '{}' not allowed for identity '{}'", name, identity)}});
                                println!("{}", err);
                                continue;
                            }
                            let args = params.get("args").cloned().unwrap_or(serde_json::json!({}));
                            // Validate against schema and concurrency
                            if let Ok(reg) = MCP_TOOLS.read() {
                                if let Some(meta) = reg.get(name) {
                                    if let Some(schema) = meta.get("schema") {
                                        if let Err(e) =
                                            Self::mcp_validate_args_schema(schema, &args)
                                        {
                                            let err = serde_json::json!({"jsonrpc":"2.0","id": id, "error": {"code": -32012, "message": format!("{}", e)}});
                                            println!("{}", err);
                                            continue;
                                        }
                                    }
                                    if let Err(e) = Self::mcp_concurrency_check(name, meta) {
                                        let err = serde_json::json!({"jsonrpc":"2.0","id": id, "error": {"code": -32013, "message": format!("{}", e)}});
                                        println!("{}", err);
                                        continue;
                                    }
                                }
                            }
                            // Streaming progress (optional): env LEXON_MCP_STREAM=1
                            let do_stream = std::env::var("LEXON_MCP_STREAM")
                                .ok()
                                .map(|v| v == "1")
                                .unwrap_or(false);
                            let progress_running = Arc::new(AtomicBool::new(true));
                            let id_clone = id.clone();
                            if do_stream {
                                let name_s = name.to_string();
                                let pr_flag = progress_running.clone();
                                thread::spawn(move || {
                                    let mut stdout = io::stdout();
                                    while pr_flag.load(Ordering::Relaxed) {
                                        let ev = serde_json::json!({"jsonrpc":"2.0","method":"progress","params":{"tool": name_s, "id": id_clone}});
                                        let _ = writeln!(&mut stdout, "{}", ev);
                                        let _ = stdout.flush();
                                        std::thread::sleep(std::time::Duration::from_millis(300));
                                    }
                                });
                            }
                            // Execute tool inline (progress thread provides streaming ticks)
                            let res = self.execute_tool_call_json(name, &args);
                            match res {
                                Ok(v) => {
                                    self.mcp_append_audit(
                                        &serde_json::json!({"tool": name, "ok": true}),
                                    );
                                    let out = serde_json::json!({"jsonrpc":"2.0","id": id, "result": {"name": name, "output": v}});
                                    println!("{}", out);
                                }
                                Err(e) => {
                                    self.mcp_append_audit(&serde_json::json!({"tool": name, "ok": false, "error": format!("{}", e)}));
                                    let err = serde_json::json!({"jsonrpc":"2.0","id": id, "error": {"code": -32011, "message": format!("{}", e)}});
                                    println!("{}", err);
                                }
                            }
                            progress_running.store(false, Ordering::Relaxed);
                            // release slot
                            Self::mcp_concurrency_done(name);
                            // Reset cooperative cancel flag after handling one call
                            self.cancel_requested = false;
                        }
                        "set_quota" => {
                            let params = val
                                .get("params")
                                .and_then(|p| p.as_object())
                                .cloned()
                                .unwrap_or_default();
                            let name = params.get("name").and_then(|x| x.as_str()).unwrap_or("");
                            let quota = params
                                .get("allowed_calls")
                                .and_then(|x| x.as_u64())
                                .unwrap_or(0);
                            if let Ok(mut reg) = MCP_TOOLS.write() {
                                let entry =
                                    reg.entry(name.to_string()).or_insert(serde_json::json!({}));
                                if let Some(obj) = entry.as_object_mut() {
                                    obj.insert(
                                        "allowed_calls".to_string(),
                                        serde_json::json!(quota),
                                    );
                                }
                            }
                            let out = serde_json::json!({"jsonrpc":"2.0","id": id, "result": {"ok": true}});
                            println!("{}", out);
                        }
                        "ping" => {
                            let pong = serde_json::json!({"jsonrpc":"2.0","id": id, "result": {"pong": true}});
                            println!("{}", pong);
                        }
                        "rpc.discover" => {
                            let out = serde_json::json!({"jsonrpc":"2.0","id": id, "result": {"methods": ["ping","list_tools","tool_info","set_quota","rpc.cancel"], "version": "1.1"}});
                            println!("{}", out);
                        }
                        "rpc.cancel" => {
                            // Mark id as cancelled
                            if let Some(i) = id.as_i64() {
                                cancelled.lock().unwrap().insert(i.to_string());
                            }
                            // Set cooperative cancel flag in env
                            self.cancel_requested = true;
                            let ack = serde_json::json!({"jsonrpc":"2.0","id": id, "result": {"cancelled": true}});
                            println!("{}", ack);
                        }

                        _ => {
                            let err = serde_json::json!({
                                "jsonrpc":"2.0",
                                "id": id,
                                "error": {"code": -32601, "message": format!("Method not found: {}", method)}
                            });
                            println!("{}", err);
                        }
                    }
                    io::stdout().flush().ok();
                }
                Err(_) => {
                    // Not JSON; echo ack
                    let ack = serde_json::json!({"ok": true, "received": line});
                    println!("{}", ack);
                    io::stdout().flush().ok();
                }
            }
        }
        // stop heartbeat
        // (if not started, flag is false by default)
        // Only drop if set earlier
        #[allow(unused_must_use)]
        {
            hb_running.store(false, Ordering::Relaxed);
        }
        if let Some(res) = result {
            self.store_value(res, RuntimeValue::Boolean(true))?;
        }
        Ok(())
    }

    #[allow(dead_code)]
    pub(crate) fn handle_mcp_run_ws(
        &mut self,
        args: &[LexExpression],
        result: Option<&ValueRef>,
    ) -> Result<(), ExecutorError> {
        // mcp.run_ws(url)
        if args.is_empty() {
            return Err(ExecutorError::ArgumentError(
                "mcp.run_ws requires url".to_string(),
            ));
        }
        let url = match self.evaluate_expression(args[0].clone())? {
            RuntimeValue::String(s) => s,
            v => format!("{:?}", v),
        };
        // No registry needed for client oneshot
        let r = tokio::task::block_in_place(move || {
            tokio::runtime::Handle::current().block_on(async move {
                use futures_util::{SinkExt, StreamExt};
                use tokio_tungstenite::connect_async;
                let c_to_ms: u64 = std::env::var("LEXON_MCP_WS_CONNECT_TIMEOUT_MS")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(1000);
                match tokio::time::timeout(
                    std::time::Duration::from_millis(c_to_ms),
                    connect_async(url),
                )
                .await
                {
                    Ok(Ok((mut ws, _resp))) => {
                        // Send hello (ping) and list_tools if configured
                        let ping =
                            serde_json::json!({"jsonrpc":"2.0","id":1,"method":"ping"}).to_string();
                        let _ = ws
                            .send(tokio_tungstenite::tungstenite::Message::Text(ping))
                            .await;
                        if std::env::var("LEXON_MCP_WS_LIST_TOOLS").ok().as_deref() == Some("1") {
                            let list =
                                serde_json::json!({"jsonrpc":"2.0","id":2,"method":"list_tools"})
                                    .to_string();
                            let _ = ws
                                .send(tokio_tungstenite::tungstenite::Message::Text(list))
                                .await;
                        }
                        // Try to read one message (optional)
                        let _ = ws.next().await;
                        let _ = ws.close(None).await;
                        Ok(())
                    }
                    Ok(Err(e)) => Err(ExecutorError::RuntimeError(format!(
                        "WS connect error: {}",
                        e
                    ))),
                    Err(_) => Err(ExecutorError::RuntimeError(
                        "WS connect timeout".to_string(),
                    )),
                }
            })
        });
        match r {
            Ok(()) => {
                if let Some(res) = result {
                    self.store_value(res, RuntimeValue::Boolean(true))?;
                }
                Ok(())
            }
            Err(e) => Err(e),
        }
    }

    pub fn handle_mcp_run_ws_server(
        &mut self,
        args: &[LexExpression],
        result: Option<&ValueRef>,
    ) -> Result<(), ExecutorError> {
        // mcp.run_ws_server(addr?)
        let addr_str = if !args.is_empty() {
            match self.evaluate_expression(args[0].clone())? {
                RuntimeValue::String(s) => s,
                v => format!("{:?}", v),
            }
        } else {
            std::env::var("LEXON_MCP_WS_ADDR").unwrap_or_else(|_| "127.0.0.1:8080".to_string())
        };
        let oneshot = std::env::var("LEXON_MCP_SERVER_ONESHOT").ok().as_deref() == Some("1");
        // Clone tool registry for use inside async move
        // use MCP_TOOLS static directly
        let base_path = self.config.memory_path.clone();
        let config = self.config.clone();

        let r = tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async move {
                use futures_util::{SinkExt, StreamExt};
                use tokio::net::TcpListener;
                use tokio_tungstenite::{accept_hdr_async, tungstenite::Message};
                use tokio_tungstenite::tungstenite::handshake::server::{Request, Response, ErrorResponse};
                use tokio_rustls::rustls;
                // TLS optional
                let tls_cfg = {
                    let cert_path = std::env::var("LEXON_MCP_TLS_CERT").ok();
                    let key_path = std::env::var("LEXON_MCP_TLS_KEY").ok();
                    let client_ca_path = std::env::var("LEXON_MCP_TLS_CLIENT_CA").ok();
                    if let (Some(cert), Some(key)) = (cert_path, key_path) {
                        let mut cert_reader = std::io::BufReader::new(std::fs::File::open(cert).map_err(|e| ExecutorError::RuntimeError(format!("TLS cert open: {}", e)))?);
                        let mut key_reader = std::io::BufReader::new(std::fs::File::open(key).map_err(|e| ExecutorError::RuntimeError(format!("TLS key open: {}", e)))?);
                        let certs = rustls_pemfile::certs(&mut cert_reader)
                            .map_err(|_| ExecutorError::RuntimeError("TLS cert parse error".to_string()))?
                            .into_iter()
                            .map(rustls::Certificate)
                            .collect::<Vec<_>>();
                        let keys = rustls_pemfile::pkcs8_private_keys(&mut key_reader)
                            .map_err(|_| ExecutorError::RuntimeError("TLS key parse error".to_string()))?;
                        if certs.is_empty() || keys.is_empty() {
                            return Err(ExecutorError::RuntimeError("TLS cert/key missing entries".to_string()));
                        }
                        let key = rustls::PrivateKey(keys[0].clone());
                        let cfg = if let Some(ca) = client_ca_path {
                            let mut ca_reader = std::io::BufReader::new(std::fs::File::open(ca).map_err(|e| ExecutorError::RuntimeError(format!("TLS client CA open: {}", e)))?);
                            let ca_certs = rustls_pemfile::certs(&mut ca_reader).map_err(|_| ExecutorError::RuntimeError("TLS client CA parse".to_string()))?;
                            let mut roots = rustls::RootCertStore::empty();
                            for der in ca_certs { let _ = roots.add(&rustls::Certificate(der)); }
                            let verifier = rustls::server::AllowAnyAuthenticatedClient::new(roots);
                            rustls::ServerConfig::builder()
                                .with_safe_defaults()
                                .with_client_cert_verifier(std::sync::Arc::new(verifier))
                                .with_single_cert(certs, key)
                                .map_err(|e| ExecutorError::RuntimeError(format!("TLS config: {}", e)))?
                        } else {
                            rustls::ServerConfig::builder()
                                .with_safe_defaults()
                                .with_no_client_auth()
                                .with_single_cert(certs, key)
                                .map_err(|e| ExecutorError::RuntimeError(format!("TLS config: {}", e)))?
                        };
                        Some(std::sync::Arc::new(cfg))
                    } else {
                        None
                    }
                };
                let expected_bearer = std::env::var("LEXON_MCP_AUTH_BEARER").ok();

                let listener = TcpListener::bind(&addr_str).await.map_err(|e| ExecutorError::RuntimeError(format!("bind {}: {}", addr_str, e)))?;
                println!("🔌 MCP WS server listening on {}", addr_str);
                // Ensure builtin tool meta exists (schema + max_concurrency)
                {
                    if let Ok(mut reg) = MCP_TOOLS.write() {
                        reg.entry("web.search".to_string()).or_insert(serde_json::json!({
                            "description": "Search the web; args {query,n}",
                            "schema": {
                                "required": ["query"],
                                "properties": { "query": "string", "n": "number" }
                            },
                            "max_concurrency": 4
                        }));
                        reg.entry("read_file".to_string()).or_insert(serde_json::json!({
                            "description": "Read file from disk",
                            "schema": { "required": ["path"], "properties": { "path": "string" } },
                            "max_concurrency": 8
                        }));
                        reg.entry("write_file".to_string()).or_insert(serde_json::json!({
                            "description": "Write file to disk",
                            "schema": { "required": ["path","content"], "properties": { "path": "string", "content": "string" } },
                            "max_concurrency": 4
                        }));
                    }
                }
                // Optional self-test: connect to our own server and send ping so accept() completes
                if std::env::var("LEXON_MCP_SELF_TEST").ok().as_deref() == Some("1") {
                    let addr_clone = addr_str.clone();
                    tokio::spawn(async move {
                        use tokio_tungstenite::connect_async;
                        let url = format!("ws://{}", addr_clone);
                        let _ = tokio::time::sleep(std::time::Duration::from_millis(50)).await;
                        if let Ok((mut ws, _)) = connect_async(url).await {
                            let ping = serde_json::json!({"jsonrpc":"2.0","id":1,"method":"ping"}).to_string();
                            let _ = ws.send(tokio_tungstenite::tungstenite::Message::Text(ping)).await;
                            let _ = ws.close(None).await;
                        }
                    });
                }
                let mut handled = 0usize;
                loop {
                    let accept_fut = listener.accept();
                    let res = if oneshot { tokio::time::timeout(std::time::Duration::from_secs(30), accept_fut).await } else { Ok(accept_fut.await) };
                    let (stream, _peer) = match res {
                        Ok(Ok(tuple)) => tuple,
                        Ok(Err(e)) => { return Err(ExecutorError::RuntimeError(format!("accept error: {}", e))); }
                        Err(_timeout) => { println!("⏱️ WS accept timed out"); break; }
                    };

                    // WS handshake with optional auth header validation and TLS
                    let auth_checker = |req: &Request, resp: Response| {
                        if let Some(exp) = &expected_bearer {
                            let ok = req.headers()
                                .get("authorization")
                                .and_then(|v| v.to_str().ok())
                                .map(|s| s == format!("Bearer {}", exp))
                                .unwrap_or(false);
                            if !ok {
                                return Err(ErrorResponse::new(Some("Unauthorized".into())));
                            }
                        }
                        Ok(resp)
                    };
                    let (mut ws_writer, mut ws_reader): (WsSink, WsStream) = if let Some(cfg) = &tls_cfg {
                        let acceptor = tokio_rustls::TlsAcceptor::from(cfg.clone());
                        let tls_stream = acceptor.accept(stream).await.map_err(|e| ExecutorError::RuntimeError(format!("tls handshake: {}", e)))?;
                        let ws = accept_hdr_async(tls_stream, auth_checker).await.map_err(|e| ExecutorError::RuntimeError(format!("handshake error: {}", e)))?;
                        let (w, r) = ws.split();
                        (Box::new(w), Box::new(r))
                    } else {
                        let ws = accept_hdr_async(stream, auth_checker).await.map_err(|e| ExecutorError::RuntimeError(format!("handshake error: {}", e)))?;
                        let (w, r) = ws.split();
                        (Box::new(w), Box::new(r))
                    };
                    #[allow(unused)]
                    {
                        #[cfg(feature = "otel")]
                        {
                            use tracing::info_span;
                            let _span = info_span!("mcp_ws_connection", addr = addr_str.as_str()).entered();
                        }
                    }
                    // Send ready banner
                    let banner = serde_json::json!({"jsonrpc":"2.0","id":0,"result": {"server":"lexon-mcp","status":"ready"}}).to_string();
                    let _ = ws_writer.send(Message::Text(banner)).await;
                    // Optional heartbeat
                    let heartbeat_ms: u64 = std::env::var("LEXON_MCP_HEARTBEAT_MS").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
                    let mut hb = if heartbeat_ms > 0 {
                        Some(tokio::time::interval(std::time::Duration::from_millis(heartbeat_ms)))
                    } else { None };

                    loop {
                        tokio::select! {
                            // Heartbeat tick
                            _ = async {
                                if let Some(h) = &mut hb { h.tick().await; }
                            }, if hb.is_some() => {
                                let ts = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_millis();
                                let beat = serde_json::json!({"jsonrpc":"2.0","method":"heartbeat","params":{"ts_ms": ts}}).to_string();
                                let _ = ws_writer.send(Message::Text(beat)).await;
                            }
                            // Incoming message
                            incoming = ws_reader.next() => {
                                match incoming {
                                    Some(msg) => {
                        match msg {
                            Ok(Message::Text(txt)) => {
                                #[allow(unused)]
                                {
                                    #[cfg(feature = "otel")]
                                    {
                                        use tracing::info_span;
                                        let _span = info_span!("mcp_ws_message").entered();
                                    }
                                }
                                let parsed: Result<serde_json::Value, _> = serde_json::from_str(&txt);
                                match parsed {
                                    Ok(v) => {
                                        let id = v.get("id").cloned().unwrap_or(serde_json::Value::Null);
                                        let method = v.get("method").and_then(|m| m.as_str()).unwrap_or("");
                                        // Optional per-message timeout
                                        let msg_timeout_ms: u64 = std::env::var("LEXON_MCP_MSG_TIMEOUT_MS").ok().and_then(|v| v.parse().ok()).unwrap_or(3000);
                                        let resp = match method {
                                            "ping" => serde_json::json!({"jsonrpc":"2.0","id": id, "result": {"pong": true}}),
                                            "list_tools" => {
                                                // reflect registered tools (live)
                                                let mut tools = Vec::new();
                                                if let Ok(reg) = MCP_TOOLS.read() {
                                                    for (k, v) in reg.iter() { tools.push(serde_json::json!({"name": k, "meta": v})); }
                                                }
                                                if tools.is_empty() { tools.push(serde_json::json!({"name":"emit_file"})); tools.push(serde_json::json!({"name":"read_file"})); tools.push(serde_json::json!({"name":"write_file"})); }
                                                serde_json::json!({"jsonrpc":"2.0","id": id, "result": {"tools": tools}})
                                            }
                                            "tool_call" => {
                                                #[cfg(feature = "otel")]
                                                let _span_guard = {
                                                    use tracing::info_span;
                                                    let name = v.get("params").and_then(|p| p.get("name")).and_then(|n| n.as_str()).unwrap_or("");
                                                    info_span!("mcp_tool_call_ws", tool = name).entered()
                                                };
                                                // tool_call { name, args }
                                                let params = v.get("params").and_then(|p| p.as_object()).cloned().unwrap_or_default();
                                                let name = params.get("name").and_then(|x| x.as_str()).unwrap_or("");
                                                let args = params.get("args").cloned().unwrap_or(serde_json::json!({}));
                                                // Permission profile
                                                let profile = Self::mcp_permission_profile();
                                                if !Self::mcp_is_tool_allowed(&profile, name) {
                                                    serde_json::json!({
                                                        "jsonrpc":"2.0","id": id,
                                                        "error": {"code": -32010, "message": format!("Tool '{}' not allowed under profile '{}'", name, profile)}
                                                    })
                                                } else {
                                                // Sandbox: optional allowlist from env
                                                let allowed = std::env::var("LEXON_MCP_ALLOWED_TOOLS").ok().unwrap_or_default();
                                                let not_allowed = if !allowed.is_empty() {
                                                    !allowed.split(',').any(|t| t.trim() == name)
                                                } else { false };
                                                if not_allowed {
                                                    serde_json::json!({
                                                        "jsonrpc":"2.0","id": id,
                                                        "error": {"code": -32001, "message": format!("Tool '{}' not allowed", name)}
                                                    })
                                                } else {
                                                    // Enforce concurrency and quota using in-process registry
                                                    let mut denied = false;
                                                    // Concurrency check (if meta.concurrency_limit set)
                                                    if let Ok(reg) = MCP_TOOLS.read() {
                                                        if let Some(meta) = reg.get(name) {
                                                            if let Err(_e) = Self::mcp_concurrency_check(name, meta) {
                                                                denied = true;
                                                            }
                                                        }
                                                    }
                                                    if let Ok(mut reg) = MCP_TOOLS.write() {
                                                        let entry = reg.entry(name.to_string()).or_insert(serde_json::json!({}));
                                                        let allowed = entry.get("allowed_calls").and_then(|v| v.as_u64());
                                                        let used = entry.get("used_calls").and_then(|v| v.as_u64()).unwrap_or(0);
                                                        if let Some(limit) = allowed {
                                                            if used >= limit { denied = true; }
                                                        }
                                                        if !denied {
                                                            let new_used = used.saturating_add(1);
                                                            if let Some(obj) = entry.as_object_mut() { obj.insert("used_calls".to_string(), serde_json::json!(new_used)); }
                                                        }
                                                        // Persist if configured
                                                        if let Some(base) = &base_path {
                                                            let path = std::path::Path::new(base).join("mcp_tools.json");
                                                            let json = serde_json::to_string_pretty(&*reg).unwrap_or("{}".to_string());
                                                            let _ = std::fs::create_dir_all(path.parent().unwrap_or(std::path::Path::new(".")));
                                                            let _ = std::fs::write(path, json);
                                                        }
                                                    }
                                                    if denied {
                                                        serde_json::json!({
                                                            "jsonrpc":"2.0","id": id,
                                                            "error": {"code": -32002, "message": format!("Tool '{}' quota exceeded", name)}
                                                        })
                                                    } else {
                                                        // Bridge to executor handlers for tracked tools with optional streaming progress
                                                        let do_stream = std::env::var("LEXON_MCP_STREAM").ok().map(|v| v=="1").unwrap_or(false);
                                                        // Prepare environment for tool
                                                        let mut env = super::ExecutionEnvironment::new(config.clone());
                                                        // Shared cancellation flag for cooperative cancel
                                                        let cancel_flag = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
                                                        env.set_cancel_flag(cancel_flag.clone());
                                                        if let Ok(reg) = MCP_TOOLS.read() {
                                                            if let Some(meta) = reg.get(name) {
                                                                if let Some(limit) = meta.get("allowed_calls").and_then(|v| v.as_u64()) {
                                                                    env.tool_registry.insert(name.to_string(), super::ToolMeta { allowed_calls: Some(limit), used_calls: 0, scope: None });
                                                                }
                                                            }
                                                        }
                                                        let tool_timeout_ms: u64 = std::env::var("LEXON_MCP_TOOL_TIMEOUT_MS").ok().and_then(|v| v.parse().ok()).unwrap_or(2000);
                                                        let name_s = name.to_string();
                                                        let id_clone = id.clone();
                                                        // Build the tool future (timeout-wrapped)
                                                        let tool_fut = tokio::time::timeout(
                                                            std::time::Duration::from_millis(tool_timeout_ms),
                                                            async move { env.execute_tool_call_json(name, &args) }
                                                        );
                                                        tokio::pin!(tool_fut);
                                                        // Optional progress interval
                                                        let mut pr = if do_stream {
                                                            Some(tokio::time::interval(std::time::Duration::from_millis(300)))
                                                        } else { None };
                                                        // Emit an initial progress tick so fast tools still surface progress
                                                        if do_stream {
                                                            let ev = serde_json::json!({"jsonrpc":"2.0","method":"progress","params":{"tool": name_s, "id": id_clone}});
                                                            let _ = ws_writer.send(tokio_tungstenite::tungstenite::Message::Text(ev.to_string())).await;
                                                        }
                                                        // Interleave tool execution with progress, heartbeats and listen for rpc.cancel
                                                        let exec_result = loop {
                                                            tokio::select! {
                                                                res = &mut tool_fut => break res,
                                                                // optional WS progress
                                                                _ = async {
                                                                    if let Some(p) = &mut pr { p.tick().await; }
                                                                }, if pr.is_some() => {
                                                                    let ev = serde_json::json!({"jsonrpc":"2.0","method":"progress","params":{"tool": name_s, "id": id_clone}});
                                                                    let _ = ws_writer.send(tokio_tungstenite::tungstenite::Message::Text(ev.to_string())).await;
                                                                }
                                                                // keep WS heartbeat active during long tool calls
                                                                _ = async {
                                                                    if let Some(h) = &mut hb { h.tick().await; }
                                                                }, if hb.is_some() => {
                                                                    let ts = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_millis();
                                                                    let beat = serde_json::json!({"jsonrpc":"2.0","method":"heartbeat","params":{"ts_ms": ts}});
                                                                    let _ = ws_writer.send(tokio_tungstenite::tungstenite::Message::Text(beat.to_string())).await;
                                                                }
                                                                // listen for rpc.cancel for this id
                                                                msg_opt = ws_reader.next() => {
                                                                    if let Some(Ok(tokio_tungstenite::tungstenite::Message::Text(txt))) = msg_opt {
                                                                        if let Ok(val) = serde_json::from_str::<serde_json::Value>(&txt) {
                                                                            let mid = val.get("id").and_then(|x| x.as_i64());
                                                                            let mname = val.get("method").and_then(|m| m.as_str()).unwrap_or("");
                                                                            if mname == "rpc.cancel" && mid == id.as_i64() {
                                                                                // cooperative: set cancel flag and ACK
                                                                                cancel_flag.store(true, std::sync::atomic::Ordering::Relaxed);
                                                                                let ack = serde_json::json!({"jsonrpc":"2.0","id": id, "result": {"cancelled": true}});
                                                                                let _ = ws_writer.send(tokio_tungstenite::tungstenite::Message::Text(ack.to_string())).await;
                                                                            } else {
                                                                                // Non-cancel messages during execution: optional nack
                                                                                let nack = serde_json::json!({"jsonrpc":"2.0","id": val.get("id").cloned().unwrap_or(serde_json::Value::Null), "error": {"code": -32005, "message": "busy"}});
                                                                                let _ = ws_writer.send(tokio_tungstenite::tungstenite::Message::Text(nack.to_string())).await;
                                                                            }
                                                                        }
                                                                    } else if msg_opt.is_none() {
                                                                        // client closed
                                                                        cancel_flag.store(true, std::sync::atomic::Ordering::Relaxed);
                                                                    }
                                                                }
                                                            }
                                                        };
                                                        let resp_val = match exec_result {
                                                            Ok(Ok(val)) => { append_audit(&base_path, &serde_json::json!({"tool": name, "ok": true})); serde_json::json!({"jsonrpc":"2.0","id": id, "result": {"name": name, "output": val}}) },
                                                            Ok(Err(e)) => { append_audit(&base_path, &serde_json::json!({"tool": name, "ok": false, "error": format!("{}", e)})); serde_json::json!({"jsonrpc":"2.0","id": id, "error": {"code": -32003, "message": format!("{}", e)}}) },
                                                            Err(_elapsed) => { append_audit(&base_path, &serde_json::json!({"tool": name, "ok": false, "error": "timeout"})); serde_json::json!({"jsonrpc":"2.0","id": id, "error": {"code": -32004, "message": "tool timeout"}}) },
                                                        };
                                                        // Release concurrency slot regardless of outcome
                                                        Self::mcp_concurrency_done(name);
                                                        resp_val
                                                    }
                                                }
                                                }
                                            }
                                            "tool_info" => {
                                                let params = v.get("params").and_then(|p| p.as_object()).cloned().unwrap_or_default();
                                                let name = params.get("name").and_then(|x| x.as_str()).unwrap_or("");
                                                let mut meta = serde_json::json!({});
                                                if let Ok(reg) = MCP_TOOLS.read() { if let Some(m) = reg.get(name) { meta = m.clone(); } }
                                                serde_json::json!({"jsonrpc":"2.0","id": id, "result": {"name": name, "meta": meta}})
                                            }
                                            ,"set_quota" => {
                                                let params = v.get("params").and_then(|p| p.as_object()).cloned().unwrap_or_default();
                                                let name = params.get("name").and_then(|x| x.as_str()).unwrap_or("");
                                                let quota = params.get("allowed_calls").and_then(|x| x.as_u64()).unwrap_or(0);
                                                // Store in in-process registry and persist if configured
                                                if let Ok(mut reg) = MCP_TOOLS.write() {
                                                    let entry = reg.entry(name.to_string()).or_insert(serde_json::json!({}));
                                                    if let Some(obj) = entry.as_object_mut() {
                                                        obj.insert("allowed_calls".to_string(), serde_json::json!(quota));
                                                        obj.insert("used_calls".to_string(), serde_json::json!(0u64));
                                                    }
                                                    if let Some(base) = &base_path {
                                                        let path = std::path::Path::new(base).join("mcp_tools.json");
                                                        let json = serde_json::to_string_pretty(&*reg).unwrap_or("{}".to_string());
                                                        let _ = std::fs::create_dir_all(path.parent().unwrap_or(std::path::Path::new(".")));
                                                        let _ = std::fs::write(path, json);
                                                    }
                                                }
                                                serde_json::json!({"jsonrpc":"2.0","id": id, "result": {"ok": true}})
                                            }
                                            ,"rpc.cancel" => serde_json::json!({"jsonrpc":"2.0","id": id, "result": {"cancelled": true}}),
                                            "rpc.discover" => serde_json::json!({
                                                "jsonrpc":"2.0","id": id,
                                                "result": {"methods": ["ping","list_tools","tool_call","rpc.cancel"], "version": "1.1"}
                                            }),
                                            _ => serde_json::json!({"jsonrpc":"2.0","id": id, "error": {"code": -32601, "message": format!("Method not found: {}", method)}}),
                                        };
                                        // Enforce send timeout to avoid hanging
                                        let send_fut = ws_writer.send(Message::Text(resp.to_string()));
                                        let _ = tokio::time::timeout(std::time::Duration::from_millis(msg_timeout_ms), send_fut).await;
                                        if method == "exit" { break; }
                                    }
                                    Err(_) => {
                                        let ack = serde_json::json!({"jsonrpc":"2.0","id": null, "error": {"code": -32700, "message":"Parse error"}});
                                        let _ = ws_writer.send(Message::Text(ack.to_string())).await;
                                    }
                                }
                            }
                            Ok(Message::Close(_)) => break,
                            Ok(_) => continue,
                            Err(_) => break,
                        }
                                    }
                                    None => break,
                                }
                            }
                        }
                    }
                    let _ = futures_util::SinkExt::close(&mut ws_writer).await;
                    handled += 1;
                    if oneshot { break; }
                }
                println!("🔌 MCP WS server exiting (handled {} connections)", handled);
                Ok(())
            })
        });
        match r {
            Ok(()) => {
                if let Some(res) = result {
                    self.store_value(res, RuntimeValue::Boolean(true))?;
                }
                Ok(())
            }
            Err(e) => Err(e),
        }
    }
}
