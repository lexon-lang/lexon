// lexc/src/executor/mcp.rs
// Minimal MCP 1.1 scaffold (no server yet): list_tools handler stub

use crate::lexir::LexExpression;
use super::{ExecutionEnvironment, ExecutorError, RuntimeValue, ValueRef};
use std::collections::HashMap;
use std::sync::RwLock;
use once_cell::sync::Lazy;

static MCP_TOOLS: Lazy<RwLock<HashMap<String, serde_json::Value>>> = Lazy::new(|| RwLock::new(HashMap::new()));

fn append_audit(base: &Option<String>, entry: &serde_json::Value) {
    if let Some(base) = base {
        let path = std::path::Path::new(base).join("mcp_audit.json");
        let mut items: Vec<serde_json::Value> = Vec::new();
        if path.exists() {
            if let Ok(s) = std::fs::read_to_string(&path) {
                if let Ok(mut arr) = serde_json::from_str::<Vec<serde_json::Value>>(&s) { items.append(&mut arr); }
            }
        }
        let mut e = entry.clone();
        let ts = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_millis();
        if let Some(obj) = e.as_object_mut() { obj.insert("ts_ms".to_string(), serde_json::json!(ts)); }
        items.push(e);
        let _ = std::fs::create_dir_all(path.parent().unwrap_or(std::path::Path::new(".")));
        let _ = std::fs::write(path, serde_json::to_string_pretty(&items).unwrap_or("[]".to_string()));
    }
}

impl ExecutionEnvironment {
    fn mcp_permission_profile() -> String {
        std::env::var("LEXON_MCP_PERMISSION_PROFILE").unwrap_or_else(|_| "standard".to_string())
    }

    fn mcp_is_tool_allowed(profile: &str, name: &str) -> bool {
        if profile.eq_ignore_ascii_case("readonly") {
            // Allow read-only queries and discovery
            matches!(name,
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
            let ts = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_millis();
            if let Some(obj) = e.as_object_mut() { obj.insert("ts_ms".to_string(), serde_json::json!(ts)); }
            items.push(e);
            let _ = std::fs::create_dir_all(path.parent().unwrap_or(std::path::Path::new(".")));
            let _ = std::fs::write(path, serde_json::to_string_pretty(&items).unwrap_or("[]".to_string()));
        }
    }
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
                    if let Ok(map) = serde_json::from_str::<std::collections::HashMap<String, serde_json::Value>>(&s) {
                        let mut reg = MCP_TOOLS.write().unwrap();
                        for (k, v) in map { reg.entry(k).or_insert(v); }
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
        if let Some(res) = result { self.store_value(res, RuntimeValue::Json(tools))?; }
        Ok(())
    }

    pub(crate) fn handle_mcp_start_stdio(
        &mut self,
        _args: &[LexExpression],
        result: Option<&ValueRef>,
    ) -> Result<(), ExecutorError> {
        // Stub: pretend server started
        let info = serde_json::json!({"server": "mcp-stdio", "status": "started"});
        if let Some(res) = result { self.store_value(res, RuntimeValue::Json(info))?; }
        Ok(())
    }

    pub(crate) fn handle_mcp_register_tools(
        &mut self,
        args: &[LexExpression],
        result: Option<&ValueRef>,
    ) -> Result<(), ExecutorError> {
        // mcp.register_tools({ name: {"schema": {...}, "desc": "..." }, ... })
        if args.is_empty() { return Err(ExecutorError::ArgumentError("mcp.register_tools requires a JSON object".to_string())); }
        let obj = match self.evaluate_expression(args[0].clone())? {
            RuntimeValue::Json(serde_json::Value::Object(map)) => map,
            _ => return Err(ExecutorError::ArgumentError("mcp.register_tools requires object".to_string())),
        };
        {
            let mut reg = MCP_TOOLS.write().unwrap();
            for (k, v) in obj { reg.insert(k, v); }
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
        if let Some(res) = result { self.store_value(res, RuntimeValue::Boolean(true))?; }
        Ok(())
    }

    pub fn handle_mcp_run_stdio(
        &mut self,
        _args: &[LexExpression],
        result: Option<&ValueRef>,
    ) -> Result<(), ExecutorError> {
        use std::io::{self, BufRead, Write};
        use std::collections::HashSet;
        use std::sync::{Arc, Mutex};
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
            if let Some(res) = result { self.store_value(res, RuntimeValue::Boolean(true))?; }
            return Ok(());
        }

        // Minimal JSON-RPC 2.0 loop over stdio
        // Accepts lines containing JSON objects: {"jsonrpc":"2.0","id":X,"method":"list_tools"}
        // Responds with {"jsonrpc":"2.0","id":X,"result":...}
        let stdin = io::stdin();
        let cancelled: Arc<Mutex<HashSet<String>>> = Arc::new(Mutex::new(HashSet::new()));
        for line in stdin.lock().lines() {
            let line = match line { Ok(s) => s.trim().to_string(), Err(_) => break };
            if line.is_empty() { continue; }
            if line.eq_ignore_ascii_case("exit") { break; }
            let parsed: Result<serde_json::Value, _> = serde_json::from_str(&line);
            match parsed {
                Ok(val) => {
                    let id = val.get("id").cloned().unwrap_or(serde_json::Value::Null);
                    let method = val.get("method").and_then(|m| m.as_str()).unwrap_or("");
                    match method {
                        "list_tools" => {
                            let tools = serde_json::json!({
                                "jsonrpc":"2.0",
                                "id": id,
                                "result": {
                                    "tools": [
                                        {"name": "emit_file", "description": "Emit binary/text file from model"},
                                        {"name": "read_file", "description": "Read file from disk"},
                                        {"name": "write_file", "description": "Write file to disk"}
                                    ]
                                }
                            });
                            println!("{}", tools);
                        }
                        "tool_info" => {
                            let name = val.get("params").and_then(|p| p.get("name")).and_then(|x| x.as_str()).unwrap_or("");
                            // For stdio, reflect in-memory only
                            let mut meta = serde_json::json!({});
                            if let Ok(reg) = MCP_TOOLS.read() { if let Some(m) = reg.get(name) { meta = m.clone(); } }
                            let out = serde_json::json!({"jsonrpc":"2.0","id": id, "result": {"name": name, "meta": meta}});
                            println!("{}", out);
                        }
                        "tool_call" => {
                            let params = val.get("params").and_then(|p| p.as_object()).cloned().unwrap_or_default();
                            let name = params.get("name").and_then(|x| x.as_str()).unwrap_or("");
                            let profile = Self::mcp_permission_profile();
                            if !Self::mcp_is_tool_allowed(&profile, name) {
                                let err = serde_json::json!({"jsonrpc":"2.0","id": id, "error": {"code": -32010, "message": format!("Tool '{}' not allowed under profile '{}'", name, profile)}});
                                println!("{}", err);
                                continue;
                            }
                            let args = params.get("args").cloned().unwrap_or(serde_json::json!({}));
                            let res = self.execute_tool_call_json(name, &args);
                            match res {
                                Ok(v) => {
                                    self.mcp_append_audit(&serde_json::json!({"tool": name, "ok": true}));
                                    let out = serde_json::json!({"jsonrpc":"2.0","id": id, "result": {"name": name, "output": v}});
                                    println!("{}", out);
                                }
                                Err(e) => {
                                    self.mcp_append_audit(&serde_json::json!({"tool": name, "ok": false, "error": format!("{}", e)}));
                                    let err = serde_json::json!({"jsonrpc":"2.0","id": id, "error": {"code": -32011, "message": format!("{}", e)}});
                                    println!("{}", err);
                                }
                            }
                        }
                        "set_quota" => {
                            let params = val.get("params").and_then(|p| p.as_object()).cloned().unwrap_or_default();
                            let name = params.get("name").and_then(|x| x.as_str()).unwrap_or("");
                            let quota = params.get("allowed_calls").and_then(|x| x.as_u64()).unwrap_or(0);
                            if let Ok(mut reg) = MCP_TOOLS.write() {
                                let entry = reg.entry(name.to_string()).or_insert(serde_json::json!({}));
                                if let Some(obj) = entry.as_object_mut() { obj.insert("allowed_calls".to_string(), serde_json::json!(quota)); }
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
                            if let Some(i) = id.as_i64() { cancelled.lock().unwrap().insert(i.to_string()); }
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
        if let Some(res) = result { self.store_value(res, RuntimeValue::Boolean(true))?; }
        Ok(())
    }

    pub(crate) fn handle_mcp_run_ws(
        &mut self,
        args: &[LexExpression],
        result: Option<&ValueRef>,
    ) -> Result<(), ExecutorError> {
        // mcp.run_ws(url)
        if args.len() < 1 { return Err(ExecutorError::ArgumentError("mcp.run_ws requires url".to_string())); }
        let url = match self.evaluate_expression(args[0].clone())? { RuntimeValue::String(s) => s, v => format!("{:?}", v) };
        // No registry needed for client oneshot
        let r = tokio::task::block_in_place(move || {
            tokio::runtime::Handle::current().block_on(async move {
                use futures_util::{SinkExt, StreamExt};
                use tokio_tungstenite::connect_async;
                let c_to_ms: u64 = std::env::var("LEXON_MCP_WS_CONNECT_TIMEOUT_MS").ok().and_then(|v| v.parse().ok()).unwrap_or(1000);
                match tokio::time::timeout(std::time::Duration::from_millis(c_to_ms), connect_async(url)).await {
                    Ok(Ok((mut ws, _resp))) => {
                        // Send hello (ping) and list_tools if configured
                        let ping = serde_json::json!({"jsonrpc":"2.0","id":1,"method":"ping"}).to_string();
                        let _ = ws.send(tokio_tungstenite::tungstenite::Message::Text(ping)).await;
                        if std::env::var("LEXON_MCP_WS_LIST_TOOLS").ok().as_deref() == Some("1") {
                            let list = serde_json::json!({"jsonrpc":"2.0","id":2,"method":"list_tools"}).to_string();
                            let _ = ws.send(tokio_tungstenite::tungstenite::Message::Text(list)).await;
                        }
                        // Try to read one message (optional)
                        let _ = ws.next().await;
                        let _ = ws.close(None).await;
                        Ok(())
                    }
                    Ok(Err(e)) => Err(ExecutorError::RuntimeError(format!("WS connect error: {}", e))),
                    Err(_) => Err(ExecutorError::RuntimeError("WS connect timeout".to_string())),
                }
            })
        });
        match r { Ok(()) => { if let Some(res) = result { self.store_value(res, RuntimeValue::Boolean(true))?; } Ok(()) }, Err(e) => Err(e) }
    }

    pub fn handle_mcp_run_ws_server(
        &mut self,
        args: &[LexExpression],
        result: Option<&ValueRef>,
    ) -> Result<(), ExecutorError> {
        // mcp.run_ws_server(addr?)
        let addr_str = if !args.is_empty() {
            match self.evaluate_expression(args[0].clone())? { RuntimeValue::String(s) => s, v => format!("{:?}", v) }
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
                use tokio_tungstenite::{accept_async, tungstenite::Message};

                let listener = TcpListener::bind(&addr_str).await.map_err(|e| ExecutorError::RuntimeError(format!("bind {}: {}", addr_str, e)))?;
                println!("🔌 MCP WS server listening on {}", addr_str);
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

                    let mut ws = accept_async(stream).await.map_err(|e| ExecutorError::RuntimeError(format!("handshake error: {}", e)))?;
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
                    let _ = ws.send(Message::Text(banner)).await;

                    while let Some(msg) = ws.next().await {
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
                                                    // Enforce quota using in-process registry (allowed_calls/used_calls)
                                                    let mut denied = false;
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
                                                        // Bridge to executor handlers for tracked tools
                                                        let exec_result = {
                                                            let mut env = super::ExecutionEnvironment::new(config.clone());
                                                            // hint: mirror quotas into env tool_registry if present
                                                            if let Ok(reg) = MCP_TOOLS.read() {
                                                                if let Some(meta) = reg.get(name) {
                                                                    if let Some(limit) = meta.get("allowed_calls").and_then(|v| v.as_u64()) {
                                                                        env.tool_registry.insert(name.to_string(), super::ToolMeta { allowed_calls: Some(limit), used_calls: 0, scope: None });
                                                                    }
                                                                }
                                                            }
                                                            // Optional tool timeout
                                                            let tool_timeout_ms: u64 = std::env::var("LEXON_MCP_TOOL_TIMEOUT_MS").ok().and_then(|v| v.parse().ok()).unwrap_or(2000);
                                                            tokio::runtime::Handle::current().block_on(async move {
                                                                tokio::time::timeout(std::time::Duration::from_millis(tool_timeout_ms), async move { env.execute_tool_call_json(name, &args) }).await
                                                            })
                                                        };
                                                        match exec_result {
                                                            Ok(Ok(val)) => { append_audit(&base_path, &serde_json::json!({"tool": name, "ok": true})); serde_json::json!({"jsonrpc":"2.0","id": id, "result": {"name": name, "output": val}}) },
                                                            Ok(Err(e)) => { append_audit(&base_path, &serde_json::json!({"tool": name, "ok": false, "error": format!("{}", e)})); serde_json::json!({"jsonrpc":"2.0","id": id, "error": {"code": -32003, "message": format!("{}", e)}}) },
                                                            Err(_elapsed) => { append_audit(&base_path, &serde_json::json!({"tool": name, "ok": false, "error": "timeout"})); serde_json::json!({"jsonrpc":"2.0","id": id, "error": {"code": -32004, "message": "tool timeout"}}) },
                                                        }
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
                                        let send_fut = ws.send(Message::Text(resp.to_string()));
                                        let _ = tokio::time::timeout(std::time::Duration::from_millis(msg_timeout_ms), send_fut).await;
                                        if method == "exit" { break; }
                                    }
                                    Err(_) => {
                                        let ack = serde_json::json!({"jsonrpc":"2.0","id": null, "error": {"code": -32700, "message":"Parse error"}});
                                        let _ = ws.send(Message::Text(ack.to_string())).await;
                                    }
                                }
                            }
                            Ok(Message::Close(_)) => break,
                            Ok(_) => continue,
                            Err(_) => break,
                        }
                    }
                    let _ = ws.close(None).await;
                    handled += 1;
                    if oneshot { break; }
                }
                println!("🔌 MCP WS server exiting (handled {} connections)", handled);
                Ok(())
            })
        });
        match r { Ok(()) => { if let Some(res) = result { self.store_value(res, RuntimeValue::Boolean(true))?; } Ok(()) }, Err(e) => Err(e) }
    }
}


