use futures_util::{SinkExt, StreamExt};
use tokio_tungstenite::tungstenite::Message;

#[tokio::main(flavor = "current_thread")]
async fn main() {
    let url = std::env::var("WS_URL").unwrap_or_else(|_| "ws://127.0.0.1:8090".to_string());
    let (mut ws, _resp) = match tokio_tungstenite::connect_async(&url).await {
        Ok(ok) => ok,
        Err(e) => {
            eprintln!("connect error: {}", e);
            std::process::exit(2);
        }
    };
    // Allow the server to send its ready banner
    let _ = tokio::time::timeout(std::time::Duration::from_millis(300), ws.next()).await;

    // Send list_tools and a tool_call to exercise streaming progress/heartbeat
    let list = r#"{"jsonrpc":"2.0","id":1,"method":"list_tools"}"#.to_string();
    let _ = ws.send(Message::Text(list)).await;

    let tool_call = serde_json::json!({
        "jsonrpc":"2.0","id":2,"method":"tool_call",
        "params":{"name":"web.search","args":{"query":"lexon language","n":2}}
    })
    .to_string();
    let _ = ws.send(Message::Text(tool_call)).await;

    // Read a few messages to observe heartbeat/progress and final result
    let mut received = 0usize;
    loop {
        match tokio::time::timeout(std::time::Duration::from_secs(3), ws.next()).await {
            Ok(Some(Ok(Message::Text(txt)))) => {
                println!("{}", txt);
                received = received.saturating_add(1);
                if received >= 8 {
                    break;
                }
            }
            Ok(Some(Ok(Message::Close(_)))) => break,
            Ok(Some(Ok(_))) => continue,
            Ok(Some(Err(_e))) => break,
            Ok(None) => break,
            Err(_elapsed) => break,
        }
    }
    let _ = ws.close(None).await;
}
