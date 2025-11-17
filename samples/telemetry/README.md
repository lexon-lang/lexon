Lexon Telemetry: Starter Dashboard

This repo ships an optional Grafana dashboard stub to visualize Lexon runtime metrics.

How to use
- Run the local OTEL collector (`otel/otel-collector-config.yaml`) and point it to Prometheus/Grafana.
- Import the dashboard JSON: `otel/grafana/lexon-dashboard.json` into Grafana.
- Adjust data source and metric names to your setup (the stub expects Prometheus metrics like `lexon_llm_calls_total`).

Notes
- The dashboard is a starter template: latency, cost, calls by model, provider health, tool calls, and error rates.
- In CI and smokes we default to simulated providers for determinism.





