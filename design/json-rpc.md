# Cube JSON-RPC implementation discussion with Claude

## Nico

How does one implement a JSON-RPC server?
Do I still need to implement one REST endpoint (POST) for all request to go in? something like this:
```
All requests go to a single
``POST /`` endpoint; the method name is in the JSON body.
```
I though JSON-RPC avoided REST endpoints alltogether...?

## Claude

Great question, Nicolas. JSON-RPC and REST are different paradigms, but JSON-RPC still needs a transport layer — it doesn't magically avoid HTTP. Here's the distinction:
REST maps operations to HTTP verbs and URL paths (GET /users/42, DELETE /tasks/7). The routing logic is in the URL structure.
JSON-RPC collapses all of that into a single endpoint. The method name and parameters live inside the JSON body, not in the URL. So yes, when running JSON-RPC over HTTP, you typically expose one POST / (or POST /rpc, or whatever you like) and dispatch internally based on the method field in the payload.
So your instinct is right — you do still have one HTTP endpoint, but it's not really "REST" in the architectural sense. It's just using HTTP as a dumb transport pipe. The request body does all the talking:
```
json{
  "jsonrpc": "2.0",
  "method": "benchmark.list",
  "params": {},
  "id": 1
}
```
That said, HTTP isn't the only option. JSON-RPC is transport-agnostic by design — it works equally well over WebSockets, raw TCP, stdin/stdout (which is actually how MCP typically uses it via the stdio transport), or even Unix domain sockets. In those cases there's no "endpoint" at all — you're just reading/writing JSON-RPC frames on a bidirectional stream.
For CUBE's MCP context specifically, you're most likely dealing with either stdio (for local tool servers) or SSE/streamable HTTP (for remote ones), where the JSON-RPC framing is handled by the MCP SDK rather than you manually wiring up a Flask/FastAPI route. The SDK gives you a server object, you register your method handlers, and the transport plumbing is abstracted away.
So to summarize: JSON-RPC doesn't avoid endpoints — it avoids multiple endpoints and URL-based routing. One pipe in, dispatch by method field internally.

## Nico

I see... what are the different pros & cons for the transport of JSON-RPC?
In the case of cube, we want to have remote cube benchmarks for the following reasons:

1. Cross-language benchmarks (original motivation) A benchmark written in Java, Go, Rust, or Node.js can implement the CUBE protocol. The harness speaks JSON-RPC to it the same way it talks to a Python cube — language doesn't matter.

2. Process isolation / fault tolerance A crashing, OOM-ing, or deadlocked task doesn't kill the harness. The harness just closes the socket. Especially important for code-execution or browser benchmarks where a step can hang.

3. Scaling across machines The design doc already shows Ray workers receiving TaskConfig and calling task_config.make(). A remote cube server generalizes this: a Ray worker (or Kubernetes pod) runs the task server; the harness is elsewhere. Each task server can be co-located with the resource it needs (GPU, browser, specific OS).

4. Security sandbox For benchmarks running untrusted agent-generated code, the task runs inside a container/VM that is network-isolated — only the CUBE protocol port is exposed.

5. Shared expensive infrastructure WebArena-style: one VM runs all services; many tasks connect to the same VM. The benchmark server manages the VM, spawns per-task servers that point at it. No need to copy VM state per task.

## Claude

This is a really well-motivated set of requirements. Let me break down the transport options through the lens of what CUBE actually needs.

**stdio** is the simplest — parent process spawns a child, they talk over stdin/stdout. Zero network setup, no ports, no auth. It's what most MCP tool servers use today. But it's fundamentally local and single-machine. It can't cross language boundaries cleanly (you'd need the harness to shell out to a Java/Go binary), and it ties the child's lifecycle to the parent. For your requirements 1, 3, 4, and 5, it's essentially a non-starter. It could work for "local Python cube, same machine, simple case" but that's the easy path you've already solved.

**HTTP (single POST endpoint)** is the most universally supported option. Every language has an HTTP server library. It works across machines, through firewalls, behind load balancers, inside Kubernetes — all the infra you mentioned in point 3 just works. It's request-response though, so the harness polls or the server can't push state changes. For CUBE's model where the harness calls `task.setup()`, `task.score()`, etc., that request-response pattern actually fits well — the harness is always the initiator. The downside is overhead per call (HTTP headers, connection setup unless you use keep-alive or HTTP/2), and no server-initiated messages without bolting on something extra.

**SSE (Server-Sent Events over HTTP)** is what the current MCP spec uses for its "HTTP with streaming" transport. The client opens a regular HTTP connection for an event stream, and sends requests via POST. This gives you server→client push (useful if a task wants to stream progress or logs), while keeping HTTP's universality. The catch is it's unidirectional streaming — only server-to-client. If you need bidirectional streaming (say, an interactive benchmark where the agent and environment exchange messages rapidly), SSE gets awkward.

**WebSockets** give you a persistent, full-duplex channel. Low overhead after the handshake, bidirectional, great for high-frequency back-and-forth. This maps well to interactive benchmarks (point 5 — WebArena-style where a task might need to push environment state changes). The downsides: WebSocket support is spottier in some languages' standard libraries, it doesn't play as nicely with HTTP-oriented infra (some proxies, load balancers need special config), and connection management gets more complex (reconnection, heartbeats, etc.).

**Raw TCP** is the lowest overhead but you'd be reimplementing framing, connection management, and everything HTTP gives you for free. Hard to justify unless you have extreme latency requirements, which benchmarks typically don't.

Given your five requirements, I'd argue **HTTP (single POST)** is the pragmatic default for CUBE, for a few reasons. Cross-language support (requirement 1) is maximized — every language has rock-solid HTTP libraries. Kubernetes/Ray scaling (requirement 3) is built around HTTP. Security sandboxing (requirement 4) is easiest when you only expose one port serving a well-understood protocol. And for the interaction pattern CUBE uses — harness-initiated RPC calls to a task server — request-response is the natural fit.

You could layer SSE on top for optional streaming (task logs, progress), and the MCP spec already defines this pattern so you'd be aligned with the ecosystem. WebSockets would be worth considering only if you identify benchmarks that genuinely need rapid bidirectional exchange, and even then you could offer it as an alternative transport for those specific cases rather than the default.

One design consideration worth flagging: whatever transport you pick, keep the JSON-RPC message layer identical. The whole point is that a CUBE harness shouldn't care whether it's talking to a local stdio process or a remote HTTP server — it sends the same `{"method": "task.setup", ...}` either way, and the transport is a pluggable concern underneath. That way you can start with HTTP, add WebSocket later for specific use cases, and nothing above the transport layer changes.

## Nico

I see. We also want to support multi-agent interaction with cube benchmakrs, async action executions, and streaming observations (such as sound or video). Adding this consideration on top of everything else, what do you suggest we do?

## Claude

These additions shift the picture significantly. Let me think through what each one demands from the transport

**Multi-agent interaction** means multiple clients (agents) are connected to the same benchmark environment simultaneously. The server needs to distinguish between agents, route messages to the right one, and potentially broadcast environment state changes to all of them. With plain HTTP POST, each request is stateless and independent — you'd need to bolt on session management, and there's no way for the server to initiate a message to a specific agent. You end up fighting the protocol.

**Async action execution** means an agent sends an action and the result isn't immediately available — maybe the environment needs to simulate for a while, or it's waiting on another agent's turn, or the action is inherently long-running. With request-response HTTP, the client is blocked waiting. You could return a task ID and have the client poll, but that's clunky and adds latency. What you really want is: "here's my action" → "acknowledged" → ... time passes ... → "here's the result, pushed to you."

**Streaming observations** (audio, video, sensor data) is the strongest constraint. These are continuous, high-bandwidth, server-to-client data flows. SSE can technically handle this as a stream of events, but it's unidirectional, text-based (you'd be base64-encoding binary data with ~33% overhead), and you can't multiplex multiple streams cleanly. HTTP wasn't designed for this.

Taken together, you're describing a system that needs bidirectional communication, server-initiated messages, persistent sessions, and efficient binary streaming. That points clearly toward **WebSockets as the primary transport**, with a few architectural nuances.

Here's what I'd suggest for CUBE:

**WebSocket as the default remote transport**. A CUBE task server opens a WebSocket endpoint. The harness (or agent) connects, and the JSON-RPC messages flow bidirectionally over that connection. The connection itself is the session — the server knows which agent is which by its socket. Multi-agent is natural: N agents open N connections to the same server. Server-initiated pushes (async results, turn notifications, environment state changes) just flow back on the socket. This covers requirements 1 through 5 from before, plus all three new ones.

**For streaming observations, use a sideband approach**. Don't try to push raw video frames through the JSON-RPC message channel — that would choke the control flow. Instead, have the JSON-RPC layer negotiate a media stream. Something like: the server sends a JSON-RPC notification `{"method": "observation.stream_available", "params": {"type": "video", "url": "ws://same-host:PORT/stream/video"}}` and the agent opens a separate connection for the raw binary stream. Or use WebSocket binary frames on a dedicated channel. This keeps the control plane (JSON-RPC) and the data plane (media) cleanly separated, which matters for backpressure, prioritization, and the fact that different consumers might want different streams.

**Keep HTTP POST as a fallback transport for simple benchmarks** that don't need any of the advanced features — text-only, single-agent, synchronous. Not every benchmark will be a multi-agent video-streaming environment. A simple QA benchmark shouldn't require WebSocket infrastructure. The JSON-RPC messages are identical either way; the transport is the only thing that changes.

**For the protocol design**, I'd suggest extending CUBE's JSON-RPC vocabulary with a few patterns:

The server can push via JSON-RPC notifications (no `id` field, no response expected) — these are already part of the JSON-RPC 2.0 spec and they're perfect for async results, turn signals, and observation events. An async action flow would look like: agent sends `{"method": "task.act", "params": {"action": "move_north"}, "id": 42}`, server immediately replies `{"id": 42, "result": {"status": "accepted"}}`, and later pushes `{"method": "task.action_result", "params": {"action_id": 42, "observation": "..."}}` as a notification.

For multi-agent, the connection identity handles routing, but you might want an explicit `agent_id` in the protocol so that observers or spectators can connect too without being treated as players.

**In terms of implementation burden across languages** (your requirement 1), WebSocket libraries are mature in Java (Jetty, Tyrus), Go (gorilla/websocket, nhooyr), Rust (tokio-tungstenite), Node (ws), and Python (websockets, or the MCP SDK's own transport). It's not as universal as HTTP, but it's close enough for the languages you'd realistically see CUBE benchmarks written in.

So the short version: **WebSocket for the control plane, with optional sideband channels for heavy media streaming, and HTTP POST preserved as a simple fallback**. The JSON-RPC message schema stays transport-agnostic, so benchmark authors don't think about transports at all — they implement method handlers, and the CUBE SDK handles the rest.
