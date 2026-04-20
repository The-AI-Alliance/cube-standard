# Deltas — JSON-RPC Streaming

**Targets:** `openspec/specs/server/spec.md`

Phase 1 (HTTP POST transport) is already applied to the current spec. Remaining phases
extend the server with streaming and sideband support.

## MODIFIED — existing transport clarified
**Spec:** server/spec.md

Phase 1 already delivered — current spec documents it. No delta here beyond marking
where streaming and sideband slot in.

## ADDED — WebSocket transport (phase 2)
**Spec:** server

- `GET /ws` upgrade endpoint for the same JSON-RPC 2.0 methods
- Server can push `tools/call_stream` partial results with the same `action_id`
- Client opens one WS per Task; multiplex methods over it

Backwards-compatible with HTTP POST.

## ADDED — Media sideband (phase 3)
**Spec:** server

Large binary payloads (images, video, audio) travel out-of-band to avoid bloating
JSON messages.

- Server exposes `GET /media/{content_id}` for blob retrieval
- Serialized `ImageContent` / `AudioContent` / `VideoContent` can reference a `content_id` instead of inlining base64
- Client resolves `content_id` via the sideband endpoint

Backwards-compatible: inlined base64 still works.

---

See `proposal.md` for the original plan and per-phase status.
