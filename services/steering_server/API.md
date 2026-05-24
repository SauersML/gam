# API

The OpenAPI schema is available from a running server at `/openapi.json`.

Generate a checked-in snapshot:

```bash
python scripts/export_openapi.py > API.openapi.json
```

## Route Signatures

```text
POST /v1/auth/token
  body: { "api_key": string }
  response: { "access_token": string, "token_type": "bearer" }

POST /v1/steerers
  auth: x-api-key or Bearer token
  body: SteererCreateRequest(model, layer, concept_targets[])
  response: SteererResponse

GET /v1/steerers
  auth: x-api-key or Bearer token
  response: SteererListResponse

POST /v1/steerers/{steerer_id}/steer
  auth: x-api-key or Bearer token
  body: SteerRequest(concept_targets[], alpha, prompts[], max_tokens)
  response: SteerResponse

GET /v1/steerers/{steerer_id}/diagnostics
  auth: x-api-key or Bearer token
  response: DiagnosticsResponse

WebSocket /v1/stream?token=<api-key>
  first client message: SteerRequest plus steerer_id
  server messages: completion_start, token, completion_end
```
