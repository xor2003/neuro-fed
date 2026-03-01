# Brain Sharing (Phase 2, Task 2)

Brain sharing enables NeuroFed nodes to exchange their PC‑brain weights via the Nostr network using the Blossom protocol (NIP‑94). This allows users to transfer their trained predictive‑coding brains to other nodes, making the project viral and fostering a decentralized AGI ecosystem.

## Overview

A **brain** is a set of weights from a Predictive Coding hierarchy, serialized as a [safetensors](https://github.com/huggingface/safetensors) file. Each brain is identified by a SHA‑256 hash of its weights and includes metadata such as base model ID, version, description, and license.

Brain sharing involves three components:

1. **Blossom client** – uploads/downloads brain files to/from Nostr‑compatible storage.
2. **NIP‑94 support** – publishes and subscribes to brain‑share events on Nostr relays.
3. **Brain manager** – manages local brain storage, compatibility checking, and integration with the PC hierarchy.

## Configuration

Brain sharing is configured in `config.toml` under the `brain_sharing_config` section:

```toml
[brain_sharing_config]
enabled = false
relay_urls = ["wss://relay.damus.io"]
brain_storage_dir = "./brains"
cache_dir = "./cache/brains"
base_model_id = "unknown"
allow_untrusted_authors = false
max_brain_size = 2147483648  # 2 GB
```

- `enabled` – turn brain sharing on/off.
- `relay_urls` – Nostr relays used for brain‑share events (overrides the general Nostr config).
- `brain_storage_dir` – local directory where brains are stored permanently.
- `cache_dir` – directory for caching downloaded brains.
- `base_model_id` – identifier of the base LLM this node uses (e.g., `"llama-3-8b"`). Used for compatibility checking.
- `allow_untrusted_authors` – whether to accept brains from authors not in a trust list.
- `max_brain_size` – maximum brain file size in bytes.

## Usage

### Saving a Brain

```rust
use neuro_fed_node::brain_manager::BrainManager;

let brain_manager = ...;
let weights = ...; // HashMap<String, Vec<f32>>
let (brain_id, path) = brain_manager.save_brain(
    weights,
    "My fine‑tuned brain",
    vec!["fine‑tuned".to_string(), "science".to_string()]
).await?;
```

This serializes the weights to a safetensors file, computes its SHA‑256 hash, stores it in `brain_storage_dir`, and creates a metadata JSON sidecar.

### Sharing a Brain

```rust
brain_manager.share_brain(&brain_id).await?;
```

This uploads the brain file to a Blossom‑compatible storage server (via the `BlossomClient`) and publishes a NIP‑94 event on Nostr relays. The event contains the brain metadata and a download URL.

### Importing a Brain

```rust
let record = brain_manager.import_brain(
    "abc123...",
    Some("llama-3-8b")
).await?;
```

The manager:
1. Queries Nostr relays for a NIP‑94 event with the given brain ID.
2. Downloads the brain file from the URL referenced in the event.
3. Verifies the SHA‑256 hash.
4. Moves the file to `brain_storage_dir` and adds a `BrainRecord`.
5. Checks base‑model compatibility (fails if `base_model_id` mismatch).

### Loading a Brain into the PC Hierarchy

```rust
let safetensors = brain_manager.load_brain(&brain_id).await?;
// Integrate weights into PredictiveCoding hierarchy
```

The `load_brain` method returns a `safetensors::SafeTensors` object that can be used to reconstruct the PC‑hierarchy weights.

## NIP‑94 Event Format

Brain‑share events are Nostr events of kind `1064` (NIP‑94 – File Metadata). The event’s `content` is a JSON‑serialized `BrainMetadata` object. The event tags include:

- `["brain", "<brain_id>"]`
- `["url", "<download_url>"]`
- `["base_model", "<base_model_id>"]`
- `["license", "<license>"]` (optional)
- `["author", "<nostr_pubkey>"]`

Example event:

```json
{
  "kind": 1064,
  "content": "{\"brain_id\":\"abc123...\",\"base_model_id\":\"llama-3-8b\",...}",
  "tags": [
    ["brain", "abc123..."],
    ["url", "https://example.com/brains/abc123....safetensors"],
    ["base_model", "llama-3-8b"],
    ["license", "MIT"],
    ["author", "npub1..."]
  ]
}
```

## Blossom Client

The `BlossomClient` handles the actual file transfer. It supports:

- **Upload** – sends a brain file to a Blossom‑compatible HTTP server (NIP‑96) and returns a public URL.
- **Download** – fetches a brain file from a given URL and caches it locally.
- **Hash verification** – ensures file integrity via SHA‑256.

Currently, the client uses a dummy HTTP server for demonstration; a production implementation should integrate with a real Blossom server (e.g., [nostr‑blossom](https://github.com/nostr‑protocol/blossom)).

## Compatibility Checking

Each brain is tagged with a `base_model_id`. A node can only load brains that match its own `base_model_id` (unless `allow_untrusted_authors` is true). This prevents incompatible weight shapes from crashing the PC hierarchy.

## Security Considerations

- **Hash verification** – every downloaded brain is validated against its advertised SHA‑256 hash.
- **Size limits** – `max_brain_size` prevents memory‑exhaustion attacks.
- **Author trust** – optional whitelist of trusted Nostr pubkeys.
- **Network isolation** – brain sharing can be completely disabled by setting `enabled = false`.

## Testing

Run the brain‑sharing unit tests with:

```bash
cargo test blossom_client
cargo test brain_manager
cargo test nostr_federation::nip94
```

Integration tests simulate uploading and downloading brains using a temporary local server.

## Future Work

- **NIP‑96 integration** – support for paying for storage with Bitcoin/Lightning.
- **Peer‑to‑peer transfer** – use libp2p or WebRTC for direct node‑to‑node brain transfers.
- **Incremental updates** – share only weight deltas to reduce bandwidth.
- **Reputation system** – track brain quality via Nostr zaps and user ratings.

## References

- [NIP‑94: File Metadata](https://github.com/nostr-protocol/nips/blob/master/94.md)
- [NIP‑96: HTTP File Storage](https://github.com/nostr-protocol/nips/blob/master/96.md)
- [safetensors](https://github.com/huggingface/safetensors)
- [Blossom protocol](https://github.com/nostr-protocol/blossom)