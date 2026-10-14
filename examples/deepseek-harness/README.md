# DeepSeek Harness integration example (experimental)

This is an **opt-in** configuration for AI Control Centre. It does not install
DeepSeek Harness or change the default Agent/Computer DMZ selection. Tested
separately on Orson with DeepSeek Harness `0.2.0-rc.2`, a local llama.cpp API,
and a simple Python file creation task; the **Control Centre-managed lifecycle
has not yet been tested on Orson**.

## Prepare Orson

1. In the independently running DeepSeek Harness, retain the working
   `local-llama` provider at `http://127.0.0.1:8080/v1`. Its model selection
   is owned by DSH, not automatically synchronised to Control Centre.
2. Stop the manually launched DSH instance (Ctrl+C) before testing the managed
   service. Control Centre must not claim or stop an external instance.
3. Confirm Node executable location using `command -v node` and
   `command -v npx`. The example uses Orson's pi-node installation; edit the
   executable and PATH settings on other Linux systems.
4. Confirm the pinned version is available to npx without an install prompt:
   `npx --no-install @deepseek-ai/dsh@0.2.0-rc.2 --version`.
   If unavailable, install it deliberately from a terminal first. Never let
   a background service depend on an interactive npm install prompt.
5. Ensure the example working directory exists:
   `mkdir -p ~/Documents/deepseek-harness/default-workspace`.
6. Back up your user TOML files. Append the provided service and harness
   sections to the corresponding files in `~/.config/ai-control-centre/`.
   Do not replace existing sections.
7. In **Preferences > Settings**, choose **DeepSeek Harness** as the Agent
   harness and save. Start the Agent task with a reviewed local model.

## Readiness and browser authentication

The current ProcessService controller requires a successful HTTP health check
to report **Ready**. DeepSeek Harness protects its Web UI with a session token,
so an unauthenticated request to `/` might not be a valid readiness signal.
Before enabling the commented health section in `services.toml`, test:

```bash
curl -sS -o /dev/null -w '%{http_code}\\n' http://127.0.0.1:3080/
```

If it returns anything other than 200, leave the health section disabled and
expect **Running, not ready** instead of **Ready**. A robust token-aware
readiness/open implementation is future work. The normal `open_url` points to
the loopback address, but **does not contain the launch token**. Inspect the
launcher-owned DSH log for the token-bearing URL on first launch; keep it
private and do not paste it into published screenshots, GitHub, or diagnostics.

When DSH is configured with the local llama.cpp server, no DeepSeek cloud API
key should be needed. Retain DSH's existing private credentials and settings;
Control Centre does not overwrite them.

## Known limitations

- The first version's service discovery is configuration-based, not automatic
  discovery of an arbitrary npx installation.
- The launcher currently needs a valid DSH readiness endpoint or it will show
  the service as **Running, not ready**; readiness is not inferred solely from a
  TCP listener.
- The DSH model selector is independent of Control Centre's model selector.
  If you change the running llama.cpp model, confirm its ID in DSH.
- DSH has shell/file tools and is **not** a sandbox. Use a dedicated workspace
  and do not grant it unrestricted access to sensitive files.
- The installed npm package is a developer preview. Keep the version pinned
  until a newer version is tested.

Upstream: https://github.com/deepseek-ai/deepseek-harness
