# DeepSeek Harness integration (experimental)

This opt-in integration preserves Computer DMZ as the existing default. Orson
successfully tested DeepSeek Harness 0.2.0-rc.2 with a local llama.cpp provider
and verified a Python file on disk. **Control Centre-managed operation is not
validated yet.** Use the matching source branch, not an older installed release.

## Install pinned DeepSeek Harness on Orson

Stop the manually launched DeepSeek instance with Ctrl+C before the managed test.
Node 22 is already installed in Orson's user-local pi-node directory.

```bash
mkdir -p ~/ai/apps/deepseek-harness
npm install --prefix ~/ai/apps/deepseek-harness --save-exact @deepseek-ai/dsh@0.2.0-rc.2
~/ai/apps/deepseek-harness/node_modules/.bin/dsh --version
```

This is a deliberate, persistent local package installation. The earlier
`npx --no-install` failed because npx did not resolve the one-off cached
package as a locally installed dependency. Control Centre must not fetch npm
packages or prompt for consent during startup.

## Configure Control Centre

1. Retain your working `local-llama` provider in DeepSeek Harness, pointing at
   `http://127.0.0.1:8080/v1`. Keep its credentials private.
2. Check that `~/Documents/deepseek-harness/default-workspace` exists, or
   configure a different dedicated workspace.
3. Back up `~/.config/ai-control-centre/services.toml` and `profiles.toml`.
4. Append the corresponding sections in the example TOML files. Do **not**
   overwrite the existing services, profiles or the Computer DMZ harness.
5. Start Control Centre from the **feature/deepseek-harness** branch after
   running its automated tests. The installed older release will not recognise
   expected HTTP 401 as ready.
6. Under Preferences > Settings, select **DeepSeek Harness** for the Agent task.
   Start with a reviewed local model and confirm the model ID matches in DSH.

## Health and authentication

Your verified `GET http://127.0.0.1:3080/` returned HTTP **401**. This is a
protected UI's challenge, not an unhealthy server. In this branch, the generic
HTTP health checker accepts an HTTP error response **only when it equals the
explicitly configured expected_status**, and only when there is no required
body substring. Existing services expecting 200 continue to reject 401.

HTTP 401 confirms an HTTP responder is present; it does not independently
prove that the model provider, browser session or agent tools work. Another
process could occupy the same port. Control Centre's process ownership rules
must remain in force.

The web interface issues an authentication token in its startup URL.
`open_url = http://127.0.0.1:3080` does **not** automatically include that
token. After starting it through Control Centre, check the service log for
its printed token-bearing URL and open that locally. Do not publish the URL,
copy it into repository files or make it part of diagnostic exports. A proper
secure 'Open' action needs separate implementation and validation.

## Known limitations

- The service configuration references Orson's Node installation path; this
  is an example, not a portable default.
- A DSH process launched outside Control Centre remains externally owned.
  Do not automatically stop it as if the launcher owned it.
- Changing the selected model in Control Centre does not automatically change
  the model selected in the DeepSeek web application.
- DeepSeek Harness has powerful shell and filesystem tools, and is not a
  sandbox. Use a dedicated non-sensitive workspace.
- DeepSeek is a developer-preview package: keep this tested version pinned.

Upstream: https://github.com/deepseek-ai/deepseek-harness
