# Deployment

## Docker

> [!TIP]
> The `-v ~/.vingobot:/home/vingobot/.vingobot` flag mounts your local config directory into the container, so your config and workspace persist across container restarts.
> The container runs as the non-root user `vingobot` (UID 1000) and reads config from `/home/vingobot/.vingobot`. Always mount your host config directory to `/home/vingobot/.vingobot`, not `/root/.vingobot`.
> If you get **Permission denied**, fix ownership on the host first: `sudo chown -R 1000:1000 ~/.vingobot`, or pass `--user $(id -u):$(id -g)` to match your host UID. Podman users can use `--userns=keep-id` instead.
>
> [!IMPORTANT]
> Official Docker usage currently means building from this repository with the included `Dockerfile`. Docker Hub images under third-party namespaces are not maintained or verified by this project; do not mount API keys or bot tokens into them unless you trust the publisher.

### Docker Compose

```bash
docker compose run --rm vingobot-cli onboard   # first-time setup
vim ~/.vingobot/config.json                     # add API keys
docker compose up -d vingobot-gateway           # start gateway
```

```bash
docker compose run --rm vingobot-cli agent -m "Hello!"   # run CLI
docker compose logs -f vingobot-gateway                   # view logs
docker compose down                                      # stop
```

### Docker

```bash
# Build the image
docker build -t vingobot .

# Initialize config (first time only)
docker run -v ~/.vingobot:/home/vingobot/.vingobot --rm vingobot onboard

# Edit config on host to add API keys
vim ~/.vingobot/config.json

# Run gateway (connects to enabled channels, e.g. Telegram/Discord/Mochat)
docker run -v ~/.vingobot:/home/vingobot/.vingobot -p 18790:18790 vingobot gateway

# Or run a single command
docker run -v ~/.vingobot:/home/vingobot/.vingobot --rm vingobot agent -m "Hello!"
docker run -v ~/.vingobot:/home/vingobot/.vingobot --rm vingobot status
```

## Linux Service

Run the gateway as a systemd user service so it starts automatically and restarts on failure.

**1. Find the vingobot binary path:**

```bash
which vingobot   # e.g. /home/user/.local/bin/vingobot
```

**2. Create the service file** at `~/.config/systemd/user/vingobot-gateway.service` (replace `ExecStart` path if needed):

```ini
[Unit]
Description=vingobot Gateway
After=network.target

[Service]
Type=simple
ExecStart=%h/.local/bin/vingobot gateway
Restart=always
RestartSec=10
NoNewPrivileges=yes
ProtectSystem=strict
ReadWritePaths=%h

[Install]
WantedBy=default.target
```

**3. Enable and start:**

```bash
systemctl --user daemon-reload
systemctl --user enable --now vingobot-gateway
```

**Common operations:**

```bash
systemctl --user status vingobot-gateway        # check status
systemctl --user restart vingobot-gateway       # restart after config changes
journalctl --user -u vingobot-gateway -f        # follow logs
```

If you edit the `.service` file itself, run `systemctl --user daemon-reload` before restarting.

> **Note:** User services only run while you are logged in. To keep the gateway running after logout, enable lingering:
>
> ```bash
> loginctl enable-linger $USER
> ```

## macOS LaunchAgent

Use a LaunchAgent when you want `vingobot gateway` to stay online after you log in, without keeping a terminal open.

**1. Get the absolute `vingobot` path:**

```bash
which vingobot   # e.g. /Users/youruser/.local/bin/vingobot
```

Use that exact path in the plist. It keeps the Python environment from your install method.

**2. Create `~/Library/LaunchAgents/ai.vingobot.gateway.plist`:**

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>ai.vingobot.gateway</string>

  <key>ProgramArguments</key>
  <array>
    <string>/Users/youruser/.local/bin/vingobot</string>
    <string>gateway</string>
    <string>--workspace</string>
    <string>/Users/youruser/.vingobot/workspace</string>
  </array>

  <key>WorkingDirectory</key>
  <string>/Users/youruser/.vingobot/workspace</string>

  <key>RunAtLoad</key>
  <true/>

  <key>KeepAlive</key>
  <dict>
    <key>SuccessfulExit</key>
    <false/>
  </dict>

  <key>StandardOutPath</key>
  <string>/Users/youruser/.vingobot/logs/gateway.log</string>

  <key>StandardErrorPath</key>
  <string>/Users/youruser/.vingobot/logs/gateway.error.log</string>
</dict>
</plist>
```

**3. Load and start it:**

```bash
mkdir -p ~/Library/LaunchAgents ~/.vingobot/logs
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/ai.vingobot.gateway.plist
launchctl enable gui/$(id -u)/ai.vingobot.gateway
launchctl kickstart -k gui/$(id -u)/ai.vingobot.gateway
```

**Common operations:**

```bash
launchctl list | grep ai.vingobot.gateway
launchctl kickstart -k gui/$(id -u)/ai.vingobot.gateway   # restart
launchctl bootout gui/$(id -u) ~/Library/LaunchAgents/ai.vingobot.gateway.plist
```

After editing the plist, run `launchctl bootout ...` and `launchctl bootstrap ...` again.

> **Note:** if startup fails with "address already in use", stop the manually started `vingobot gateway` process first.
