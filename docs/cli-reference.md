# CLI Reference

| Command | Description |
|---------|-------------|
| `vingobot onboard` | Initialize config & workspace at `~/.vingobot/` |
| `vingobot onboard --wizard` | Launch the interactive onboarding wizard |
| `vingobot onboard -c <config> -w <workspace>` | Initialize or refresh a specific instance config and workspace |
| `vingobot agent -m "..."` | Chat with the agent |
| `vingobot agent -w <workspace>` | Chat against a specific workspace |
| `vingobot agent -w <workspace> -c <config>` | Chat against a specific workspace/config |
| `vingobot agent` | Interactive chat mode |
| `vingobot agent --no-markdown` | Show plain-text replies |
| `vingobot agent --logs` | Show runtime logs during chat |
| `vingobot serve` | Start the OpenAI-compatible API |
| `vingobot gateway` | Start the gateway |
| `vingobot status` | Show status |
| `vingobot provider login openai-codex` | OAuth login for providers |
| `vingobot channels login <channel>` | Authenticate a channel interactively |
| `vingobot channels status` | Show channel status |

Interactive mode exits: `exit`, `quit`, `/exit`, `/quit`, `:q`, or `Ctrl+D`.
