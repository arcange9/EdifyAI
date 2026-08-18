# Edify AI

*Turn Knowledge Into Understanding.*

Edify AI is a privacy-first AI-powered study workspace for Windows. Import documents, URLs, and YouTube videos — then generate study notes, flashcards, quizzes, and chat with your materials using any AI provider you choose.

## Features

1. Document ingestion — PDF, DOCX, TXT, Markdown, URLs, YouTube transcripts
2. Multi-provider AI — OpenRouter, Google AI Studio, Groq, and custom OpenAI-compatible endpoints
3. AI chat with RAG — ask questions about your materials with source citations
4. Study notes generation — structured Markdown notes with key takeaways
5. Flashcards — AI-generated with topic tagging and spaced repetition
6. Quizzes — multiple choice, true/false, short answer with explanations
7. Study projects — organize materials by subject
8. Library — search across all documents, notes, flashcards, and quizzes
9. Dark/light mode — system, light, or dark theme
10. Offline-friendly — browse and manage projects without a provider

## Architecture

```
src/
  ai/providers/     Provider abstraction + adapters (OpenRouter, Google, Groq, Custom)
  lib/              Database, credentials, ingestion, RAG, app context
  components/       Shared UI components
  pages/            Route-level pages (Dashboard, Settings, ProjectView, etc.)
  styles/           Theme and global CSS
electron/
  main.mjs          Secure Electron main process (contextIsolation, no nodeIntegration)
  preload.mjs       Minimal IPC bridge
tests/              Unit tests for ingestion, RAG, and provider logic
```

### Provider System

Edify AI uses a unified `AIProvider` interface. All providers implement:

- `chat()` and `streamChat()` — streaming text generation
- `generateStructured()` — JSON-schema-constrained output
- `summarize()`, `generateNotes()`, `generateFlashcards()`, `generateQuiz()`
- `explainConcept()` — AI tutor mode
- `healthCheck()` — connection testing
- `listModels()` — dynamic model discovery

The rest of the app never depends on provider-specific code. Adding a new provider means implementing one interface.

### Security

- API keys stored via Electron `safeStorage` (OS keychain encryption)
- `contextIsolation: true`, `nodeIntegration: false`, `sandbox: true`
- No API keys in source code, environment variables, or frontend bundles
- All data stored locally in IndexedDB

## Requirements

- Windows 10 x64 or Windows 11 x64
- Node.js 20.19+ (for development)
- An API key for at least one AI provider

## Installation

### Download (recommended)

Download the latest `Edify-AI-Setup` installer from the [Releases page](https://github.com/arcange9/EdifyAI/releases).

### From source

```bash
git clone https://github.com/arcange9/EdifyAI.git
cd EdifyAI
npm install
npm run dev
```

To build the Windows installer:

```bash
npm run dist:win
```

The installer will be in `release/Edify-AI-Setup-x64.exe`.

## Provider Configuration

1. Open Edify AI — the first-run wizard will guide you
2. Choose a provider: OpenRouter, Google AI Studio, Groq, or Custom
3. Enter your API key (stored securely on your device)
4. Click "Test Connection"
5. Select a model
6. Start studying

You can add or change providers anytime via Settings → AI Providers.

### Getting API keys

- OpenRouter: https://openrouter.ai/keys
- Google AI Studio: https://aistudio.google.com/apikey
- Groq: https://console.groq.com/keys
- Custom: any OpenAI-compatible API endpoint

## Privacy

- Your API keys never leave your device
- Edify AI does not send your data to Edify servers
- Learning materials are stored locally in IndexedDB
- Content is only sent to your chosen AI provider during generation
- "Delete Local Data" option available in Settings

## Development

```bash
npm run dev        # Start Vite dev server
npm run app        # Build + launch Electron
npm run typecheck  # TypeScript strict mode check
npm run test       # Run unit tests
npm run dist:win   # Build Windows installer
```

## Troubleshooting

- "No AI Provider Configured" — Go to Settings → AI Providers and configure a provider
- "Connection failed" — Check your API key and internet connection
- "Rate limit reached" — Wait a moment and retry
- "Model unavailable" — Select a different model in provider settings
- App won't launch — Ensure Windows 10/11 x64, check antivirus isn't blocking

## Contributing

Contributions welcome. Please ensure:
- TypeScript strict mode passes
- No API keys committed
- Tests pass (`npm run test`)

## License

MIT

## Disclaimer

Edify AI is an independent project and is not affiliated with NitroAI.
