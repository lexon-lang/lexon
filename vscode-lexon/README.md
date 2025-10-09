# LEXON - AI-Native Programming Language Extension

The official VS Code extension for **LEXON**, the world's first LLM-native programming language with async/await support.

## Features

### 🎨 Syntax Highlighting
- Complete syntax highlighting for LEXON language
- Special highlighting for `async`/`await` keywords
- LLM function highlighting (`ask`, `ask_safe`, `ask_parallel`, etc.)
- Data operation highlighting (`map`, `filter`, `reduce`, etc.)

### ⚡ Async/Await Support
- Intelligent snippets for async functions
- Auto-completion for `async fn` declarations
- `await` expression snippets
- Hover documentation for async/await concepts

### 🤖 LLM Function Snippets
- **ask** - Query LLM models
- **ask_safe** - Anti-hallucination validation
- **ask_parallel** - Parallel prompt execution
- **ask_ensemble** - Consensus-based responses
- **ask_multioutput** - Generate multiple files
- **model_arbitrage** - Model debate system
- **session_ask** - Persistent context

### 🔧 Development Tools
- **Run File** (`Ctrl+Shift+R` / `Cmd+Shift+R`) - Execute LEXON files
- **Check Syntax** - Validate syntax
- **Lint File** (`Ctrl+Shift+L` / `Cmd+Shift+L`) - Missing await warnings
- Auto-linting on save and change
- Real-time error diagnostics

### 📊 Data Processing Snippets
- **range**, **map**, **filter**, **reduce** - Functional programming
- **data** - CSV file loading
- **memory_store**, **memory_load** - Memory operations
- **vector_search** - RAG functionality

## Installation

1. Install the LEXON compiler (`lexc`) on your system
2. Install this extension from the VS Code marketplace
3. Configure the executable path in settings (if needed)

## Configuration

Access settings via `File > Preferences > Settings` and search for "LEXON":

- **lexon.executablePath**: Path to lexc executable (default: "lexc")
- **lexon.enableLinting**: Enable/disable auto-linting (default: true)
- **lexon.defaultModel**: Default LLM model (default: "gpt-4")
- **lexon.enableAsyncHighlighting**: Special async/await highlighting (default: true)

## Usage

### Creating an Async Function

Type `async` and use the snippet:

```lexon
async fn analyze_data() {
    let data = await load_data("file.csv");
    let result = await ask("Analyze this data", "gpt-4");
    return result;
}
```

### LLM Operations

Type `ask` for basic LLM queries:

```lexon
let response = ask("What is machine learning?", "gpt-4");
```

Type `ask_safe` for validated responses:

```lexon
let result = ask_safe("Explain quantum physics", "gpt-4", {
    validation_strategy: "basic",
    confidence_threshold: 0.8
});
```

### Data Processing

Type `map`, `filter`, or `reduce` for functional operations:

```lexon
let numbers = range(1, 10);
let doubled = map(numbers, 'x * 2');
let evens = filter(doubled, 'x % 2 == 0');
let sum = reduce(evens, 0, 'acc + x');
```

## Commands

- **LEXON: Run File** - Execute the current LEXON file
- **LEXON: Check Syntax** - Check file syntax
- **LEXON: Lint File** - Run linter with async/await warnings

## Keyboard Shortcuts

- `Ctrl+Shift+R` (Windows/Linux) / `Cmd+Shift+R` (Mac) - Run file
- `Ctrl+Shift+L` (Windows/Linux) / `Cmd+Shift+L` (Mac) - Lint file

## Language Features

### Async/Await
LEXON is the first LLM-native language with full async/await support:

```lexon
async fn process_documents() {
    let summaries = await ask_parallel([
        "Summarize document 1",
        "Summarize document 2"
    ], "gpt-4");
    
    let consensus = await ask_ensemble(summaries, "MajorityVote", "gpt-4");
    return consensus;
}
```

### Anti-Hallucination
Built-in validation and confidence scoring:

```lexon
let score = confidence_score("Paris is the capital of France");
let is_valid = validate_response("Response text", "basic");
```

### Multimodal Operations
Generate multiple files in a single operation:

```lexon
let website = ask_multioutput("Create a landing page", [
    "index.html",
    "styles.css",
    "script.js"
]);
```

## Requirements

- VS Code 1.74.0 or higher
- LEXON compiler (`lexc`) installed on system
- Node.js 16.x or higher (for development)

## Extension Development

To contribute to this extension:

1. Clone the repository
2. Run `npm install`
3. Open in VS Code
4. Press `F5` to launch extension development host
5. Make changes and test

## Release Notes

### 1.0.0

- Initial release
- Complete syntax highlighting
- Async/await support
- LLM function snippets
- Auto-linting with missing await warnings
- Hover documentation
- Auto-completion
- Command palette integration

## Support

- [GitHub Repository](https://github.com/lexon-lang/lexon)
- [Documentation](https://lexon-lang.org)
- [Issue Tracker](https://github.com/lexon-lang/lexon/issues)

## License

MIT License - see LICENSE file for details.

---

**LEXON**: Revolutionizing AI-native programming with async/await and LLM integration. 