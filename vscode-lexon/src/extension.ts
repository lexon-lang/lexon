import { exec } from 'child_process';
import * as vscode from 'vscode';

export function activate(context: vscode.ExtensionContext) {
    console.log('LEXON extension is now active!');

    // Register commands
    const runFileCommand = vscode.commands.registerCommand('lexon.runFile', () => {
        runLexonFile();
    });

    const checkSyntaxCommand = vscode.commands.registerCommand('lexon.checkSyntax', () => {
        checkLexonSyntax();
    });

    const lintFileCommand = vscode.commands.registerCommand('lexon.lintFile', () => {
        lintLexonFile();
    });

    // Register diagnostics provider
    const diagnosticCollection = vscode.languages.createDiagnosticCollection('lexon');
    context.subscriptions.push(diagnosticCollection);

    // Auto-lint on save
    const onSaveDisposable = vscode.workspace.onDidSaveTextDocument(document => {
        if (document.languageId === 'lexon') {
            const config = vscode.workspace.getConfiguration('lexon');
            if (config.get('enableLinting')) {
                lintDocument(document, diagnosticCollection);
            }
        }
    });

    // Auto-lint on change (debounced)
    let lintTimeout: NodeJS.Timeout;
    const onChangeDisposable = vscode.workspace.onDidChangeTextDocument(event => {
        if (event.document.languageId === 'lexon') {
            const config = vscode.workspace.getConfiguration('lexon');
            if (config.get('enableLinting')) {
                clearTimeout(lintTimeout);
                lintTimeout = setTimeout(() => {
                    lintDocument(event.document, diagnosticCollection);
                }, 1000);
            }
        }
    });

    // Register hover provider for async/await
    const hoverProvider = vscode.languages.registerHoverProvider('lexon', {
        provideHover(document, position, token) {
            const range = document.getWordRangeAtPosition(position);
            const word = document.getText(range);

            switch (word) {
                case 'async':
                    return new vscode.Hover([
                        '**async** keyword',
                        'Declares an asynchronous function that returns a Future<T>',
                        '```lexon\nasync fn my_function() {\n    let result = await some_operation();\n    return result;\n}\n```'
                    ]);
                case 'await':
                    return new vscode.Hover([
                        '**await** expression',
                        'Waits for a Future<T> to complete and returns T',
                        '```lexon\nlet result = await async_operation();\n```'
                    ]);
                case 'ask':
                    return new vscode.Hover([
                        '**ask** function',
                        'Queries an LLM model with a prompt',
                        '```lexon\nlet response = ask("What is 2+2?", "gpt-4");\n```'
                    ]);
                case 'ask_safe':
                    return new vscode.Hover([
                        '**ask_safe** function',
                        'Queries an LLM with anti-hallucination validation',
                        '```lexon\nlet result = ask_safe("prompt", "gpt-4", {\n    validation_strategy: "basic",\n    confidence_threshold: 0.8\n});\n```'
                    ]);
                default:
                    return null;
            }
        }
    });

    // Register completion provider
    const completionProvider = vscode.languages.registerCompletionItemProvider('lexon', {
        provideCompletionItems(document, position, token, context) {
            const completions: vscode.CompletionItem[] = [];

            // Add async/await completions
            const asyncCompletion = new vscode.CompletionItem('async', vscode.CompletionItemKind.Keyword);
            asyncCompletion.detail = 'Async function declaration';
            asyncCompletion.documentation = 'Declares an asynchronous function';
            asyncCompletion.insertText = new vscode.SnippetString('async fn ${1:function_name}() {\n    ${2:// async function body}\n    let result = await ${3:async_operation()};\n    return result;\n}');
            completions.push(asyncCompletion);

            const awaitCompletion = new vscode.CompletionItem('await', vscode.CompletionItemKind.Keyword);
            awaitCompletion.detail = 'Await expression';
            awaitCompletion.documentation = 'Waits for a Future<T> to complete';
            awaitCompletion.insertText = new vscode.SnippetString('await ${1:async_expression}');
            completions.push(awaitCompletion);

            // Add LLM function completions
            const llmFunctions = [
                { name: 'ask', detail: 'Query LLM model', snippet: 'ask("${1:prompt}", "${2:gpt-4}")' },
                { name: 'ask_safe', detail: 'Query LLM with validation', snippet: 'ask_safe("${1:prompt}", "${2:gpt-4}", {\n    validation_strategy: "${3:basic}",\n    confidence_threshold: ${4:0.8}\n})' },
                { name: 'ask_parallel', detail: 'Query multiple prompts in parallel', snippet: 'ask_parallel([\n    "${1:prompt1}",\n    "${2:prompt2}"\n], "${3:gpt-4}")' },
                { name: 'ask_ensemble', detail: 'Query with ensemble consensus', snippet: 'ask_ensemble([\n    "${1:prompt1}",\n    "${2:prompt2}"\n], "${3:MajorityVote}", "${4:gpt-4}")' }
            ];

            llmFunctions.forEach(func => {
                const completion = new vscode.CompletionItem(func.name, vscode.CompletionItemKind.Function);
                completion.detail = func.detail;
                completion.insertText = new vscode.SnippetString(func.snippet);
                completions.push(completion);
            });

            return completions;
        }
    });

    // Add all disposables to context
    context.subscriptions.push(
        runFileCommand,
        checkSyntaxCommand,
        lintFileCommand,
        onSaveDisposable,
        onChangeDisposable,
        hoverProvider,
        completionProvider
    );
}

function runLexonFile() {
    const editor = vscode.window.activeTextEditor;
    if (!editor) {
        vscode.window.showErrorMessage('No active LEXON file');
        return;
    }

    const document = editor.document;
    if (document.languageId !== 'lexon') {
        vscode.window.showErrorMessage('Current file is not a LEXON file');
        return;
    }

    const filePath = document.fileName;
    const config = vscode.workspace.getConfiguration('lexon');
    const executablePath = config.get('executablePath', 'lexc');

    const terminal = vscode.window.createTerminal('LEXON');
    terminal.show();
    terminal.sendText(`${executablePath} "${filePath}"`);
}

function checkLexonSyntax() {
    const editor = vscode.window.activeTextEditor;
    if (!editor) {
        vscode.window.showErrorMessage('No active LEXON file');
        return;
    }

    const document = editor.document;
    if (document.languageId !== 'lexon') {
        vscode.window.showErrorMessage('Current file is not a LEXON file');
        return;
    }

    const filePath = document.fileName;
    const config = vscode.workspace.getConfiguration('lexon');
    const executablePath = config.get('executablePath', 'lexc');

    exec(`${executablePath} check "${filePath}"`, (error, stdout, stderr) => {
        if (error) {
            vscode.window.showErrorMessage(`Syntax check failed: ${stderr}`);
        } else {
            vscode.window.showInformationMessage('Syntax check passed!');
        }
    });
}

function lintLexonFile() {
    const editor = vscode.window.activeTextEditor;
    if (!editor) {
        vscode.window.showErrorMessage('No active LEXON file');
        return;
    }

    const document = editor.document;
    if (document.languageId !== 'lexon') {
        vscode.window.showErrorMessage('Current file is not a LEXON file');
        return;
    }

    const filePath = document.fileName;
    const config = vscode.workspace.getConfiguration('lexon');
    const executablePath = config.get('executablePath', 'lexc');

    exec(`${executablePath} --lint "${filePath}"`, (error, stdout, stderr) => {
        if (error) {
            vscode.window.showErrorMessage(`Linting failed: ${stderr}`);
        } else {
            vscode.window.showInformationMessage('Linting completed!');
            vscode.window.showInformationMessage(stdout);
        }
    });
}

function lintDocument(document: vscode.TextDocument, diagnosticCollection: vscode.DiagnosticCollection) {
    const filePath = document.fileName;
    const config = vscode.workspace.getConfiguration('lexon');
    const executablePath = config.get('executablePath', 'lexc');

    exec(`${executablePath} --lint "${filePath}"`, (error, stdout, stderr) => {
        const diagnostics: vscode.Diagnostic[] = [];

        if (stdout) {
            // Parse linting output for warnings
            const lines = stdout.split('\n');
            lines.forEach(line => {
                const match = line.match(/Warning: (.+) at line (\d+)/);
                if (match) {
                    const message = match[1];
                    const lineNumber = parseInt(match[2]) - 1; // VS Code uses 0-based line numbers
                    
                    const diagnostic = new vscode.Diagnostic(
                        new vscode.Range(lineNumber, 0, lineNumber, Number.MAX_VALUE),
                        message,
                        vscode.DiagnosticSeverity.Warning
                    );
                    diagnostic.source = 'lexon-linter';
                    diagnostics.push(diagnostic);
                }
            });
        }

        diagnosticCollection.set(document.uri, diagnostics);
    });
}

export function deactivate() {} 