# chopgrep


> [!WARNING]
> This project is a work in progress.

## Installation

To install the project dependencies, run the following command in your terminal:

```bash
bun link
```

## Usage

### Indexing a Directory

To index code chunks from a directory, use the `index` command:

```bash
bun run chopper-grep.ts index <directory_path>
```

If you've already install `chopgrep`,
```bash
chopgrep index <directory_path>
```

- Replace `<directory_path>` with the path to the directory you want to index. If omitted, it defaults to the current
directory (`.`).
- To output results in JSON format, use the `--json` or `-j` flag.

**Example:**
```bash
chopgrep index ./src -j
```

### Querying Code Chunks

To search for code chunks similar to a given query text, use the `query` command:

```bash
chopgrep query <query_text> [k]
```

- Replace `<query_text>` with the text you want to search for.
- `[k]` is an optional argument specifying the number of results to return (defaults to 5).
- To output results in JSON format, use the `--json` or `-j` flag.

**Example:**
```bash
chopgrep chopper-grep.ts query "function to calculate factorial" 3 -j
```

## Project Structure

- `chopper-grep.ts`: The main entry point for the CLI tool.
- `db.ts`: Handles all database operations for storing and retrieving code chunks and their embeddings.
- `package.json`: Project metadata and dependencies.
- `tsconfig.json`: TypeScript compiler options.
- `.gitignore`: Specifies intentionally untracked files that Git should ignore.

This project was created using `bun init` in bun v1.2.21.
