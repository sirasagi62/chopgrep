#!/usr/bin/env bun
import { createParserFactory, readDirectoryAndChunk, } from "code-chopper";
import * as path from "path";
import { env } from "@huggingface/transformers";
import { HFLocalEmbeddingModel, VeqliteDB } from "veqlite";
import yargs from "yargs"; // Import yargs
import { hideBin } from "yargs/helpers"; // Helper to get arguments excluding node and script path
// Set environment variables for transformers.js
env.allowRemoteModels = true; // Allow fetching models from Hugging Face Hub if not found locally
async function initDB() {
    // Initialize the embedding pipeline
    const embeddingModel = await HFLocalEmbeddingModel.init("sirasagi62/granite-embedding-107m-multilingual-ONNX", 384, "q8");
    const db = new VeqliteDB(embeddingModel, {
        // Use in-memory database
        embeddingDim: 384,
        dbPath: ":memory:"
    });
    return db;
}
async function indexDirectory(dirPath, isJsonOutput) {
    const db = await initDB();
    const factory = createParserFactory();
    const options = {
        filter: (_, node) => {
            if (node.type.includes("import") || node.type.includes("comment")) {
                return false;
            }
            return true;
        },
        excludeDirs: [/node_modules/, /\.git/],
    };
    let indexedCount = 0;
    try {
        const chunks = await readDirectoryAndChunk(factory, options, dirPath);
        if (chunks.length > 0) {
            db.bulkInsertChunks(chunks.map(c => {
                return {
                    content: c.content,
                    filepath: c.filePath,
                    fileName: path.basename(c.filePath),
                    inlineDocument: c.boundary.docs ?? "",
                    parentInfo: c.boundary.parent?.join(".") ?? "",
                    entity: c.boundary.name ?? "",
                };
            })); // Use bulkInsertChunks for efficiency
            indexedCount = chunks.length;
            if (isJsonOutput) {
                console.log(JSON.stringify({
                    status: "success",
                    message: `Indexed ${indexedCount} code chunks.`,
                    directory: path.resolve(dirPath),
                }));
            }
            else {
                console.log(`Indexed ${indexedCount} code chunks.`);
            }
        }
        else {
            if (isJsonOutput) {
                console.log(JSON.stringify({
                    status: "success",
                    message: "No code chunks found to index.",
                    directory: path.resolve(dirPath),
                }));
            }
            else {
                console.log("No code chunks found to index.");
            }
        }
    }
    catch (error) {
        if (isJsonOutput) {
            if (error instanceof Error)
                console.error(JSON.stringify({
                    status: "error",
                    message: `Error during directory indexing: ${error.message}`,
                    directory: path.resolve(dirPath),
                }));
        }
        else {
            console.error(`Error during directory indexing:`, error);
        }
    }
    finally {
        factory.dispose();
    }
}
async function query(queryText, k, isJsonOutput) {
    const db = await initDB();
    const results = await db.searchSimilar(queryText, k);
    if (results.length > 0) {
        const output = results.map((result, index) => ({
            rank: index + 1,
            file: result.filepath,
            fileName: result.fileName,
            contentSnippet: isJsonOutput
                ? result.content
                : result.content.substring(0, 100) + "...",
            entity: result.entity,
            parent_info: result.parentInfo,
            score: result.distance.toFixed(4),
        }));
        if (isJsonOutput) {
            console.log(JSON.stringify({
                status: "success",
                query: queryText,
                k: k,
                results: output,
            }));
        }
        else {
            console.log(`Top ${k} results for query "${queryText}":`);
            output.forEach((res) => {
                console.log(`- File: ${res.file}`);
                console.log(`  Content Snippet: ${res.contentSnippet}`);
                console.log(`  Score: ${res.score}`);
                console.log(`  Entity: ${res.entity}`);
                console.log(`  Parent: ${res.parent_info}`);
            });
        }
    }
    else {
        if (isJsonOutput) {
            console.log(JSON.stringify({
                status: "success",
                query: queryText,
                k: k,
                results: [],
            }));
        }
        else {
            console.log(`No results found for query "${queryText}".`);
        }
    }
}
function main() {
    const msg = yargs(hideBin(process.argv))
        .command("index [directory]", "Index code chunks from a directory", (yargs) => {
        return yargs.positional("directory", {
            describe: "The directory to index",
            default: ".",
        });
    }, async (args) => {
        await indexDirectory(args.directory, args.json);
        //db.close(); // Close the database connection after command execution
    })
        .command("query <queryText> [k]", "Search for code chunks similar to the query text", (yargs) => {
        return yargs
            .positional("queryText", {
            describe: "The text to query",
        })
            .positional("k", {
            describe: "Number of results to return",
            default: 5,
        });
    }, async (args) => {
        await query(args.queryText, args.k, args.json);
        //db.close(); // Close the database connection after command execution
    })
        .option("json", {
        alias: "j",
        type: "boolean",
        description: "Output results in JSON format",
        default: false,
    })
        .demandCommand(1, "You need to provide a command (index or query).")
        .help("h")
        .alias("help", "h")
        .strict() // Enforce strict argument parsing
        .parse();
}
try {
    main();
}
catch {
    console.error("An unexpected error occurred:");
    //db.close(); // Ensure DB is closed even on error
    process.exit(1);
}
;
