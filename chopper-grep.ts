#!/usr/bin/env bun
import {
  createParserFactory,
  readDirectoryAndChunk,
  type Options,
} from "code-chopper";
import * as path from "path";
import { env, pipeline } from "@huggingface/transformers";
import {
  bulkInsertChunks,
  searchSimilar,
  close,
  EMBEDDING_DIM,
  type CodeChunkRow,
} from "./db"; // Import from db.ts
import yargs from "yargs"; // Import yargs
import { hideBin } from "yargs/helpers"; // Helper to get arguments excluding node and script path

// Set environment variables for transformers.js
env.allowRemoteModels = true; // Allow fetching models from Hugging Face Hub if not found locally

// Initialize the embedding pipeline
const embeddingPipeline = await pipeline(
  "feature-extraction",
  "sirasagi62/granite-embedding-107m-multilingual-ONNX",
  { dtype: "q8" }
);

// Function to calculate embeddings using the transformers pipeline
const calculateEmbedding = async (text: string): Promise<Float32Array> => {
  try {
    const output = await embeddingPipeline(text, {
      pooling: "mean",
      normalize: true,
    });
    const embeddingArray = output.data;
    if (embeddingArray) {
      // Ensure the output is a Float32Array and matches the expected dimension
      if (embeddingArray.length !== EMBEDDING_DIM) {
        console.warn(
          `Embedding dimension mismatch: expected ${EMBEDDING_DIM}, got ${embeddingArray.length}. Truncating or padding.`
        );
        // Basic handling: truncate or pad if necessary. A more robust solution might be needed.
        const truncatedOrPadded = new Float32Array(EMBEDDING_DIM);
        const copyLength = Math.min(embeddingArray.length, EMBEDDING_DIM);
        truncatedOrPadded.set(Float32Array.from(embeddingArray).slice(0, copyLength));
        return truncatedOrPadded;
      }
      return Float32Array.from(embeddingArray);
    }
    console.warn("Unexpected output format from embedding pipeline:", output);
    return new Float32Array(EMBEDDING_DIM); // Return empty embedding of correct dimension
  } catch (error) {
    console.error("Error calculating embedding:", error);
    return new Float32Array(EMBEDDING_DIM); // Return empty embedding on error
  }
};

async function indexDirectory(
  dirPath: string,
  isJsonOutput: boolean
): Promise<void> {
  const factory = createParserFactory();
  const options: Options = {
    filter: (_, node) => {
      if (node.type.includes("import") || node.type.includes("comment")) {
        return false;
      }
      return true;
    },
    excludeDirs: [/node_modules/, /\.git/],
  };

  const chunksToInsert: CodeChunkRow[] = [];
  let indexedCount = 0;

  try {
    const chunks = await readDirectoryAndChunk(factory, options, dirPath);

    for (const chunk of chunks) {
      const embedding = await calculateEmbedding(chunk.content);
      chunksToInsert.push({
        file_name: path.basename(chunk.filePath),
        file_path: chunk.filePath,
        chunk_text: chunk.content,
        inline_document: chunk.boundary.docs || "", // Assuming BoundaryChunk has a 'doc' property for documentation
        embedding: embedding,
        parent_info: chunk.boundary.parent?.join(".") || "",
        entity: chunk.boundary.name || "",
      });
    }

    if (chunksToInsert.length > 0) {
      bulkInsertChunks(chunksToInsert); // Use bulkInsertChunks for efficiency
      indexedCount = chunksToInsert.length;
      if (isJsonOutput) {
        console.log(
          JSON.stringify({
            status: "success",
            message: `Indexed ${indexedCount} code chunks.`,
            directory: path.resolve(dirPath),
          })
        );
      } else {
        console.log(`Indexed ${indexedCount} code chunks.`);
      }
    } else {
      if (isJsonOutput) {
        console.log(
          JSON.stringify({
            status: "success",
            message: "No code chunks found to index.",
            directory: path.resolve(dirPath),
          })
        );
      } else {
        console.log("No code chunks found to index.");
      }
    }
  } catch (error: unknown) {
    if (isJsonOutput) {
      if (error instanceof Error)
        console.error(
          JSON.stringify({
            status: "error",
            message: `Error during directory indexing: ${error.message}`,
            directory: path.resolve(dirPath),
          })
        );
    } else {
      console.error(`Error during directory indexing:`, error);
    }
  } finally {
    factory.dispose();
  }
}

async function query(
  queryText: string,
  k: number,
  isJsonOutput: boolean
): Promise<void> {
  const queryEmbedding = await calculateEmbedding(queryText);
  const results = searchSimilar(queryEmbedding, k); // Use searchSimilar from db.ts

  if (results.length > 0) {
    const output = results.map((result, index) => ({
      rank: index + 1,
      file: result.file_path,
      fileName: result.file_name,
      contentSnippet: isJsonOutput
        ? result.chunk_text
        : result.chunk_text.substring(0, 100) + "...",
      entity: result.entity,
      parent_info: result.parent_info,
      score: result.distance.toFixed(4),
    }));

    if (isJsonOutput) {
      console.log(
        JSON.stringify({
          status: "success",
          query: queryText,
          k: k,
          results: output,
        })
      );
    } else {
      console.log(`Top ${k} results for query "${queryText}":`);
      output.forEach((res) => {
        console.log(`- File: ${res.file}`);
        console.log(`  Content Snippet: ${res.contentSnippet}`);
        console.log(`  Score: ${res.score}`);
        console.log(`  Entity: ${res.entity}`);
        console.log(`  Parent: ${res.parent_info}`);
      });
    }
  } else {
    if (isJsonOutput) {
      console.log(
        JSON.stringify({
          status: "success",
          query: queryText,
          k: k,
          results: [],
        })
      );
    } else {
      console.log(`No results found for query "${queryText}".`);
    }
  }
}

async function main() {
  await yargs(hideBin(process.argv))
    .command(
      "index [directory]",
      "Index code chunks from a directory",
      (yargs) => {
        return yargs.positional("directory", {
          describe: "The directory to index",
          default: ".",
        });
      },
      async (args) => {
        await indexDirectory(args.directory as string, args.json as boolean);
        close(); // Close the database connection after command execution
      }
    )
    .command(
      "query <queryText> [k]",
      "Search for code chunks similar to the query text",
      (yargs) => {
        return yargs
          .positional("queryText", {
            describe: "The text to query",
          })
          .positional("k", {
            describe: "Number of results to return",
            default: 5,
          });
      },
      async (args) => {
        await query(
          args.queryText as string,
          args.k as number,
          args.json as boolean
        );
        close(); // Close the database connection after command execution
      }
    )
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

main().catch((error) => {
  console.error("An unexpected error occurred:", error);
  close(); // Ensure DB is closed even on error
  process.exit(1);
});
