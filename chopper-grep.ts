import { createParserFactory, readDirectoryAndChunk, type Options, type BoundaryChunk } from "code-chopper";
import * as path from 'path';
import { env, pipeline } from '@huggingface/transformers';
import { bulkInsertChunks, searchSimilar, close, EMBEDDING_DIM, CodeChunkRow } from './db'; // Import from db.ts

// Set environment variables for transformers.js
env.allowRemoteModels = true; // Allow fetching models from Hugging Face Hub if not found locally

// Initialize the embedding pipeline
const embeddingPipeline = await pipeline('feature-extraction', 'sirasagi62/granite-embedding-107m-multilingual-ONNX', { dtype: 'q8' });

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
        console.log(embeddingArray[EMBEDDING_DIM])
        throw new Error(`Embedding dimension mismatch: expected ${EMBEDDING_DIM}, got ${embeddingArray.length}. Truncating or padding.`);

      }
      return new Float32Array(embeddingArray);
    }
    console.warn("Unexpected output format from embedding pipeline:", output);
    return new Float32Array(EMBEDDING_DIM); // Return empty embedding of correct dimension
  } catch (error) {
    console.error("Error calculating embedding:", error);
    return new Float32Array(EMBEDDING_DIM); // Return empty embedding on error
  };
}

async function indexDirectory(dirPath: string): Promise<void> {
  const factory = createParserFactory();
  const options: Options = {
    filter: (_, node) => {
      if (node.type.includes("import") || node.type.includes("comment")) {
        return false;
      }
      return true;
    },
    excludeDirs: [/node_modules/, /\.git/]
  };

  const chunksToInsert: CodeChunkRow[] = [];

  try {
    const chunks = await readDirectoryAndChunk(factory, options, dirPath);

    for (const chunk of chunks) {
      const embedding = await calculateEmbedding(chunk.content);
      chunksToInsert.push({
        file_name: path.basename(chunk.filePath),
        file_path: chunk.filePath,
        chunk_text: chunk.content,
        inline_document: chunk.boundary.docs || '', // Assuming BoundaryChunk has a 'doc' property for documentation
        embedding: embedding
      });
    }

    if (chunksToInsert.length > 0) {
      bulkInsertChunks(chunksToInsert); // Use bulkInsertChunks for efficiency
      console.log(`Indexed ${chunksToInsert.length} code chunks.`);
    } else {
      console.log("No code chunks found to index.");
    }
  } catch (error) {
    console.error(`Error during directory indexing:`, error);
  } finally {
    factory.dispose();
  }
}

async function query(queryText: string, k: number = 5): Promise<void> {
  const queryEmbedding = await calculateEmbedding(queryText);
  const results = searchSimilar(queryEmbedding, k); // Use searchSimilar from db.ts

  if (results.length > 0) {
    console.log(`Top ${k} results for query "${queryText}":`);
    results.forEach((result, index) => {
      console.log(`${index + 1}. File: ${result.file_path}`);
      console.log(`   Type: ${result.chunk_text.split('\n')[0]}...`); // Simplified type display
      console.log(`   Name: ${result.file_name}`);
      console.log(`   Content Snippet: ${result.chunk_text.substring(0, 100)}...`);
      console.log(`   Score: ${result.distance.toFixed(4)}`); // Distance is used as a similarity score here
    });
  } else {
    console.log(`No results found for query "${queryText}".`);
  }
}

async function main() {
  const args = process.argv.slice(2);
  const command = args[0];

  if (command === 'index') {
    const dirToIndex = args[1] || '.';
    console.log(`Indexing directory: ${path.resolve(dirToIndex)}`);
    await indexDirectory(dirToIndex);
    console.log("Indexing complete.");
  } else if (command === 'query') {
    const queryText = args[1];
    if (!queryText) {
      console.error("Query text is required.");
      process.exit(1);
    }
    const k = parseInt(args[2] || '5', 10);
    await query(queryText, k);
  } else {
    console.log("Usage:");
    console.log("  ts-node chopper-grep.ts index [directory_path]");
    console.log("  ts-node chopper-grep.ts query <query_text> [k]");
  }

  close(); // Close the database connection
}

main().catch(error => {
  console.error("An unexpected error occurred:", error);
  close(); // Ensure DB is closed even on error
  process.exit(1);
});
