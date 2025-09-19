// db.ts
import Database from "bun:sqlite";
import * as sqliteVec from "sqlite-vec";

export type CodeChunkRow = {
  id?: number;
  file_name: string;
  file_path: string;
  chunk_text: string;
  inline_document?: string | null;
  // embedding as Float32Array (length EMBEDDING_DIM)
  embedding: Float32Array;
};

const DB_PATH = "code_chunks.db";
export const EMBEDDING_DIM = 384;

const db = new Database(DB_PATH);

// load sqlite-vec extension/wrapping (this registers functions / virtual tables)
sqliteVec.load(db);

// --- Schema ---
db.exec(`
CREATE TABLE IF NOT EXISTS code_chunks (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  file_name TEXT NOT NULL,
  file_path TEXT NOT NULL,
  chunk_text TEXT NOT NULL,
  inline_document TEXT,
  embedding BLOB NOT NULL
);
`);

// vec0 virtual table for indexing vectors
db.exec(`
CREATE VIRTUAL TABLE IF NOT EXISTS vec_index USING vec0(
  embedding float[${EMBEDDING_DIM}] distance_metric=cosine
);
`);

// triggers to keep vec_index in sync with code_chunks
db.exec(`
CREATE TRIGGER IF NOT EXISTS code_chunks_after_insert
AFTER INSERT ON code_chunks
BEGIN
  INSERT INTO vec_index(rowid, embedding) VALUES (new.id, new.embedding);
END;
`);
db.exec(`
CREATE TRIGGER IF NOT EXISTS code_chunks_after_delete
AFTER DELETE ON code_chunks
BEGIN
  DELETE FROM vec_index WHERE rowid = old.id;
END;
`);
db.exec(`
CREATE TRIGGER IF NOT EXISTS code_chunks_after_update
AFTER UPDATE ON code_chunks
BEGIN
  DELETE FROM vec_index WHERE rowid = old.id;
  INSERT INTO vec_index(rowid, embedding) VALUES (new.id, new.embedding);
END;
`);

// --- Helpers ---
function assertEmbeddingDim(arr: Float32Array) {
  if (arr.length !== EMBEDDING_DIM) {
    throw new Error(`embedding must have length ${EMBEDDING_DIM}, got ${arr.length}`);
  }
}
function float32ToBuffer(arr: Float32Array): Buffer {
  // ensure a copy with proper byteOffset/length
  return Buffer.from(arr.buffer, arr.byteOffset, arr.byteLength);
}

// Insert single chunk
export function insertChunk(chunk: CodeChunkRow): number {
  assertEmbeddingDim(chunk.embedding);
  const buf = float32ToBuffer(chunk.embedding);
  const stmt = db.prepare(
    `INSERT INTO code_chunks (file_name, file_path, chunk_text, inline_document, embedding)
     VALUES (?, ?, ?, ?, ?)`
  );
  const info = stmt.run(chunk.file_name, chunk.file_path, chunk.chunk_text, chunk.inline_document ?? null, buf);
  return Number(info.lastInsertRowid);
}

// Bulk insert array of chunks (transactional)
export function bulkInsertChunks(chunks: CodeChunkRow[], batchSize = 500) {
  // chunked batch insert to avoid giant single statement / memory spikes
  const insertOne = db.prepare(
    `INSERT INTO code_chunks (file_name, file_path, chunk_text, inline_document, embedding)
     VALUES (?, ?, ?, ?, ?)`
  );
  const insertMany = db.transaction((batch: CodeChunkRow[]) => {
    for (const c of batch) {
      assertEmbeddingDim(c.embedding);
      insertOne.run(c.file_name, c.file_path, c.chunk_text, c.inline_document ?? null, float32ToBuffer(c.embedding));
    }
  });

  for (let i = 0; i < chunks.length; i += batchSize) {
    const slice = chunks.slice(i, i + batchSize);
    insertMany(slice);
  }
}

// Search top-k similar using vec_index KNN (returns joined code_chunks rows plus distance)
export type SearchResult = {
  id: number;
  file_name: string;
  file_path: string;
  chunk_text: string;
  inline_document: string | null;
  distance: number;
};

export function searchSimilar(queryEmbedding: Float32Array, k = 5): SearchResult[] {
  assertEmbeddingDim(queryEmbedding);
  const qBuf = float32ToBuffer(queryEmbedding);

  // Use vec_index.match KNN; vec_index.embedding MATCH ? accepts either JSON text or binary
  // We pass binary. k is provided via parameter for vec_index.k; LIMIT is provided too.
  const sql = `
    WITH q AS (SELECT ? AS embedding)
    SELECT
      c.id,
      c.file_name,
      c.file_path,
      c.chunk_text,
      c.inline_document,
      v.distance as distance
    FROM vec_index v
    JOIN q ON 1=1
    JOIN code_chunks c ON c.id = v.rowid
    WHERE v.embedding MATCH q.embedding
      AND v.k = ?
    ORDER BY distance
    LIMIT ?
  `;
  const stmt = db.prepare(sql);
  // parameters: qBuf, k (for v.k), k (for LIMIT). Some sqlite-vec usages pass k as v.k parameter; include both.
  const rows = stmt.all(qBuf, k, k) as any[];
  return rows.map(r => ({
    id: Number(r.id),
    file_name: r.file_name,
    file_path: r.file_path,
    chunk_text: r.chunk_text,
    inline_document: r.inline_document,
    distance: Number(r.distance)
  }));
}

export function close() {
  db.close();
}

