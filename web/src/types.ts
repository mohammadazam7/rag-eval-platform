/** Shape of web/public/data.json, written by scripts/export_gallery_data.py. */

export interface GalleryChunk {
  id: string;
  doc_id: string;
  x: number;
  y: number;
  z: number;
  preview: string;
  gold: boolean;
}

export interface RetrievalHit {
  chunk_id: string;
  rank: number;
  score: number;
}

export type RetrievalMethod = "bm25" | "hybrid";

export interface GalleryData {
  corpus: string;
  question: {
    id: string;
    text: string;
    gold_chunk_ids: string[];
  };
  chunks: GalleryChunk[];
  retrieval: Record<RetrievalMethod, RetrievalHit[]>;
}
