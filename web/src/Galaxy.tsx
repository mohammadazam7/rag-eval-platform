import { OrbitControls } from "@react-three/drei";
import { Canvas } from "@react-three/fiber";
import { useMemo } from "react";

import { chunkStatus } from "./gallery";
import type { GalleryChunk, GalleryData, RetrievalMethod } from "./types";

const COLOR = {
  "gold-and-retrieved": "#22c55e",
  gold: "#eab308",
  retrieved: "#38bdf8",
  none: "#64748b",
} as const;

const SCALE = 6;

function Point({
  chunk,
  selected,
  color,
  onSelect,
}: {
  chunk: GalleryChunk;
  selected: boolean;
  color: string;
  onSelect: (chunkId: string) => void;
}) {
  return (
    <mesh
      position={[chunk.x * SCALE, chunk.y * SCALE, chunk.z * SCALE]}
      onClick={(event) => {
        event.stopPropagation();
        onSelect(chunk.id);
      }}
    >
      <sphereGeometry args={[selected ? 0.14 : 0.06, 12, 12]} />
      <meshStandardMaterial
        color={color}
        emissive={selected ? color : "#000000"}
        emissiveIntensity={selected ? 0.8 : 0}
      />
    </mesh>
  );
}

interface Props {
  data: GalleryData;
  method: RetrievalMethod;
  selectedId: string | null;
  onSelect: (chunkId: string) => void;
}

/** The chunks laid out in 3D by embedding similarity (see
 * scripts/export_gallery_data.py), coloured by whether the selected retrieval method
 * found each one and whether it is part of the question's gold evidence. Decorative:
 * ChunkTable carries the same information and all of the keyboard access. */
export function Galaxy({ data, method, selectedId, onSelect }: Props) {
  const points = useMemo(
    () => data.chunks.map((chunk) => ({ chunk, status: chunkStatus(data, method, chunk) })),
    [data, method],
  );

  return (
    <Canvas camera={{ position: [0, 0, 10], fov: 50 }} aria-hidden="true">
      <ambientLight intensity={0.6} />
      <pointLight position={[10, 10, 10]} intensity={1.2} />
      {points.map(({ chunk, status }) => (
        <Point
          key={chunk.id}
          chunk={chunk}
          selected={chunk.id === selectedId}
          color={COLOR[status]}
          onSelect={onSelect}
        />
      ))}
      <OrbitControls enablePan={false} />
    </Canvas>
  );
}
