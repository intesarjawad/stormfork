/** A decoded one-channel atlas: bytes as the exporter quantised them. */
export interface DecodedAtlas {
  readonly width: number;
  readonly height: number;
  readonly bytes: Uint8Array;
}

export async function decodeAtlas(url: string): Promise<DecodedAtlas> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`${url} responded ${String(response.status)}`);
  }
  const bitmap = await createImageBitmap(await response.blob(), { colorSpaceConversion: "none" });
  const canvas = document.createElement("canvas");
  canvas.width = bitmap.width;
  canvas.height = bitmap.height;
  const context = canvas.getContext("2d", { willReadFrequently: true });
  if (context === null) {
    throw new Error("a 2D canvas is not available to read the pressure atlas");
  }
  context.drawImage(bitmap, 0, 0);
  const rgba = context.getImageData(0, 0, bitmap.width, bitmap.height).data;
  const bytes = new Uint8Array(bitmap.width * bitmap.height);
  for (let index = 0; index < bytes.length; index += 1) {
    bytes[index] = rgba[index * 4] ?? 0;
  }
  return { width: bitmap.width, height: bitmap.height, bytes };
}
