export function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function extractFirstTensor(value) {
  if (!value) return null;
  if (value.data && value.dims) return value;
  if (Array.isArray(value)) return extractFirstTensor(value[0]);
  return null;
}

export function extractMaskTensor(value) {
  return extractFirstTensor(value);
}

export function extractScores(scoresTensor, count) {
  if (!scoresTensor?.data) return new Array(count).fill(0);

  const scores = Array.from(scoresTensor.data);
  if (scores.length >= count) return scores.slice(scores.length - count);

  const padded = new Array(count).fill(0);
  for (let i = 0; i < scores.length; i += 1) {
    padded[i] = scores[i];
  }
  return padded;
}

export function getMaskMeta(tensor) {
  const dims = tensor.dims;
  const height = dims[dims.length - 2];
  const width = dims[dims.length - 1];
  let maskCount = 1;

  if (dims.length >= 3) maskCount = dims[dims.length - 3];
  return { width, height, maskCount };
}

export function getMaskOffset(maskIndex, width, height) {
  return maskIndex * width * height;
}

export function sampleMaskValue(maskData, width, height, x, y) {
  const px = clamp(Math.round(x), 0, width - 1);
  const py = clamp(Math.round(y), 0, height - 1);
  return maskData[py * width + px] > 0 ? 1 : 0;
}

export function countMaskArea(maskData) {
  let area = 0;
  for (let i = 0; i < maskData.length; i += 1) {
    if (maskData[i] > 0) area += 1;
  }
  return area;
}

export function buildAutoBox(nextPoints, width, height) {
  if (nextPoints.length < 2) return null;

  let minX = Number.POSITIVE_INFINITY;
  let minY = Number.POSITIVE_INFINITY;
  let maxX = Number.NEGATIVE_INFINITY;
  let maxY = Number.NEGATIVE_INFINITY;

  for (const point of nextPoints) {
    minX = Math.min(minX, point.x);
    minY = Math.min(minY, point.y);
    maxX = Math.max(maxX, point.x);
    maxY = Math.max(maxY, point.y);
  }

  const spanX = Math.max(1, maxX - minX);
  const spanY = Math.max(1, maxY - minY);
  const spanMax = Math.max(spanX, spanY);
  const padding = Math.max(32, spanMax * 0.35);

  const x1 = clamp(Math.round(minX - padding), 0, width - 1);
  const y1 = clamp(Math.round(minY - padding), 0, height - 1);
  const x2 = clamp(Math.round(maxX + padding), 0, width - 1);
  const y2 = clamp(Math.round(maxY + padding), 0, height - 1);

  if (x2 <= x1 || y2 <= y1) return null;
  return [x1, y1, x2, y2];
}

export function selectBestMask(maskTensor, pointsForScoring, iouScores) {
  const { width, height, maskCount } = getMaskMeta(maskTensor);
  const scores = extractScores(iouScores, maskCount);
  const planeSize = width * height;

  let bestMask = null;
  let bestScore = Number.NEGATIVE_INFINITY;

  for (let maskIndex = 0; maskIndex < maskCount; maskIndex += 1) {
    const offset = getMaskOffset(maskIndex, width, height);
    const maskData = maskTensor.data.slice(offset, offset + planeSize);
    const area = countMaskArea(maskData);
    const areaRatio = area / planeSize;

    let positiveHits = 0;

    for (const point of pointsForScoring) {
      const hit = sampleMaskValue(maskData, width, height, point.x, point.y);
      positiveHits += hit;
    }

    let heuristicScore = 0;
    heuristicScore += (scores[maskIndex] ?? 0) * 5;
    heuristicScore += positiveHits * 3;

    if (areaRatio < 0.003) heuristicScore -= 3;
    if (areaRatio > 0.9) heuristicScore -= 2;

    if (heuristicScore > bestScore) {
      bestScore = heuristicScore;
      bestMask = { data: maskData, width, height };
    }
  }

  return bestMask;
}
