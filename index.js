import "./style.css";
import {
  AutoModelForMaskGeneration,
  AutoProcessor,
  RawImage,
  env,
} from "@huggingface/transformers";
import { extractMaskTensor, buildAutoBox, selectBestMask } from "./utils.js";

env.allowLocalModels = false;

const MODEL_ID = "onnx-community/sam2.1-hiera-small-ONNX";
const DEVICE = "gpu" in navigator ? "webgpu" : "wasm";
const POSITIVE_COLOR = "#22c55e";
const MASK_RGB = [0, 191, 255];

const statusEl = document.querySelector("#status");
const fileInputEl = document.querySelector("#file-input");
const defaultImageBtn = document.querySelector("#default-image-button");
const clearButtonEl = document.querySelector("#clear-button");
const canvasEl = document.querySelector("#canvas");
const resultCanvasEl = document.querySelector("#result-canvas");

const ctx = canvasEl.getContext("2d");
const resultCtx = resultCanvasEl.getContext("2d");

let processor = null;
let model = null;
let imageBitmap = null;
let rawImage = null;
let imageFeatures = null;
let points = [];
let currentMask = null;
let isBusy = false;
let isModelLoaded = false;

fileInputEl.addEventListener("change", async (event) => {
  const file = event.target.files?.[0];
  if (!file) return;
  await encodeImage(file);
});

defaultImageBtn.addEventListener("click", async () => {
  setBusy(true);
  setStatus("Fetching default image...");

  const response = await fetch("./default-image.jpg");
  if (!response.ok) throw new Error("Failed to fetch default image");

  const blob = await response.blob();
  const file = new File([blob], "default-image.jpg", { type: blob.type });

  await encodeImage(file);
});

clearButtonEl.addEventListener("click", clear);

canvasEl.addEventListener("click", async (event) => {
  await handleCanvasPoint(event);
});

function setStatus(message) {
  statusEl.textContent = message;
}

function setBusy(nextBusy) {
  isBusy = nextBusy;

  fileInputEl.disabled = nextBusy || !isModelLoaded;
  defaultImageBtn.disabled = nextBusy || !isModelLoaded;
  clearButtonEl.disabled = nextBusy || !imageBitmap;
}

function drawResultCanvas() {
  if (!imageBitmap) {
    resultCtx.clearRect(0, 0, resultCanvasEl.width, resultCanvasEl.height);
    return;
  }

  resultCanvasEl.width = imageBitmap.width;
  resultCanvasEl.height = imageBitmap.height;
  resultCtx.clearRect(0, 0, resultCanvasEl.width, resultCanvasEl.height);

  if (!currentMask) return;

  resultCtx.drawImage(imageBitmap, 0, 0);
  const frame = resultCtx.getImageData(
    0,
    0,
    resultCanvasEl.width,
    resultCanvasEl.height,
  );
  const pixels = frame.data;

  for (let i = 0; i < currentMask.data.length; i += 1) {
    if (currentMask.data[i] > 0) continue;
    const offset = i * 4;
    pixels[offset + 0] = 0;
    pixels[offset + 1] = 0;
    pixels[offset + 2] = 0;
    pixels[offset + 3] = 0;
  }
  resultCtx.putImageData(frame, 0, 0);
}

function drawCanvas() {
  if (!imageBitmap) {
    ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);
    drawResultCanvas();
    return;
  }

  canvasEl.width = imageBitmap.width;
  canvasEl.height = imageBitmap.height;
  ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);
  ctx.drawImage(imageBitmap, 0, 0);

  if (currentMask) {
    const frame = ctx.getImageData(0, 0, canvasEl.width, canvasEl.height);
    const pixels = frame.data;

    for (let i = 0; i < currentMask.data.length; i += 1) {
      if (currentMask.data[i] > 0) {
        const offset = i * 4;
        pixels[offset + 0] = pixels[offset + 0] * 0.42 + MASK_RGB[0] * 0.58;
        pixels[offset + 1] = pixels[offset + 1] * 0.42 + MASK_RGB[1] * 0.58;
        pixels[offset + 2] = pixels[offset + 2] * 0.42 + MASK_RGB[2] * 0.58;
        pixels[offset + 3] = 255;
      }
    }
    ctx.putImageData(frame, 0, 0);
  }

  for (const point of points) {
    ctx.beginPath();
    ctx.arc(point.x, point.y, 6, 0, Math.PI * 2);
    ctx.fillStyle = POSITIVE_COLOR;
    ctx.fill();
    ctx.lineWidth = 2;
    ctx.strokeStyle = "#ffffff";
    ctx.stroke();
  }
  drawResultCanvas();
}

function resetImageState() {
  imageBitmap = null;
  rawImage = null;
  imageFeatures = null;
  points = [];
  currentMask = null;
  drawCanvas();
}

function clear() {
  points = [];
  currentMask = null;
  drawCanvas();
  setStatus("Prompts cleared.");
}

function getCanvasPoint(event) {
  const rect = canvasEl.getBoundingClientRect();
  const scaleX = canvasEl.width / rect.width;
  const scaleY = canvasEl.height / rect.height;

  return {
    x: (event.clientX - rect.left) * scaleX,
    y: (event.clientY - rect.top) * scaleY,
  };
}

async function loadModel() {
  try {
    setStatus("Loading model...");
    const [loadedProcessor, loadedModel] = await Promise.all([
      AutoProcessor.from_pretrained(MODEL_ID),
      AutoModelForMaskGeneration.from_pretrained(MODEL_ID, {
        device: DEVICE,
        dtype: {
          vision_encoder: "fp32",
          prompt_encoder_mask_decoder: "fp32",
        },
      }),
    ]);

    processor = loadedProcessor;
    model = loadedModel;
    isModelLoaded = true;
    setBusy(false);
    setStatus(`Model ready (${DEVICE})`);
  } catch (error) {
    console.error(error);
    setStatus("Failed to load model. Check MODEL_ID and model repo format.");
  }
}

async function encodeImage(file) {
  if (!processor || !model) return;
  const objectUrl = URL.createObjectURL(file);

  try {
    setBusy(true);
    setStatus("Loading image...");
    resetImageState();

    imageBitmap = await createImageBitmap(file);
    drawCanvas();
    rawImage = await RawImage.read(objectUrl);

    setStatus("Encoding image...");
    const imageInputs = await processor(rawImage);
    imageFeatures = await model.get_image_embeddings(imageInputs);

    setStatus("Image encoded.");
  } catch (error) {
    console.error(error);
    resetImageState();
    setStatus("Failed to encode image.");
  } finally {
    URL.revokeObjectURL(objectUrl);
    setBusy(false);
  }
}

async function decodeMask(nextPoints) {
  if (
    !processor ||
    !model ||
    !rawImage ||
    !imageFeatures ||
    nextPoints.length === 0
  )
    return;

  try {
    setBusy(true);
    setStatus("Decoding mask...");

    const inputPoints = [
      [nextPoints.map((point) => [Math.round(point.x), Math.round(point.y)])],
    ];
    const inputLabels = [[nextPoints.map(() => 1)]];
    const autoBox = buildAutoBox(nextPoints, canvasEl.width, canvasEl.height);

    const processorInputs = {
      input_points: inputPoints,
      input_labels: inputLabels,
    };
    if (autoBox) processorInputs.input_boxes = [[autoBox]];

    const promptInputs = await processor(rawImage, processorInputs);
    const modelInputs = {
      ...imageFeatures,
      input_points: promptInputs.input_points,
      input_labels: promptInputs.input_labels,
      multimask_output: true,
    };
    if (promptInputs.input_boxes)
      modelInputs.input_boxes = promptInputs.input_boxes;

    const outputs = await model(modelInputs);
    const masks = await processor.post_process_masks(
      outputs.pred_masks,
      promptInputs.original_sizes,
      promptInputs.reshaped_input_sizes,
    );

    const maskTensor = extractMaskTensor(masks);
    if (!maskTensor) throw new Error("No mask tensor returned.");

    const bestMask = selectBestMask(maskTensor, nextPoints, outputs.iou_scores);
    if (!bestMask) throw new Error("No suitable mask candidate found.");

    currentMask = bestMask;
    drawCanvas();

    if (autoBox) {
      setStatus(
        "Mask updated. Auto box is helping the model keep the full person.",
      );
    } else {
      setStatus(
        "Mask updated. Add one more positive point on another body part for better full-body selection.",
      );
    }
  } catch (error) {
    console.error(error);
    setStatus("Failed to decode mask.");
  } finally {
    setBusy(false);
  }
}

async function handleCanvasPoint(event) {
  if (!imageBitmap || isBusy) return;

  const { x, y } = getCanvasPoint(event);
  const nextPoints = [...points, { x, y }];

  points = nextPoints;
  drawCanvas();
  await decodeMask(nextPoints);
}

setBusy(true);
loadModel();
