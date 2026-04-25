import { useEffect, useRef, useState } from "react";
import * as Progress from "@radix-ui/react-progress";
import {
  Download,
  FileText,
  LoaderCircle,
  RefreshCcw,
  Upload,
} from "lucide-react";
import rehypeKatex from "rehype-katex";
import rehypeRaw from "rehype-raw";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";

import {
  checkModelDir,
  createOCRJob,
  getModelsStatus,
  getTask,
  startModelDownload,
} from "./api";
import type {
  DownloadSource,
  OCRTaskResult,
  OCRModelResult,
  ModelStatus,
  ModelsStatusResponse,
} from "./types";

type UIState = "idle" | "image_ready" | "processing" | "success" | "error";
type ModelRunState = "idle" | "running" | "success" | "error";
type ContentView = "rendered" | "raw";
type PreviewKind = "none" | "image" | "pdf";

const MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024;
const ACCEPTED_EXT = ["jpg", "jpeg", "png", "webp", "bmp", "gif", "pdf"];

const EMPTY_RESULT: OCRModelResult = {
  name: "OCR",
  latency_ms: null,
  confidence_avg: null,
  text: "",
  boxes: [],
  error: null,
};

const EMPTY_STATUS: Record<"glm", ModelStatus> = {
  glm: { name: "GLM-OCR", status: "not_ready", progress: 0 },
};

const STATUS_LABELS: Record<ModelStatus["status"], string> = {
  not_ready: "Not Ready",
  downloading: "Downloading",
  ready: "Ready",
  error: "Error",
};

const STATUS_BADGE_CLASSES: Record<ModelStatus["status"], string> = {
  not_ready: "bg-amber-50 text-amber-700 ring-amber-200",
  downloading: "bg-blue-50 text-blue-700 ring-blue-200",
  ready: "bg-emerald-50 text-emerald-700 ring-emerald-200",
  error: "bg-rose-50 text-rose-700 ring-rose-200",
};

type ImageSize = {
  width: number | null;
  height: number | null;
};

function getFileExtension(filename: string): string {
  const chunks = filename.toLowerCase().split(".");
  return chunks.length > 1 ? chunks[chunks.length - 1] : "";
}

function getCropPreviewUrl(taskId: string | null, rawSrc?: string | null): string | null {
  if (!taskId || !rawSrc) {
    return null;
  }

  const match = rawSrc.match(/^page=(\d+),bbox=\[([^\]]+)\]$/);
  if (!match) {
    return null;
  }

  const page = Number(match[1]);
  const bbox = match[2]
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean)
    .join(",");
  if (!Number.isFinite(page) || !bbox) {
    return null;
  }

  return `/api/tasks/${taskId}/crop?page=${page}&bbox=${encodeURIComponent(bbox)}`;
}

function getPdfEmbedUrl(url: string): string {
  return `${url}#toolbar=0&navpanes=0&scrollbar=0&view=FitH`;
}

function parsePageSelection(value: string): number[] | null {
  const cleaned = value.trim();
  if (!cleaned) {
    return null;
  }

  const pages: number[] = [];
  const seen = new Set<number>();
  for (const chunk of cleaned.split(",")) {
    const token = chunk.trim();
    if (!token) {
      continue;
    }

    if (token.includes("-")) {
      const [startRaw, endRaw] = token.split("-", 2).map((item) => item.trim());
      const start = Number(startRaw);
      const end = Number(endRaw);
      if (!Number.isInteger(start) || !Number.isInteger(end) || start <= 0 || end < start) {
        throw new Error(`无效页码范围: ${token}`);
      }
      for (let page = start; page <= end; page += 1) {
        if (!seen.has(page)) {
          pages.push(page);
          seen.add(page);
        }
      }
      continue;
    }

    const page = Number(token);
    if (!Number.isInteger(page) || page <= 0) {
      throw new Error(`无效页码: ${token}`);
    }
    if (!seen.has(page)) {
      pages.push(page);
      seen.add(page);
    }
  }

  if (pages.length === 0) {
    throw new Error("页码不能为空");
  }
  return pages;
}

function parseTaskPageProgress(message?: string | null): {
  currentPage: number;
  totalPages: number;
} | null {
  if (!message) {
    return null;
  }

  const match = message.match(/page=(\d+)\/(\d+)/);
  if (!match) {
    return null;
  }

  const currentPage = Number(match[1]);
  const totalPages = Number(match[2]);
  if (!Number.isInteger(currentPage) || !Number.isInteger(totalPages)) {
    return null;
  }
  if (currentPage <= 0 || totalPages <= 0 || currentPage > totalPages) {
    return null;
  }

  return { currentPage, totalPages };
}

function ModelCard({
  model,
  preview,
  previewKind,
  isPdfSelected,
  layoutPages,
  sourcePages,
  imageSize,
  runState,
  taskId,
  elapsedMs,
  currentPage,
  totalPages,
}: {
  model: OCRModelResult;
  preview: string | null;
  previewKind: PreviewKind;
  isPdfSelected: boolean;
  layoutPages: string[];
  sourcePages: number[];
  imageSize: ImageSize;
  runState: ModelRunState;
  taskId: string | null;
  elapsedMs: number | null;
  currentPage: number | null;
  totalPages: number;
}) {
  const [contentView, setContentView] = useState<ContentView>("rendered");
  const latencyText =
    elapsedMs == null ? "--" : `${(elapsedMs / 1000).toFixed(3)} 秒`;
  const pageProgressText =
    currentPage != null && totalPages > 0 ? `第 ${currentPage}/${totalPages} 页` : null;
  const imageWidth = typeof imageSize.width === "number" ? imageSize.width : null;
  const imageHeight = typeof imageSize.height === "number" ? imageSize.height : null;
  const imageRatio =
    imageWidth !== null && imageWidth > 0 && imageHeight !== null && imageHeight > 0
      ? imageWidth / imageHeight
      : null;
  const hasImageRatio = imageRatio !== null;
  const imageAspectRatio =
    imageWidth !== null && imageHeight !== null ? `${imageWidth} / ${imageHeight}` : "16 / 9";
  const mediaFrameClassName = hasImageRatio
    ? imageRatio !== null && imageRatio < 1
      ? "relative h-full max-w-full overflow-hidden rounded-xl border border-slate-200 bg-slate-100"
      : "relative w-full max-h-full overflow-hidden rounded-xl border border-slate-200 bg-slate-100"
    : "relative h-full w-full overflow-hidden rounded-xl border border-slate-200 bg-slate-100";
  const isRunning = runState === "running";
  const hasText = Boolean(model.text);
  const downloadUrl = taskId ? `/api/tasks/${taskId}/models/glm/download` : null;
  const content = model.error
    ? `错误: ${model.error}`
    : model.text || (isRunning ? "识别中..." : "暂无结果");

  return (
    <div className="rounded-2xl border border-slate-200 bg-white shadow-sm">
      <div className="flex items-center justify-between border-b border-slate-100 px-5 py-4">
        <div className="font-semibold text-slate-800">OCR</div>
        <div className="flex items-center gap-2 text-xs text-slate-500">
          <span>总耗时 {latencyText}</span>
          {pageProgressText ? (
            <>
              <span>·</span>
              <span>{pageProgressText}</span>
            </>
          ) : null}
        </div>
      </div>

      <div className="grid gap-5 p-5 lg:grid-cols-2">
        <div className="flex min-h-[22rem] flex-col overflow-hidden rounded-2xl border border-slate-200 bg-slate-50 lg:h-[calc(100vh-21rem)] lg:min-h-[36rem] lg:max-h-[52rem]">
          <div className="flex h-14 items-center border-b border-slate-200 bg-white px-4">
            <span className="text-sm font-semibold text-slate-700">内容定位</span>
          </div>
          <div className="min-h-0 flex-1 overflow-auto p-3">
            {layoutPages.length > 0 ? (
              <div className="flex min-h-full flex-col gap-3">
                {layoutPages.map((pageUrl, index) => (
                  <div
                    key={pageUrl}
                    className="overflow-hidden rounded-xl border border-slate-200 bg-white"
                  >
                    <div className="border-b border-slate-100 px-3 py-2 text-xs font-medium text-slate-500">
                      第 {index + 1} 页
                      {isPdfSelected && sourcePages[index] ? ` / 原 PDF 第 ${sourcePages[index]} 页` : ""}
                    </div>
                    <img src={pageUrl} alt={`layout page ${index + 1}`} className="w-full" />
                  </div>
                ))}
              </div>
            ) : preview && previewKind === "image" ? (
              <div
                className={mediaFrameClassName}
                style={{
                  aspectRatio: imageAspectRatio,
                }}
              >
                <img src={preview} alt="preview" className="h-full w-full object-contain" />
                {isRunning ? (
                  <div className="pointer-events-none absolute left-1/2 top-3 z-10 -translate-x-1/2 rounded-full bg-white/85 px-4 py-2 shadow-sm ring-1 ring-slate-200">
                    <div className="flex items-center gap-2 text-base font-semibold text-slate-700">
                      <LoaderCircle className="animate-spin text-blue-600" size={18} />
                      <span>识别中</span>
                    </div>
                  </div>
                ) : null}
              </div>
            ) : (
              <div className="flex h-full min-h-[28rem] items-center justify-center rounded-xl border border-slate-200 bg-white text-sm text-slate-400">
                {previewKind === "pdf"
                  ? "识别后将在此逐页显示定位结果"
                  : "上传文件后在此查看定位结果"}
              </div>
            )}
          </div>
        </div>

        <div className="flex min-h-[22rem] min-w-0 flex-col overflow-hidden rounded-2xl border border-slate-200 bg-slate-50 lg:h-[calc(100vh-21rem)] lg:min-h-[36rem] lg:max-h-[52rem]">
          <div className="flex h-14 items-center justify-between gap-3 border-b border-slate-200 bg-white px-4">
            <span className="text-sm font-semibold text-slate-700">识别结果</span>
            <div className="flex shrink-0 items-center gap-2">
              <div className="inline-flex rounded-lg border border-slate-200 bg-slate-50 p-0.5 text-xs">
                <button
                  type="button"
                  className={`rounded-md px-2.5 py-1 transition ${contentView === "rendered" ? "bg-white text-slate-900 shadow-sm" : "text-slate-500 hover:text-slate-700"}`}
                  onClick={() => setContentView("rendered")}
                >
                  预览
                </button>
                <button
                  type="button"
                  className={`rounded-md px-2.5 py-1 transition ${contentView === "raw" ? "bg-white text-slate-900 shadow-sm" : "text-slate-500 hover:text-slate-700"}`}
                  onClick={() => setContentView("raw")}
                >
                  原文
                </button>
              </div>
              <button
                type="button"
                className="inline-flex items-center gap-1 rounded-md p-1.5 text-slate-500 hover:bg-slate-100 disabled:cursor-not-allowed disabled:opacity-40"
                disabled={!hasText || !downloadUrl}
                title="下载 Markdown"
                onClick={() => {
                  if (!downloadUrl) {
                    return;
                  }
                  window.location.href = downloadUrl;
                }}
              >
                <Download size={13} />
              </button>
            </div>
          </div>
          <div className="min-h-0 flex-1 overflow-auto bg-white px-5 py-4 text-sm leading-7">
            {contentView === "rendered" && !model.error && hasText ? (
              <div className="markdown-body">
                <ReactMarkdown
                  rehypePlugins={[rehypeRaw, rehypeKatex]}
                  remarkPlugins={[remarkGfm, remarkMath]}
                  components={{
                    img: ({ src = "", alt = "" }) => {
                      const cropSrc = getCropPreviewUrl(taskId, src);
                      const resolvedSrc = cropSrc ?? src;
                      return (
                        <img
                          src={resolvedSrc}
                          alt={alt}
                          className="my-4 max-w-full rounded-lg border border-slate-200"
                        />
                      );
                    },
                  }}
                >
                  {model.text}
                </ReactMarkdown>
              </div>
            ) : (
              <pre className="whitespace-pre-wrap break-words font-sans text-sm leading-7 text-slate-700">
                {content}
              </pre>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default function App() {
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const [uiState, setUiState] = useState<UIState>("idle");
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [previewKind, setPreviewKind] = useState<PreviewKind>("none");
  const [errorMessage, setErrorMessage] = useState<string>("");

  const [models, setModels] = useState<Record<"glm", ModelStatus>>(EMPTY_STATUS);
  const [result, setResult] = useState<OCRModelResult>(EMPTY_RESULT);
  const [modelRunState, setModelRunState] = useState<ModelRunState>("idle");
  const [imageSize, setImageSize] = useState<ImageSize>({ width: null, height: null });
  const [layoutPageUrls, setLayoutPageUrls] = useState<string[]>([]);
  const [pageCount, setPageCount] = useState(1);
  const [sourcePages, setSourcePages] = useState<number[]>([]);
  const [currentPage, setCurrentPage] = useState<number | null>(null);
  const [ocrTaskId, setOCRTaskId] = useState<string | null>(null);
  const [resultTaskId, setResultTaskId] = useState<string | null>(null);
  const [ocrStartedAtMs, setOcrStartedAtMs] = useState<number | null>(null);
  const [displayElapsedMs, setDisplayElapsedMs] = useState<number | null>(null);
  const [downloadTaskId, setDownloadTaskId] = useState<string | null>(null);
  const [downloadSource, setDownloadSource] = useState<DownloadSource>("modelscope");
  const [modelDir, setModelDir] = useState<string>("");
  const [modelDirTouched, setModelDirTouched] = useState(false);
  const [pageSelection, setPageSelection] = useState<string>("");
  const [isCheckingModelDir, setIsCheckingModelDir] = useState(false);
  const selectedExt = file ? getFileExtension(file.name) : "";
  const isPdfSelected = selectedExt === "pdf";
  const defaultModelDir = models.glm.model_root_dir?.trim() ?? "";

  function resetLayoutPages() {
    setLayoutPageUrls((prev) => {
      prev.forEach((url) => URL.revokeObjectURL(url));
      return [];
    });
  }

  function replacePreview(nextPreview: string | null, nextKind: PreviewKind) {
    setPreview((prev) => {
      if (prev && prev !== nextPreview) {
        URL.revokeObjectURL(prev);
      }
      return nextPreview;
    });
    setPreviewKind(nextKind);
  }

  useEffect(() => {
    getModelsStatus()
      .then((payload: ModelsStatusResponse) => setModels(payload.models))
      .catch((err: Error) => setErrorMessage(err.message));
  }, []);

  useEffect(() => {
    if (modelDirTouched) {
      return;
    }
    if (!defaultModelDir) {
      return;
    }
    if (modelDir === defaultModelDir) {
      return;
    }
    setModelDir(defaultModelDir);
  }, [defaultModelDir, modelDir, modelDirTouched]);

  useEffect(() => {
    if (modelRunState !== "running" || ocrStartedAtMs == null) {
      return;
    }

    const updateElapsed = () => {
      setDisplayElapsedMs(Math.max(0, Date.now() - ocrStartedAtMs));
    };

    updateElapsed();
    const timer = window.setInterval(updateElapsed, 250);
    return () => window.clearInterval(timer);
  }, [modelRunState, ocrStartedAtMs]);

  useEffect(() => {
    return () => {
      if (preview) {
        URL.revokeObjectURL(preview);
      }
      layoutPageUrls.forEach((url) => URL.revokeObjectURL(url));
    };
  }, [layoutPageUrls, preview]);

  useEffect(() => {
    if (!ocrTaskId && !downloadTaskId) {
      return;
    }

    const timer = window.setInterval(async () => {
      if (ocrTaskId) {
        try {
          const task = await getTask(ocrTaskId);
          const payload = task.result as OCRTaskResult | undefined;
          const pageProgress = parseTaskPageProgress(task.message);

          if (pageProgress) {
            setCurrentPage(pageProgress.currentPage);
          }

          if (payload?.input) {
            setImageSize({
              width: payload.input.width ?? null,
              height: payload.input.height ?? null,
            });
            setPageCount(payload.input.page_count ?? 1);
            setSourcePages(payload.input.source_pages ?? []);
          }

          if (payload?.glm) {
            setResult(payload.glm);
            setModelRunState(
              payload.glm.error
                ? "error"
                : task.status === "succeeded"
                  ? "success"
                  : "running",
            );
          }

          if (task.status === "failed") {
            setUiState("error");
            setErrorMessage(task.error?.message ?? task.message ?? "OCR 失败");
            setModelRunState("error");
            setOcrStartedAtMs(null);
            setOCRTaskId(null);
            return;
          }

          if (task.status === "succeeded") {
            const finalModel = payload?.glm;
            if (finalModel) {
              setResult(finalModel);
              setDisplayElapsedMs((prev) => finalModel.latency_ms ?? prev);
            }
            setCurrentPage((prev) => payload?.input?.page_count ?? pageProgress?.totalPages ?? prev);
            setModelRunState("success");
            setUiState("success");
            setOcrStartedAtMs(null);
            setOCRTaskId(null);
            return;
          }

          if (task.status === "running" || task.status === "queued") {
            setUiState("processing");
          }
        } catch (err) {
          const message = err instanceof Error ? err.message : "轮询失败";
          setUiState("error");
          setErrorMessage(message);
          setModelRunState("error");
          setOcrStartedAtMs(null);
          setOCRTaskId(null);
        }
      }

      if (downloadTaskId) {
        try {
          const task = await getTask(downloadTaskId);
          if (task.status === "succeeded" || task.status === "failed") {
            const statusPayload = await getModelsStatus();
            setModels(statusPayload.models);
            setDownloadTaskId(null);
          } else {
            setModels((prev) => ({
              glm: {
                ...prev.glm,
                status: "downloading",
                progress: task.progress,
                message: task.message,
              },
            }));
          }
        } catch (err) {
          const message = err instanceof Error ? err.message : "下载任务查询失败";
          setErrorMessage(message);
          setDownloadTaskId(null);
        }
      }
    }, 1500);

    return () => window.clearInterval(timer);
  }, [downloadTaskId, ocrTaskId]);

  useEffect(() => {
    if (!resultTaskId || pageCount <= layoutPageUrls.length) {
      return;
    }

    let cancelled = false;
    const timer = window.setInterval(async () => {
      const nextPage = layoutPageUrls.length;
      try {
        const response = await fetch(`/api/tasks/${resultTaskId}/layout?page=${nextPage}`);
        if (!response.ok) {
          return;
        }

        const blob = await response.blob();
        const nextUrl = URL.createObjectURL(blob);
        if (cancelled) {
          URL.revokeObjectURL(nextUrl);
          return;
        }

        setLayoutPageUrls((prev) => {
          if (prev.length !== nextPage) {
            URL.revokeObjectURL(nextUrl);
            return prev;
          }
          return [...prev, nextUrl];
        });
      } catch {
        return;
      }
    }, 1000);

    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [layoutPageUrls.length, pageCount, resultTaskId]);

  function updatePreviewImageSize(objectUrl: string) {
    const image = new Image();
    image.onload = () => {
      setImageSize({ width: image.naturalWidth, height: image.naturalHeight });
    };
    image.onerror = () => {
      setImageSize({ width: null, height: null });
    };
    image.src = objectUrl;
  }

  function openFilePicker() {
    fileInputRef.current?.click();
  }

  function onFileSelected(nextFile: File) {
    const ext = getFileExtension(nextFile.name);
    if (!ACCEPTED_EXT.includes(ext)) {
      setErrorMessage(`不支持文件类型: ${ext}`);
      setUiState("error");
      return;
    }
    if (nextFile.size > MAX_FILE_SIZE_BYTES) {
      setErrorMessage("文件超过 10MB 限制");
      setUiState("error");
      return;
    }

    setFile(nextFile);
    const objectUrl = URL.createObjectURL(nextFile);
    if (ext === "pdf") {
      replacePreview(objectUrl, "pdf");
      setImageSize({ width: null, height: null });
    } else {
      replacePreview(objectUrl, "image");
      updatePreviewImageSize(objectUrl);
    }
    resetLayoutPages();
    setPageCount(1);
    setSourcePages([]);
    setCurrentPage(null);
    setPageSelection(ext === "pdf" ? pageSelection : "");
    setResultTaskId(null);
    setOcrStartedAtMs(null);
    setDisplayElapsedMs(null);
    setErrorMessage("");
    setResult(EMPTY_RESULT);
    setModelRunState("idle");
    setUiState("image_ready");
  }

  async function startOCR() {
    if (!file || uiState === "processing") {
      return;
    }
    try {
      const selectedPages = isPdfSelected ? parsePageSelection(pageSelection) : null;
      setUiState("processing");
      setErrorMessage("");
      resetLayoutPages();
      setPageCount(selectedPages?.length ?? 1);
      setSourcePages(selectedPages ?? []);
      setCurrentPage(1);
      setResult(EMPTY_RESULT);
      setModelRunState("running");
      setOcrStartedAtMs(Date.now());
      setDisplayElapsedMs(0);
      const { task_id } = await createOCRJob(file, pageSelection);
      setOCRTaskId(task_id);
      setResultTaskId(task_id);
    } catch (err) {
      const message = err instanceof Error ? err.message : "创建任务失败";
      setUiState("error");
      setErrorMessage(message);
      setModelRunState("idle");
      setCurrentPage(null);
      setOcrStartedAtMs(null);
      setDisplayElapsedMs(null);
    }
  }

  async function triggerDownload() {
    try {
      const resolvedModelDir = modelDir.trim() || defaultModelDir || undefined;
      const { task_id } = await startModelDownload(downloadSource, resolvedModelDir);
      setDownloadTaskId(task_id);
      setModels((prev) => ({
        glm: { ...prev.glm, status: "downloading", progress: 0 },
      }));
    } catch (err) {
      const message = err instanceof Error ? err.message : "下载触发失败";
      setErrorMessage(message);
    }
  }

  async function applyModelDir() {
    const resolvedModelDir = modelDir.trim() || defaultModelDir;
    if (!resolvedModelDir) {
      setErrorMessage("请先填写模型目录");
      return;
    }

    try {
      setIsCheckingModelDir(true);
      setErrorMessage("");
      const nextStatus = await checkModelDir(resolvedModelDir);
      setModels({ glm: nextStatus });
    } catch (err) {
      const message = err instanceof Error ? err.message : "模型目录检查失败";
      setErrorMessage(message);
    } finally {
      setIsCheckingModelDir(false);
    }
  }

  const status = models.glm;
  const downloading = status.status === "downloading";

  return (
    <div className="mx-auto min-h-screen max-w-[1440px] space-y-8 px-4 py-6 xl:px-6">
      <header className="space-y-4">
        <h1 className="text-3xl font-black text-slate-800">OCR2MD</h1>

        <div className="grid gap-4 lg:grid-cols-3 lg:auto-rows-fr">
          <div className="group relative flex min-h-[240px] items-center justify-center overflow-hidden rounded-2xl border-2 border-dashed border-slate-300 bg-white lg:h-full">
            <input
              ref={fileInputRef}
              type="file"
              className="hidden"
              accept="image/*,.pdf,application/pdf"
              onChange={(event) => {
                const nextFile = event.target.files?.[0];
                if (nextFile) {
                  onFileSelected(nextFile);
                }
              }}
            />
            {preview ? (
              previewKind === "pdf" ? (
                <iframe
                  src={getPdfEmbedUrl(preview)}
                  title="upload preview"
                  className="h-full w-full bg-white"
                />
              ) : (
                <img src={preview} alt="preview" className="h-full w-full object-contain p-2" />
              )
            ) : (
              <button type="button" onClick={openFilePicker} className="px-6 text-center">
                {file && isPdfSelected ? (
                  <>
                    <FileText className="mx-auto mb-2 text-slate-500" />
                    <p className="text-sm font-semibold text-slate-700">{file.name}</p>
                    <p className="mt-1 text-xs text-slate-500">PDF 已加载，可直接开始识别</p>
                  </>
                ) : (
                  <>
                    <Upload className="mx-auto mb-2 text-slate-500" />
                    <p className="text-sm font-semibold text-slate-700">上传图片或 PDF</p>
                  </>
                )}
              </button>
            )}
            <button
              type="button"
              onClick={openFilePicker}
              className="absolute bottom-3 right-3 rounded-lg bg-white/90 px-3 py-2 text-xs font-medium text-slate-700 shadow-sm ring-1 ring-slate-200 backdrop-blur hover:bg-white"
            >
              {file ? "更换文件" : "选择文件"}
            </button>
          </div>

          <div className="space-y-3 rounded-2xl border border-slate-200 bg-white p-4 lg:col-span-2">
            <div className="flex items-center justify-between">
              <div>
                <div className="font-semibold text-slate-800">设置</div>
              </div>
              <button
                type="button"
                onClick={startOCR}
                disabled={uiState === "processing" || !file}
                className="inline-flex items-center gap-2 rounded-lg bg-blue-600 px-3 py-2 text-xs font-semibold text-white disabled:cursor-not-allowed disabled:bg-slate-400"
              >
                {uiState === "processing" ? (
                  <LoaderCircle className="animate-spin" size={14} />
                ) : (
                  <RefreshCcw size={14} />
                )}
                开始识别
              </button>
            </div>

            <div className="grid gap-3 xl:grid-cols-[1.1fr_1fr]">
              <section className="rounded-xl border border-slate-200 bg-slate-50/60 p-4">
                <div className="mb-3 flex items-center justify-between">
                  <div>
                    <div className="text-sm font-semibold text-slate-800">解析内容</div>
                    <div className="mt-1 text-xs text-slate-500">设置当前文件的解析范围</div>
                  </div>
                </div>

                <div className="rounded-lg border border-slate-200 bg-white px-3 py-2.5">
                  <div className="text-[11px] font-medium uppercase tracking-wide text-slate-400">
                    当前文件
                  </div>
                  <div className="mt-1 truncate text-sm font-medium text-slate-700">
                    {file ? file.name : "尚未选择文件"}
                  </div>
                  <div className="mt-1 text-xs text-slate-500">
                    {file
                      ? `${(file.size / 1024 / 1024).toFixed(2)} MB · ${isPdfSelected ? "PDF" : "图片"}`
                      : "支持 jpg、png、webp、bmp、gif、pdf"}
                  </div>
                </div>

                {isPdfSelected ? (
                  <label className="mt-3 block">
                    <span className="mb-1.5 block text-xs font-medium text-slate-600">PDF 页码范围</span>
                    <input
                      type="text"
                      value={pageSelection}
                      onChange={(event) => setPageSelection(event.target.value)}
                      placeholder="留空解析全部，或填 1,3-5"
                      className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm outline-none transition focus:border-blue-500"
                    />
                    <div className="mt-1.5 text-xs text-slate-500">支持单页、范围和混合输入，例如 `2`、`3-8`、`1,4,7-9`</div>
                  </label>
                ) : (
                  <div className="mt-3 rounded-lg border border-dashed border-slate-200 bg-white px-3 py-2.5 text-xs text-slate-500">
                    图片文件默认解析整张图片，无需额外分页设置。
                  </div>
                )}
              </section>

              <section className="rounded-xl border border-slate-200 bg-slate-50/60 p-4">
                <div className="mb-3 flex items-center justify-between gap-3">
                  <div>
                    <div className="text-sm font-semibold text-slate-800">模型</div>
                    <div className="mt-1 text-xs text-slate-500">下载模型，检查模型目录</div>
                  </div>
                  <span
                    className={`inline-flex items-center rounded-full px-2.5 py-1 text-[11px] font-medium ring-1 ${STATUS_BADGE_CLASSES[status.status]}`}
                  >
                    {STATUS_LABELS[status.status]}
                  </span>
                </div>

                <div className="rounded-lg border border-slate-200 bg-white p-3">
                  <div className="mb-2 flex items-center justify-between gap-3 text-sm">
                    <span className="font-medium text-slate-700">{status.name}</span>
                    <span className="text-xs text-slate-500">{status.progress}%</span>
                  </div>
                  <Progress.Root className="relative h-2 w-full overflow-hidden rounded-full bg-slate-200">
                    <Progress.Indicator
                      className="h-full bg-blue-600 transition-all"
                      style={{ width: `${status.progress}%` }}
                    />
                  </Progress.Root>
                  {status.message ? (
                    <div className="mt-2 text-xs text-slate-500">{status.message}</div>
                  ) : null}
                </div>

                <label className="mt-3 block">
                  <span className="mb-1.5 block text-xs font-medium text-slate-600">模型目录</span>
                  <input
                    type="text"
                    value={modelDir}
                    onChange={(event) => {
                      const nextValue = event.target.value;
                      if (!nextValue.trim() && defaultModelDir) {
                        setModelDir(defaultModelDir);
                        setModelDirTouched(false);
                        return;
                      }
                      setModelDir(nextValue);
                      setModelDirTouched(nextValue.trim() !== defaultModelDir);
                    }}
                    placeholder="填写模型根目录，例如 /path/to/models"
                    className={`w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm outline-none transition focus:border-blue-500 ${modelDirTouched ? "text-slate-700" : "text-slate-400"}`}
                  />
                </label>

                <label className="mt-3 block">
                  <span className="mb-1.5 block text-xs font-medium text-slate-600">下载源</span>
                  <select
                    value={downloadSource}
                    onChange={(event) => setDownloadSource(event.target.value as DownloadSource)}
                    disabled={downloading}
                    className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm outline-none transition focus:border-blue-500 disabled:cursor-not-allowed disabled:opacity-40"
                  >
                    <option value="modelscope">ModelScope</option>
                    <option value="huggingface">HuggingFace</option>
                  </select>
                </label>

                <div className="mt-3 flex flex-wrap gap-2">
                  <button
                    type="button"
                    onClick={applyModelDir}
                    disabled={isCheckingModelDir}
                    className="inline-flex items-center justify-center gap-2 rounded-lg border border-slate-300 bg-white px-3 py-2 text-xs font-medium text-slate-700 disabled:cursor-not-allowed disabled:opacity-40"
                  >
                    {isCheckingModelDir ? <LoaderCircle className="animate-spin" size={14} /> : null}
                    <span>检查目录</span>
                  </button>
                  <button
                    type="button"
                    disabled={downloading}
                    onClick={triggerDownload}
                    className="inline-flex items-center justify-center gap-2 rounded-lg border border-slate-300 bg-white px-3 py-2 text-xs font-medium text-slate-700 disabled:cursor-not-allowed disabled:opacity-40"
                  >
                    <Download size={12} />
                    <span>下载模型</span>
                  </button>
                </div>
              </section>
            </div>
          </div>
        </div>

        {errorMessage ? (
          <div className="rounded-lg bg-rose-50 p-3 text-sm text-rose-700">{errorMessage}</div>
        ) : null}
      </header>

      <section>
        <ModelCard
          model={result}
          preview={preview}
          previewKind={previewKind}
          isPdfSelected={isPdfSelected}
          layoutPages={layoutPageUrls}
          sourcePages={sourcePages}
          imageSize={imageSize}
          runState={modelRunState}
          taskId={resultTaskId}
          elapsedMs={displayElapsedMs ?? result.latency_ms}
          currentPage={currentPage}
          totalPages={pageCount}
        />
      </section>
    </div>
  );
}
