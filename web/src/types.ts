export type TaskStatus = "queued" | "running" | "succeeded" | "failed" | "expired";

export type DownloadSource = "modelscope" | "huggingface";

export type OCRBox = {
  index: number;
  label: string;
  score: number | null;
  bbox_2d: [number, number, number, number];
  polygon?: number[][] | null;
};

export type OCRModelResult = {
  name: string;
  latency_ms: number | null;
  confidence_avg: number | null;
  text: string;
  boxes: OCRBox[];
  error?: string | null;
};

export type OCRInputMeta = {
  filename: string;
  width: number | null;
  height: number | null;
  page_count?: number | null;
  source_pages?: number[] | null;
};

export type OCRResult = {
  request_id: string;
  input: OCRInputMeta;
  glm: OCRModelResult;
  manifest?: string | null;
};

export type OCRTaskResult = {
  request_id: string;
  input?: OCRInputMeta;
  glm?: OCRModelResult;
  manifest?: string | null;
};

export type ModelStatus = {
  name: string;
  status: "not_ready" | "downloading" | "ready" | "error";
  progress: number;
  message?: string | null;
  model_root_dir?: string | null;
};

export type ModelsStatusResponse = {
  models: Record<"glm", ModelStatus>;
};

export type TaskSnapshot = {
  task_id: string;
  task_type: "ocr" | "download";
  status: TaskStatus;
  progress: number;
  stage?: string | null;
  message?: string | null;
  model_key?: "glm";
  result?: OCRTaskResult;
  error?: {
    code: string;
    message: string;
    request_id: string;
  } | null;
};

export type ApiError = {
  code: string;
  message: string;
  request_id: string;
};
