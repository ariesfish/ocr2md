import type { DownloadSource, ModelStatus, ModelsStatusResponse, TaskSnapshot } from "./types";

const API_BASE = import.meta.env.VITE_API_BASE ?? "/api";

async function parseJson(response: Response) {
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    const message = payload?.message ?? `HTTP ${response.status}`;
    throw new Error(message);
  }
  return payload;
}

export async function getModelsStatus(): Promise<ModelsStatusResponse> {
  const response = await fetch(`${API_BASE}/models/status`);
  return parseJson(response);
}

export async function checkModelDir(modelDir: string): Promise<ModelStatus> {
  const response = await fetch(`${API_BASE}/models/glm/set-dir`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model_dir: modelDir }),
  });
  return parseJson(response);
}

export async function startModelDownload(
  source?: DownloadSource,
  modelDir?: string,
): Promise<{ task_id: string; task_type: "download"; status: "queued" }> {
  const response = await fetch(`${API_BASE}/models/glm/download`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ source, model_dir: modelDir }),
  });
  return parseJson(response);
}

export async function createOCRJob(
  file: File,
  pageSelection?: string,
): Promise<{ task_id: string }> {
  const formData = new FormData();
  formData.append("file", file);
  if (pageSelection?.trim()) {
    formData.append("page_selection", pageSelection.trim());
  }

  const response = await fetch(`${API_BASE}/ocr/jobs`, {
    method: "POST",
    body: formData,
  });
  return parseJson(response);
}

export async function getTask(taskId: string): Promise<TaskSnapshot> {
  const response = await fetch(`${API_BASE}/tasks/${taskId}`);
  return parseJson(response);
}
