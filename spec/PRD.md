# ocr2md PRD

## 目标

- 提供本地 `GLM-OCR` 单模型识别能力。
- 固定主链路：`PageLoader -> LayoutDetector -> OCR Backend -> ResultFormatter`。
- WebUI 聚焦单次上传、单模型状态、单模型识别结果展示，不再提供模型对比能力。

## 主要流程

1. 用户上传一张图片。
2. 前端调用 `POST /api/ocr/jobs` 创建 OCR 任务。
3. 后端执行版面检测与 `GLM-OCR` 识别。
4. 前端轮询 `GET /api/tasks/{task_id}` 获取进度与结果。
5. 页面展示识别文本、定位框、耗时、平均置信度。

## API

- `GET /api/models/status`
- `POST /api/models/{model_key}/download`
- `POST /api/ocr/jobs`
- `POST /api/ocr/run`
- `GET /api/tasks/{task_id}`
- `GET /api/tasks/{task_id}/layout?page=0`

## 返回约定

- 模型状态接口仅返回 `glm`。
- OCR 结果接口仅返回 `glm` 结果对象。
- 任务结果中保留 `manifest`，用于读取落盘产物和版面可视化。

## 非目标

- 不支持多模型对比。
- 不支持远程 OCR API。
- 不支持关闭版面检测。
