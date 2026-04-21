# ocr2md Task Notes

## 当前范围

- 保留 `GLM-OCR` 本地识别能力。
- 删除已下线 OCR 后端的相关配置、API 契约和前端对比逻辑。
- WebUI 保留单模型下载、上传、任务轮询、识别结果展示。

## 验收点

- `GET /api/models/status` 返回单模型 `glm` 状态。
- `POST /api/models/glm/download` 可创建下载任务。
- `POST /api/ocr/jobs` 可创建异步 OCR 任务。
- `POST /api/ocr/run` 返回单模型 `glm` 结果。
- 前端按钮文案和页面布局不再出现“对比”语义。
- 任务详情和落盘结果只包含 `glm` 目录与结果。
