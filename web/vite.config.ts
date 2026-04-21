import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    host: "0.0.0.0",
    port: 8844,
    strictPort: true,
    allowedHosts: ["instance-v2fos0k7.pvt.hz.smartml.cn"],
    proxy: {
      "/api": "http://localhost:8000",
    },
  },
});
