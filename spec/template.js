import React, { useState, useEffect } from 'react';
import { 
  Upload, Zap, FileText, CheckCircle2, RefreshCcw, 
  Scan, Download, HardDrive, Copy 
} from 'lucide-react';

const App = () => {
  const [image, setImage] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  
  // 模型数据状态
  const [results, setResults] = useState({
    paddle: { name: 'PaddleOCR-VL-1.5', latency: '--', confidence: '--', text: '', boxes: [] },
    glm: { name: 'GLM-OCR', latency: '--', confidence: '--', text: '', boxes: [] }
  });

  // 模型管理状态（实际应用中应从后端查询已下载状态）
  const [downloading, setDownloading] = useState({
    paddle: { progress: 100, active: false, status: 'completed' },
    glm: { progress: 0, active: false, status: 'idle' }
  });

  // --- 接口接入预留区 ---
  const fetchOCRResults = async (imageData) => {
    setIsProcessing(true);
    try {
      // TODO: 在此处调用后端 API
      // const response = await fetch('/api/ocr', { method: 'POST', body: JSON.stringify({ image: imageData }) });
      // const data = await response.json();
      
      // 成功后的状态更新
      // setResults({ paddle: data.paddle, glm: data.glm });
      
      console.log("正在请求后端接口...");
      // 模拟网络延迟
      await new Promise(resolve => setTimeout(resolve, 1500));
    } catch (error) {
      console.error("OCR 请求失败:", error);
    } finally {
      setIsProcessing(false);
    }
  };

  const handleUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        const base64Image = e.target.result;
        setImage(base64Image);
        fetchOCRResults(base64Image); // 触发识别
      };
      reader.readAsDataURL(file);
    }
  };

  const startDownload = (modelKey) => {
    // 实际应用中此处应发送指令给本地服务开始下载
    setDownloading(prev => ({ 
      ...prev, 
      [modelKey]: { ...prev[modelKey], active: true, status: 'downloading', progress: 0 } 
    }));
  };

  // 模拟下载进度（生产环境请通过 SSE 或轮询获取真实进度）
  useEffect(() => {
    const timer = setInterval(() => {
      setDownloading(prev => {
        const next = { ...prev };
        let changed = false;
        ['paddle', 'glm'].forEach(key => {
          if (next[key].active && next[key].progress < 100) {
            next[key].progress += 10;
            if (next[key].progress >= 100) {
              next[key].status = 'completed';
              next[key].active = false;
            }
            changed = true;
          }
        });
        return changed ? next : prev;
      });
    }, 500);
    return () => clearInterval(timer);
  }, []);

  const ModelCard = ({ result, modelKey, side }) => {
    const isRight = side === 'right';
    return (
      <div className={`flex flex-col bg-white rounded-2xl border-2 transition-all ${isRight ? 'border-blue-100 shadow-sm' : 'border-slate-100'} overflow-hidden h-full`}>
        <div className={`px-4 py-3 border-b flex items-center justify-between ${isRight ? 'bg-blue-50/30' : 'bg-slate-50/30'}`}>
          <div className="flex items-center gap-2">
            <div className={`w-2 h-2 rounded-full ${isRight ? 'bg-blue-600' : 'bg-indigo-600'}`}></div>
            <span className="font-bold text-slate-800">{result.name}</span>
          </div>
          <div className="flex gap-3 text-[11px] font-mono font-bold">
            <span className="text-slate-400 text-xs">耗时: <span className="text-slate-700">{result.latency}</span></span>
            <span className="text-slate-400 text-xs">置信度: <span className="text-green-600">{result.confidence}</span></span>
          </div>
        </div>
        <div className="flex-grow p-4 space-y-6 overflow-y-auto">
          <div>
            <div className="flex items-center gap-2 text-[10px] font-black text-slate-400 uppercase mb-2 tracking-widest"><Scan size={12} /> 定位</div>
            <div className="relative aspect-video bg-slate-100 rounded-xl border overflow-hidden flex items-center justify-center">
              {image ? (
                <>
                  <img src={image} className="max-w-full max-h-full opacity-40 object-contain" alt="Preview" />
                  {result.boxes.map((box, i) => (
                    <div key={i} className={`absolute border-2 ${isRight ? 'border-blue-500 bg-blue-500/10' : 'border-indigo-500 bg-indigo-500/10'} rounded-sm`}
                      style={{ left: `${box.x}%`, top: `${box.y}%`, width: `${box.w}px`, height: `${box.h}px` }}
                    />
                  ))}
                </>
              ) : <div className="text-slate-300 text-xs italic">等待图片上传...</div>}
            </div>
          </div>
          <div>
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2 text-[10px] font-black text-slate-400 uppercase tracking-widest"><FileText size={12} /> 识别文本</div>
              <button onClick={() => result.text && navigator.clipboard.writeText(result.text)}><Copy size={12} className="text-slate-300 hover:text-slate-500" /></button>
            </div>
            <div className="bg-slate-50 border rounded-xl p-3 min-h-[120px]">
              {image ? <pre className="text-xs text-slate-700 leading-relaxed font-sans whitespace-pre-wrap">{result.text || (isProcessing ? '正在计算...' : '暂无结果')}</pre> : <div className="h-full flex items-center justify-center text-slate-300 text-xs italic">暂无识别数据</div>}
            </div>
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="min-h-screen bg-[#f8fafc] text-slate-900 font-sans p-6">
      <header className="max-w-6xl mx-auto mb-8">
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-3">
            <div className="bg-slate-900 p-2 rounded-xl shadow-lg"><Zap className="text-yellow-400 w-5 h-5 fill-current" /></div>
            <h1 className="text-2xl font-black italic tracking-tighter uppercase text-slate-800">OCR2MD</h1>
          </div>
        </div>
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-1">
            <div className="relative h-44 bg-white border-2 border-dashed border-slate-200 rounded-3xl overflow-hidden group hover:border-blue-400 transition-all cursor-pointer">
              <input type="file" className="absolute inset-0 opacity-0 cursor-pointer z-20" onChange={handleUpload} />
              {image ? (
                <div className="relative w-full h-full flex items-center justify-center">
                  <img src={image} className="w-full h-full object-cover opacity-80" alt="Thumb" />
                  <div className="absolute inset-0 bg-slate-900/40 opacity-0 group-hover:opacity-100 transition-opacity flex flex-col items-center justify-center text-white z-10">
                    <RefreshCcw size={24} className="mb-2" /><span className="text-xs font-bold uppercase tracking-widest">点击更换图片</span>
                  </div>
                </div>
              ) : (
                <div className="w-full h-full flex flex-col items-center justify-center p-6 text-center">
                  <div className="w-12 h-12 bg-blue-100 text-blue-600 rounded-2xl flex items-center justify-center mb-3 group-hover:scale-110 transition-transform"><Upload size={24} /></div>
                  <span className="text-sm font-bold text-slate-700">上传测试图片</span>
                  <p className="text-[10px] text-slate-400 mt-1">支持拖拽文件到此处</p>
                </div>
              )}
            </div>
          </div>
          <div className="lg:col-span-2 bg-white border-2 border-slate-100 rounded-3xl p-6 flex flex-col justify-between shadow-sm">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-xs font-black text-slate-400 uppercase tracking-widest flex items-center gap-2"><HardDrive size={14} /> 模型管理</h3>
              {image && <button onClick={() => fetchOCRResults(image)} className="text-[10px] font-bold text-blue-600 flex items-center gap-1 hover:underline"><RefreshCcw size={12} /> 重新识别</button>}
            </div>
            <div className="grid grid-cols-2 gap-6">
              {['paddle', 'glm'].map((key) => {
                const dl = downloading[key];
                const name = key === 'paddle' ? 'PaddleOCR-VL-1.5' : 'GLM-OCR';
                return (
                  <div key={key} className="space-y-2">
                    <div className="flex justify-between items-center">
                      <span className="text-xs font-bold text-slate-700">{name}</span>
                      {dl.status === 'completed' ? <span className="text-[10px] text-green-600 font-bold flex items-center gap-1"><CheckCircle2 size={12} /> 已准备好</span> : 
                        <button disabled={dl.active} onClick={() => startDownload(key)} className="text-[10px] text-blue-600 font-bold hover:underline disabled:text-slate-300">{dl.active ? `下载中 ${dl.progress}%` : '下载权重'}</button>}
                    </div>
                    <div className="h-1.5 w-full bg-slate-100 rounded-full overflow-hidden">
                      <div className={`h-full transition-all duration-300 ${dl.status === 'completed' ? 'bg-green-500' : 'bg-blue-600 animate-pulse'}`} style={{ width: `${dl.progress}%` }}></div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      </header>
      <section className="max-w-6xl mx-auto">
        <div className="flex items-center gap-4 mb-8"><div className="h-px bg-slate-200 flex-grow"></div></div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 min-h-[500px] mb-12">
          {isProcessing ? (
            <div className="col-span-2 bg-white rounded-3xl border shadow-sm flex flex-col items-center justify-center py-32">
              <div className="w-12 h-12 border-4 border-blue-600 border-t-transparent rounded-full animate-spin mb-4"></div>
              <p className="text-sm font-bold text-slate-700">正在处理...</p>
            </div>
          ) : (
            <>
              <ModelCard result={results.paddle} modelKey="paddle" side="left" />
              <ModelCard result={results.glm} modelKey="glm" side="right" />
            </>
          )}
        </div>
      </section>
    </div>
  );
};

export default App;