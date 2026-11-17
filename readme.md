# Rope-LiveMe 实时换脸

> 解释器、CUDA 组件库、ffmpeg 全部使用系统里已有的全局安装，工程目录里只保留 Python 源码、模型与配置。

- 运行方式：双击 `Start-Rope-LiveMe.bat`
- 依赖方式：全局 Python 3.12 + torch cu130 + TensorRT 10 + PATH 上的 ffmpeg

---

## 0. 分支说明

| 分支 | 用途 |
| --- | --- |
| `main`（默认分支） | 版本基线镜像，只用于对照历史基线，不存放本项目提交 |
| `master` | 本项目的全部代码，日常开发与发布都在这里 |

> 自有改动一律提交到 `master`，不要写入 `main`。

---

## 1. 环境要求

| 项      | 要求                                                |
| ------ | ---------------------------------------------------- |
| 操作系统   | Windows 10 / 11 x64 |
| 显卡     | NVIDIA，支持 CUDA 12+/13，显存 ≥6 GB（建议 ≥12 GB）|
| 驱动     | 支持 CUDA 13 的 Game Ready / Studio 驱动 |
| Python | **CPython 3.10 – 3.12，64 位**（3.13 不支持） |
| 关键依赖   | torch（+cuXXX）、onnxruntime-gpu、tensorrt、opencv(contrib)、customtkinter、numpy、scipy、scikit-image、psutil、pywin32、pyvirtualcam、pyshark、simpleobsws、obs-websocket-py、obsws-python。见 `requirements.txt` |
| ffmpeg | `ffmpeg.exe` 在 PATH 上（同目录需有 `ffplay.exe` 供预览） |
| 磁盘     | 模型 10.3 GB + 空闲 ≥5 GB（录制/引擎）|

> CPU-only 机器理论上能起界面，但换脸/增强全部依赖 CUDA EP，实际不可用。

---

## 2. 安装

### 2.1 装 Python 3.12 x64

装到任意路径，**勾选 Add to PATH**。装完验证：

```bat
python -c "import sys,struct;print(sys.version, struct.calcsize('P')*8)"
```

### 2.2 装组件库

```bat
cd /d <项目根目录>
python -m pip install -r requirements.txt
```

`requirements.txt` 里 torch 行带 `+cu130` 后缀并配了 `--extra-index-url`，直接装即可。纯 CPU 机器把 torch/torchvision/torchaudio 三行换成
`https://download.pytorch.org/whl/cpu` 的对应版本。

装完自检：

```bat
python -c "import torch,onnxruntime,cv2,customtkinter;print(torch.__version__, torch.cuda.is_available(), onnxruntime.get_available_providers())"
```

期望看到 `True` 和包含 `TensorrtExecutionProvider` 的列表。

### 2.3 装 ffmpeg

把 `ffmpeg.exe`、`ffplay.exe` 所在目录加入 PATH，或直接：

```bat
setx FFMPEG_PATH "C:\ffmpeg\bin"
```

启动器会把 `where ffmpeg.exe` 的结果自动导出成 `FFMPEG_PATH`。

> **候选必须是「真能跑」的**：WinGet 的 `...\Microsoft\WinGet\Links\ffmpeg.exe` 是 0 字节的
> App Execution Alias（断链符号链接），文件存在但执行报「找不到路径」。启动器、`bootstrap.py`、
> `VideoManager.get_ffmpeg_path()` 三层都会实际执行一次 `ffmpeg -hide_banner -version`，
> 遇到坏候选会打印 `skip ffmpeg [broken binary]: <目录>` 后继续找下一个，最终日志里出现
> `ffmpeg: 使用 <目录>` 才是真正生效的路径。

### 2.4 放模型

把 58 个模型文件按相同相对路径放进 `models\`（约 10 GB，清单见 [models/README.md](models/README.md)，含文件名与大小）。模型不随仓库分发。

缺失的表现是启动时 ONNX 加载报 `No such file`，日志里能看到具体文件名。

### 2.5 指定解释器（推荐用环境变量）
### 2.6 首次运行会自动生成配置文件

`config\` 下的三个文件都与本机相关，**不入库**，首次启动时自动创建：

| 文件 | 内容 | 缺失/损坏时的行为 |
| --- | --- | --- |
| `config\data.json` | 语言、上次的视频/人脸/输出目录、窗口位置 | 自动写入默认值；JSON 损坏则备份为 `data.json.broken` 后重建 |
| `config\config.json` | OBS WebSocket 地址/端口/密码、tshark 过滤条件 | 自动写入骨架（密码留空，连 OBS 时再填）；读取处已做容错 |
| `config\gpu_config.json` | 显卡型号与优化参数 | 自动重新检测并生成 |

想看格式参考 `config\config.example.json`。因此**换机器不用手工准备配置**，直接双击启动即可。

工程里**不写死绝对路径**，`python_path.txt` 用环境变量引用解释器：

```
env:ROPE_PYTHON
```

只需设置一次（用户级，之后双击启动不受 PATH 影响）：

```bat
setx ROPE_PYTHON "C:\Python312\python.exe"
```

`python_path.txt` 支持的写法：

| 写法 | 含义 |
|---|---|
| `env:ROPE_PYTHON` | 读取同名环境变量 |
| `%ROPE_PYTHON%` | 直接写 `%VAR%` 形式，同样会展开 |
| `%PYTHON_HOME%\python.exe` | 变量与相对片段混写 |
| `C:\Python312\python.exe` | 也可以直接写死（不推荐） |

`#` 与 `rem` 开头的行是注释，第一条有效行生效。

---

## 3. 启动、自检与首次运行

### 3.1 三种启动方式

```bat
:: 1) 双击
Start-Rope-LiveMe.bat

:: 2) 命令行（能看完整输出）
Start-Rope-LiveMe.bat

:: 3) 只做环境自检，不进主循环
python bootstrap.py --dry-run
```

### 3.2 第一次启动会做什么

1. `bootstrap.py` 切到工程根、注入 `ROPE_SESSION` / `ROPE_LAUNCH_TIME`，短路授权检查
2. 导入 `rope\*`（约 40 个模块，含 torch/TRT 初始化，10–30 秒）
3. `Models.Models(device='cuda')` 建全部推理会话（首次较慢；TRT 引擎缺失时会重建并缓存到 `tensorrt-engines\`）
4. 创建主窗口 `Rope-LiveMe`

### 3.3 `models\` 清单（按用途分组）

| 用途           | 文件                                                                                                                                                                                                                                               |
| ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 人脸检测         | `det_10g.onnx`、`scrfd_2.5g_bnkps.onnx`、`yoloface_8n.onnx`、`yunet_n_640_640.onnx`、`peppapig_teacher_Nx3x256x256.onnx`、`face_landmarks_detector_Nx3x256x256.onnx`、`face_blendshapes_Nx146x2.onnx`                                                  |
| 关键点/形状       | `2d106det.onnx`、`1k3d68.onnx`、`2dfan4.onnx`、`landmark.onnx`、`meanshape_68.pkl`                                                                                                                                                                   |
| 换脸           | `inswapper_128.fp16.onnx`、`simswap_512_unoff.onnx`、`simswap_arcface_model.onnx`、`arcface_simswap.onnx`、`ghost_arcface_backbone.onnx`、`ghost_unet_1_block.onnx`、`ghost_unet_2_block.onnx`、`ghost_unet_3_block.onnx`、`w600k_r50.onnx`、`res50.onnx` |
| 人脸解析/遮挡      | `79999_iter.pth`、`faceparser_fp16.onnx`、`faceparser_resnet34.onnx`、`occluder.onnx`、`XSeg_model.onnx`                                                                                                                                             |
| 增强/修复        | `GFPGANv1.4.onnx`、`GPEN-BFR-256/512/1024/2048.onnx`、`codeformer_fp16.onnx`、`RestoreFormerPlusPlus(.fp16).onnx`、`VQFRv2.fp16.onnx`                                                                                                                |
| 超分           | `RealESRGAN_x2plus.fp16.onnx`、`RealESRGAN_x4plus.fp16.onnx`、`realesr-general-x4v3.onnx`、`BSRGANx2.fp16.onnx`、`BSRGANx4.fp16.onnx`、`4x-UltraSharp.fp16.onnx`、`4x-UltraMix_Smooth.fp16.onnx`                                                       |
| 上色           | `ddcolor.onnx`、`ddcolor_artistic.onnx`、`ColorizeVideo.fp16.onnx`、`ColorizeStable.fp16.onnx`、`ColorizeArtistic.fp16.onnx`                                                                                                                         |
| CLIPSeg 文本分割 | `rd64-uni-refined.pth`（配合 `rope\external\cliplib\` 使用）                                                                                                                                                                                           |
| 其它           | `epoch_16_best.ckpt`、`grid_sample_3d_plugin.dll`                                                                                                                                                                                                 |

---

## 4. 使用流程

1. 启动后主窗口分左（素材面板）、中（预览 + 时间轴）、右（参数）三块
2. **输入源**：`输入视频` 选要换脸的视频/摄像头；`输入人脸` 选目标人脸图（一张或多张）
3. **预览**：`播放` 走时间轴预览；`蒙版视图` 看解析/遮挡结果
4. **参数**：右侧面板切 `恢复(Restorer)`、`方向(Orientation)`、`强度(Strength)`、`差异(Differencing)`、`遮挡(Occluder)`、`人脸解析(Face Parser)`、`文本蒙版(CLIP)`、`颜色调整`、`面部调整`、`换脸模型` 等
5. **输出**：`录制` 开始写视频；虚拟摄像头开关把处理结果送给 OBS 等程序
6. **标记与快捷键**：时间轴上按 `f` 加标记、`q`/`w` 前后跳；全部键位见 `shortcuts.json` 与设置页

---

## 5. 配置说明

| 文件                       | 内容                                                                                 |
| ------------------------ | ---------------------------------------------------------------------------------- |
| `config\config.json`     | `obs.host/port/password`（OBS WebSocket）、`tshark.interface/display_filter`（RTMP 抓流） |
| `config\data.json`       | 界面状态：`source videos` / `source faces` / `saved videos` 路径、窗口几何、各参数开关               |
| `config\gpu_config.json` | 显存阈值、批大小、优化档位                                                                      |
| `shortcuts.json`         | 24 条快捷键（`时间轴开始`=`z`、`记录`=`r`、`Play`=`space` …）                                     |

`config\data.json` 里的路径是绝对路径，**移动工程目录后要同步改这里的两行**。

---

## 6. 日志与排障

| 文件                        | 内容                                                                               |
| ------------------------- | -------------------------------------------------------------------------------- |
| `logs\app-YYYYMMDD.log`   | **主日志**（UTF-8，10 MB × 5 滚动）。root logger 统一收集，含 GPU 检测、模型加载、各模块 INFO/ERROR、未捕获异常栈 |
| `logs\YYYYMMDD.log`       | `Rope-LiveMe` 命名 logger 的输出（`logger_config` 那一份，通常很小）                                           |
| `logs\license_client.log` | 授权客户端日志                                                                          |
| `logs\_dialogs.log`       | `bootstrap.py` 镜像的弹窗内容（每次弹窗追加，含完整 traceback）                                     |

控制台只显示 ERROR（`production_logging` 的设定），**INFO/DEBUG 全部落到 `app-*.log`**，所以窗口看起来安静不代表没记录。排障顺序：

1. `logs\_dialogs.log` 有没有 traceback（弹过窗就一定有）

2. `logs\app-YYYYMMDD.log` 最后 50 行看时间线

3. 仍不够就临时开调试模式：

   ```bat
   set DEBUG=1
   Start-Rope-LiveMe.bat
   ```

   调试模式把 root 级别降到 DEBUG 并输出到控制台，同时仍写 `app-*.log`。

排障顺序：先看 `_dialogs.log` 有没有 traceback → 再看当日主日志的时间线 → 最后才怀疑环境。

---

## 附录：变更记录

| 模块 | 内容 |
| --- | --- |
| 工程整理 | 修复运行期解包异常与配置加载崩溃；路径改为相对路径或环境变量；统一命名 `Rope-LiveMe` |
| 界面 | 修复下拉项乱码（`原始` / `默认`）、状态栏文字关闭但保留栏体、新图标与背景、标题去掉版本号 |
| 启动链 | 解释器走 `env:ROPE_PYTHON`；ffmpeg 候选改为"能跑才算"，跳过 WinGet 0 字节 App Execution Alias；启动器统一 CRLF 换行 |
| 代码质量 | 修复 GUI 时间轴处理器与激活窗口的未定义名、重复定义；`RTX50Compatibility` 的 `except … as e` 作用域与裸 `raise`；`FaceUtil` 全局变量遮蔽；`license_online` 未定义函数 |
