# 基于 FPGA 的车牌识别系统

本项目基于 PYNQ-Z2（Zynq-7020），通过 USB UVC 摄像头采集 640×480 视频，定位车牌并通过 HDMI 显示识别结果。

## 项目实现与识别范围

本项目在 PYNQ-Z2 上实现了从 USB 摄像头采集、车牌定位、透视校正、整牌 OCR 到 HDMI 显示的处理链路。PL 负责颜色掩膜、形态学预处理和 HDMI 视频输出，PS 负责候选筛选、裁剪校正和 TinyPlateCTC 推理，并将定位框和识别结果叠加到画面上。

- **蓝牌演示**。当前部署包默认运行蓝牌模式，同一帧最多识别两张蓝牌，画面上的编号框与左上角的号码逐一对应。板载按键支持抓拍、暂停或恢复检测，以及查看首选候选的字符二值图。板端实测已验证双牌同屏，仍有省份字误读的案例。
- **新能源绿牌实验**。项目实现了绿牌颜色提案、候选筛选、裁剪和八字符整牌识别，并用 CCPD 绿牌和 CBLPRD 合成绿牌进行训练与评估。板端实拍表现不稳定，出现过无候选、字符错读和输出缺字。一次复测的 7 条绿牌候选均输出了不足 8 字符的结果。离线裁剪图指标仅衡量 OCR，板端表现还受定位和裁剪影响。当前公开部署包默认提供蓝牌演示，绿牌属于实验功能。

## 系统架构

- **PL** 负责可选的颜色掩膜与形态学预处理，并通过 VDMA、视频时序和 RGB2DVI 模块输出 HDMI。蓝牌部署使用 Cb150 颜色判定。
- **PS** 负责摄像头采集、候选筛选、透视校正和 OCR。TinyPlateCTC 读取 128×32 灰度裁剪，通过 CTC 解码输出整牌号码。
- 定位完成后先显示框，OCR 完成后再补号码。蓝牌部署支持同一帧最多两条结果，按编号对应定位框。

## 源码目录

| 目录 | 内容 |
| --- | --- |
| rtl、ip | RTL 与封装的 Vivado IP |
| vivado | 硬件集成与构建脚本 |
| ps | 板端运行、模型训练、评估与 ONNX 导出 |
| tests、sim | 软件检查与硬件仿真 |
| scripts | 工程辅助脚本 |

完整构建需要 Vivado、PYNQ 环境及对应工程输入。部分 Tcl 脚本以已有 Vivado 工程为基础。重新训练需要自行准备数据集。

## 上板运行

部署包位于本仓库的 [deploy/blue_only_buttons_bundle.zip](deploy/blue_only_buttons_bundle.zip)。解压后将整个 blue_only_buttons_bundle 目录复制到板端 /home/xilinx/。部署包包含配套的 bitstream、HWH、OCR 模型、运行脚本和校验清单。

接好 USB 摄像头和 HDMI OUT，将 SW0 拨到下方，然后运行：

```bash
cd ~/blue_only_buttons_bundle
sha256sum -c SHA256SUMS
python3 run_plate.py
```

- BTN0 抓拍当前画面和结果。
- BTN1 暂停或恢复检测。
- BTN2 在实时画面和第一候选的字符二值预览之间切换。
- Ctrl+C 退出时提交黑帧。

启动器输出 Results 路径，识别日志保存在对应运行目录的 live.csv 中。

## IP 核说明

封装好的 IP 集中在 ip 目录，构建时按所需子目录注册。灰度和 Sobel 有历史封装副本，选择对应阶段的封装即可。

| IP 核 | 功能和使用位置 |
| --- | --- |
| axis_grayscale_accel | 流式 RGB 转灰度，用于早期灰度与 DMA 验证。当前 OCR 在 PS 端转灰度。 |
| axis_sobel_accel | 流式 Sobel 梯度检测，用于边缘实验。当前蓝牌演示使用颜色掩膜提案。 |
| axis_blue_mask_accel | 依据 RGB 通道关系生成蓝色二值掩膜，用于早期蓝牌提案。 |
| axis_binary_morph_accel | 对二值掩膜做形态学处理，连接断开的车牌区域并抑制小斑点。 |
| axis_cb150_mask_accel | 由颜色像素生成 Cb 阈值掩膜，是当前蓝牌 PL 定位分支的颜色提案模块。 |
| rgb2dvi | 将像素数据和视频同步信号编码为 HDMI 接口使用的 TMDS 信号。 |

工程也使用 Vivado 自带的 AXI DMA 搬运识别像素和掩膜，AXI VDMA 读取显示帧缓冲，
VTC 产生视频时序，Clocking Wizard 提供视频时钟，AXI GPIO 读取按键或选择颜色模式。
这些标准 IP 由 Vivado 工程创建。

## RTL 文件说明

rtl 保存硬件逻辑和仿真测试台，ip 保存 Vivado 可注册的 IP 封装。模块是否进入 bitstream 由 Vivado 工程的顶层和连线决定。

| 文件 | 具体作用 |
| --- | --- |
| `axis_color_mask_accel.sv`、`axis_color_mask_accel.v` | AXI4-Stream 颜色掩膜模块，根据输入像素颜色输出二值掩膜。两者为同名模块的不同源码版本。 |
| `phase1/axis_grayscale_accel.sv` | 流式 RGB 转灰度，用于早期采集和 DMA 验证。 |
| `phase2/axis_sobel_accel.sv` | 流式 Sobel 梯度检测，用于边缘提案实验。 |
| `phase2/xgui/axis_sobel_accel_v1_0.tcl` | 定义 Sobel IP 在 Vivado 中的配置界面。 |
| `phase3/axis_blue_mask_accel.sv` | 根据 RGB 通道关系生成蓝色掩膜，用于早期颜色提案。当前蓝牌部署使用 ip 中封装的 Cb150 模块。 |
| `phase3/axis_binary_morph_accel.sv` | 对二值像素流执行形态学处理，连接断开区域或抑制小斑点。 |
| `plate_color_mask.sv` | 为基础定位流水线生成颜色掩膜。 |
| `morph_close_3x3.sv` | 对二值掩膜执行 3×3 闭运算。 |
| `plate_bbox.sv` | 根据掩膜统计候选区域边界。 |
| `plate_pipeline.sv` | 连接颜色掩膜、形态学和边界统计等基础定位模块。 |
| `roi_cropper.sv` | 在硬件识别实验中按候选区域裁剪像素。 |
| `char_projection.sv` | 对字符前景做投影统计，为字符分割提供依据。 |
| `char_segmenter.sv` | 根据字符前景分布确定单字符边界。 |
| `char_resampler.sv` | 将字符区域重采样为分类器需要的尺寸。 |
| `plate_template_classifier.sv` | 在硬件识别实验中进行字模匹配分类。 |
| `roi_ocr_controller.sv` | 调度硬件 ROI 字符处理和分类流程。当前演示的 OCR 在 PS 上运行 TinyPlateCTC。 |
| `hdmi_diag_mux.v` | 在正常视频与硬件诊断色条之间切换。 |
| `hdmi_signal_probe.v` | 统计像素时钟、行和场事件，用于排查 HDMI 时序。 |
| `hdmi_stocklike_clock.v` | 用于对照原厂 HDMI 时钟方案。 |
| `tb_color_mask.v` | 颜色掩膜的仿真测试台。 |
| `tb_hdmi_signal_probe.v` | HDMI 时序探针的仿真测试台。 |

## 在 Vivado 中重新生成 bitstream

本项目三按键部署版使用 **Vivado 2022.2** 完成实现。目标为 PYNQ-Z2，器件 XC7Z020-1CLG400，顶层为 `VIP_system_wrapper`。

### 使用已有 Vivado 工程

源码包中的 Tcl 是对已有工程的增量构建。若已有本项目的绿牌流水线工程，将其放在 `build/plate_ocr_hdmi_green_pipeline/`，确认其中的 `plate_ocr.xpr`、`VIP_system.bd` 及 IP 源文件齐全。若已有三按键工程，脚本会优先打开 `build/plate_ocr_hdmi_chars_button/plate_ocr.xpr`。先在 Vivado 中检查项目引用的 IP 仓库是否指向本包 `ip/` 下的对应子目录，并重新扫描 IP Catalog。然后在已配置 Vivado 环境的终端，从仓库根目录执行：

```sh
vivado -mode batch -source vivado/build_hdmi_chars_button.tcl
```

检查 `build/plate_ocr_hdmi_chars_button/exports/` 中的时序、DRC 及输出文件。部署时将同次生成的 `phase3_hdmi_chars.bit` 与 `phase3_hdmi_chars.hwh` 一起替换到板端部署目录，再运行原启动命令。

### 从空工程重建或移植

发布包仅提供最新三按键构建入口及配套约束。增量构建需要已有的三按键工程或绿牌流水线基础工程。基础 .xpr 需另行准备。可在 Vivado 创建目标板工程，并参考 `vivado/` 中的约束与 `ip/` 中的封装重建 Block Design。

1. 在 IP Catalog 注册需要的 IP 子目录。按版本选择灰度或 Sobel 的历史封装，避免重复注册同一 VLNV。
2. 配置 Zynq PS、DDR 和 AXI 互连。复建检测 AXI DMA 和显示 VDMA 两条数据路径，连接颜色掩膜、形态学及 HDMI 输出逻辑。
3. 检查时钟、复位、AXI 地址与接口宽度。当前显示程序依赖 100 MHz FCLK0 参考，并按 640×480 配置。移植到其他板卡时，按其视频接口、时钟源、按键引脚和电气标准修改 XDC，核对 PS 程序读取的 IP 实例名和寄存器映射。
4. 在 Vivado 依次执行 Validate Design、Generate Output Products、Create HDL Wrapper、综合和实现。检查时序裕量及 DRC 后生成 bitstream，并保留该 Block Design 生成的匹配 HWH。
5. 若更换 Vivado 版本，先在工程副本检查 IP 升级和重新实现结果。

## 数据集与模型

| 数据来源 | 在项目中的用途 |
| --- | --- |
| CCPD（含 CCPD2019） | 车牌定位离线评估，并提供蓝牌、绿牌裁剪用于 OCR 训练与验证。 |
| CLPD | 补充真实场景车牌样本，并单独统计整牌识别率。 |
| CBLPRD-330k v1 | GAN 合成数据。混合微调选用普通蓝牌及两类新能源绿牌，并保留独立合成测试集。 |


CBLPRD 微调以原 TinyPlateCTC 权重为起点。最终训练清单共 219,877 张，其中原 CCPD、检测裁剪及 CLPD 共 25,761 张，CBLPRD 蓝牌 69,199 张、绿牌 124,917 张。验证集有 11,511 张，另有 1,985 张独立合成测试图。数据划分按整牌号码与完全相同的图像像素隔离。

真实裁剪验证中，原模型和微调模型的整牌准确率分别为 CCPD 蓝牌 99.55% 与 99.30%、CCPD 绿牌 95.60% 与 94.71%、CLPD 79.75% 与 83.97%。真实验证集参与选模。摄像头端到端表现还需用独立实拍统计定位漏检、裁剪缺字和识别错误。

当前蓝牌三按键部署使用原模型。复现训练需自行准备数据集，并遵守各数据集的使用条款。数据准备、训练、评估和 ONNX 导出脚本位于 `ps/`。

### Dataset links

- [CCPD2019 and CCPD-Green](https://github.com/detectRecog/CCPD)
- [CLPD](https://github.com/wangpengnorman/CLPD_dataset)
- [CBLPRD-330k](https://github.com/SunlifeV/CBLPRD-330k)

## 板端验证与复现范围

当前部署包为 PYNQ-Z2 蓝牌版，支持两个蓝牌结果同屏显示。一次板端实测从检测线程启动到首框为 0.56 秒，到首条 OCR 结果为 0.74 秒。这两项计时不包含主程序启动和 Overlay 下载。双牌实测中两个框与两行号码均正确对应。

下载部署 ZIP 即可按上板说明运行。重新生成同版 bitstream 需要匹配的基础 Vivado 工程，构建步骤及输入见上文。

## 第三方来源

HDMI 输出使用 Digilent 的 RGB2DVI IP，来源为 [Digilent vivado-library](https://github.com/Digilent/vivado-library/tree/master/ip/rgb2dvi)。第三方 IP 和数据集适用各自的许可条款。本仓库的项目源码许可证尚待选定。
