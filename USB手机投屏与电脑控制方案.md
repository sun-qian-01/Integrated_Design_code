# USB 手机投屏与电脑控制方案

> 适用环境：Windows 电脑 + 安卓手机。本文按安卓的“USB 调试”功能设计；iPhone 不适用此流程。
> 编写及实操补充日期：2026 年 9 月 24 日。已在本机确认 vivo V2502A 的 USB 调试连接，并启动 scrcpy 投屏窗口、建立画面渲染；尚未实际验证点击、长按、拖动及小车联调。

## 一、推荐方案

使用 **scrcpy + ADB，通过 USB 数据线连接手机与电脑**。

- scrcpy 在电脑窗口中实时显示安卓手机屏幕，并将鼠标和键盘操作传给手机。
- 鼠标左键可点击手机界面，按住并移动可模拟拖动，适合操作 App 按钮和部分虚拟摇杆。
- 无需 Root，也无需在手机上手动安装常驻投屏 App；运行时会通过 ADB 启动所需服务。
- ADB 负责设备发现、调试连接及授权，scrcpy 负责屏幕传输和交互。
- USB 投屏不占用手机连接小车所需的 Wi-Fi，适合本课程项目。

“实时”指低延迟交互，并非零延迟。实际表现受手机编码性能、USB 连接、分辨率和原小车视频延迟影响。

## 二、在小车项目中的连接方式

```text
电脑：scrcpy 窗口，显示手机画面并接收鼠标操作
       ⇅ USB：屏幕传输与触控指令
安卓手机：运行原小车控制 App
       ⇅ 小车原有连接方式，如 Wi-Fi 或蓝牙
带摄像头小车：发送视频并执行运动指令
```

先在手机上确认原 App 能正常查看小车画面和操控小车，再连接电脑投屏。手机与小车之间的原有通信方式保持不变。

**此方案解决“电脑看手机、电脑点手机”，并不直接开放小车的视频接口或运动协议。** 后续如需自动识别与自主导航，仍应优先验证小车原生接口；屏幕识别和自动点击可作为过渡验证路径，但依赖 App 布局，稳定性较弱。

## 三、所需设备与软件

| 项目 | 要求 |
| --- | --- |
| 电脑 | Windows 10/11，常规电脑选择 Windows 64 位版本 |
| 手机 | 支持 USB 调试的安卓手机，建议使用较新系统 |
| 数据线 | 支持数据传输，不能只是充电线 |
| scrcpy | 从官方 GitHub Releases 下载 Windows 压缩包 |
| ADB | 官方 Windows scrcpy 压缩包通常包含 adb.exe；若缺少，另安装官方 Platform Tools |
| 驱动 | 仅在 Windows 无法识别 ADB 设备时，按手机品牌安装相应官方驱动 |

官方入口：

- scrcpy 发布页：https://github.com/Genymobile/scrcpy/releases
- Windows 安装说明：https://github.com/Genymobile/scrcpy/blob/master/doc/windows.md
- Android Platform Tools：https://developer.android.com/tools/releases/platform-tools
- Windows USB 驱动说明：https://developer.android.com/studio/run/oem-usb

## 四、首次配置与启动

### 4.1 在电脑安装 scrcpy

推荐下载官方 Windows 压缩包并解压，例如放到 `C:\Tools\scrcpy`。确认目录内能找到 `scrcpy.exe` 和 `adb.exe`。以下命令均按这一示例路径编写，请按实际解压位置修改。

也可使用 WinGet 安装：

```powershell
winget install --exact Genymobile.scrcpy
```

WinGet 安装后重新打开终端，通常可以直接运行 `scrcpy` 和 `adb`。下文统一采用解压目录中的 `.\scrcpy.exe`、`.\adb.exe`，避免 PATH 配置差异。

### 4.2 在手机开启 USB 调试

1. 进入“设置 → 关于手机”，连续点击“版本号”或厂商对应的系统版本项，开启开发者选项。
2. 进入“开发者选项”，打开“USB 调试”。不同品牌的菜单路径可能不同。
3. 用 USB 数据线连接电脑，解锁手机。
4. 若弹出 USB 用途选择，可选择“文件传输”。这不是所有机型的硬性要求，但有助于排查仅充电连接问题。
5. 在“是否允许 USB 调试”提示中确认当前电脑。仅在自己的可信电脑上选择“始终允许”。

部分机型，尤其部分小米系统，可能还需要开启“USB 调试（安全设置）”或类似的输入模拟权限，并按系统提示重启。普通投屏能用，不代表触控注入权限已经具备。通常无需开启“USB 安装”。

### 4.3 确认电脑识别手机

打开 PowerShell：

```powershell
Set-Location 'C:\Tools\scrcpy'
.\adb.exe devices
```

预期输出示例：

```text
List of devices attached
ABC123456    device
```

- `device`：调试连接正常，可以启动投屏。
- `unauthorized`：在手机端解锁并接受调试授权。
- `offline`：尝试重新插拔、重新授权或重启 ADB。
- 列表为空：优先检查数据线、USB 接口、调试开关和驱动。

#### 本机实操：定位程序并确认手机（2026-09-24）

**第一步：找到可执行程序。** 本次在下载目录发现了 `scrcpy-win64-v4.1.zip`，同时在 WinGet 软件目录找到了可直接使用的 scrcpy 4.1。因此实际使用的是已安装的软件，而不是直接运行下载的压缩包，也没有重新安装。

本机实际程序目录为：

```text
C:\Users\sunqi\AppData\Local\Microsoft\WinGet\Packages\Genymobile.scrcpy_Microsoft.Winget.Source_8wekyb3d8bbwe\scrcpy-win64-v4.1
```

需要重新查找时，可在 PowerShell 运行：

```powershell
Get-Command adb,scrcpy -ErrorAction SilentlyContinue | Select-Object Name,Source
Get-ChildItem -LiteralPath "$env:LOCALAPPDATA\Microsoft\WinGet\Packages" -Recurse -File -Filter scrcpy.exe -ErrorAction SilentlyContinue | Select-Object FullName
```

第一条检查程序是否已加入 PATH；没有输出不代表未安装，第二条会搜索当前用户的 WinGet 安装目录。若自行解压在其他位置，应使用该解压目录。软件升级后版本目录名称可能变化。

**第二步：使用该目录自带的 ADB 查询手机。** 打开 PowerShell，执行：

```powershell
$scrcpyDir = 'C:\Users\sunqi\AppData\Local\Microsoft\WinGet\Packages\Genymobile.scrcpy_Microsoft.Winget.Source_8wekyb3d8bbwe\scrcpy-win64-v4.1'
& (Join-Path $scrcpyDir 'adb.exe') devices -l
```

`&` 是 PowerShell 的调用运算符，用来运行路径指向的程序；`Join-Path` 拼接文件路径；`devices -l` 列出设备及型号等信息。这种写法不依赖 PATH，也不要求先切换工作目录。

本次实际输出：

```text
List of devices attached
10AG1Z064A004P3        device product:PD2502 model:V2502A device:PD2502 transport_id:5
```

| 字段 | 本次结果与含义 |
| --- | --- |
| `10AG1Z064A004P3` | 手机序列号，后续用它指定投屏目标；更换手机需替换 |
| `device` | ADB 连接正常，当前电脑已获得 USB 调试授权 |
| `model:V2502A` | 系统报告的手机型号；后续 scrcpy 日志进一步显示品牌为 vivo |
| `transport_id:5` | 本次连接的传输编号，重新连接后可能变化，无需写入启动命令 |

本次无需重新授权、安装驱动或重启 ADB。检查时还发现电脑已有手机助手的 ADB 进程，但本次查询和投屏均成功，因此没有停止该进程。`device` 仅说明调试连接可用，不等于已完成鼠标控制或小车功能测试。

### 4.4 启动投屏和控制

先使用基础命令，确认画面和点击都能工作：

```powershell
.\scrcpy.exe --select-usb --no-audio
```

`--select-usb` 选择 USB 连接的设备，避免误连无线调试设备；如果存在多台 USB 手机，按后文指定序列号。

如需降低处理负担，再使用：

```powershell
.\scrcpy.exe --select-usb --no-audio --max-size=1280 --max-fps=30 --video-bit-rate=4M --stay-awake
```

| 参数 | 作用 |
| --- | --- |
| `--no-audio` | 不转发音频，先专注屏幕与控制 |
| `--max-size=1280` | 限制视频最大边长，减少传输与处理负担 |
| `--max-fps=30` | 限制最高帧率，不保证实际恒定达到 30 帧 |
| `--video-bit-rate=4M` | 设置视频编码码率为 4 Mbps |
| `--stay-awake` | 插电时保持唤醒，不能用于绕过锁屏或系统安全策略 |

参数以所安装版本的 `.\scrcpy.exe --help` 为准。若文字或小车画面细节不够清楚，可以提高最大边长和码率，再实测延迟。

#### 本机实操：启动窗口并核对运行结果（2026-09-24）

**第一步：指定刚才识别到的手机，启动投屏。** 日常手动使用时，在 PowerShell 执行下面两行即可，启动参数与本次实际操作一致：

```powershell
$scrcpyDir = 'C:\Users\sunqi\AppData\Local\Microsoft\WinGet\Packages\Genymobile.scrcpy_Microsoft.Winget.Source_8wekyb3d8bbwe\scrcpy-win64-v4.1'
& (Join-Path $scrcpyDir 'scrcpy.exe') --serial=10AG1Z064A004P3 --no-audio --max-size=1280 --max-fps=30 --video-bit-rate=4M --stay-awake
```

`--serial` 精确选择本次手机，替代通用示例中的 `--select-usb`，无需同时使用两者。其余参数含义见上表。未添加 `--no-control`，所以保留默认的鼠标和键盘控制。此方式会在当前终端输出日志，并打开独立投屏窗口；窗口关闭后命令结束。

**第二步：如需复现本次“独立启动并保存日志”的方式，使用以下替代命令。** 两种启动方式选一种即可；已有投屏窗口正常运行时，无需重复启动。

```powershell
$scrcpyDir = 'C:\Users\sunqi\AppData\Local\Microsoft\WinGet\Packages\Genymobile.scrcpy_Microsoft.Winget.Source_8wekyb3d8bbwe\scrcpy-win64-v4.1'
$taskLogDir = Join-Path $env:TEMP ('scrcpy-session-' + (Get-Date -Format 'yyyyMMdd-HHmmss'))
New-Item -ItemType Directory -Path $taskLogDir | Out-Null
$scrcpyArgs = @(
    '--serial=10AG1Z064A004P3'
    '--no-audio'
    '--max-size=1280'
    '--max-fps=30'
    '--video-bit-rate=4M'
    '--stay-awake'
)
$launchOptions = @{
    FilePath = Join-Path $scrcpyDir 'scrcpy.exe'
    WorkingDirectory = $scrcpyDir
    ArgumentList = $scrcpyArgs
    RedirectStandardOutput = Join-Path $taskLogDir 'stdout.log'
    RedirectStandardError = Join-Path $taskLogDir 'stderr.log'
    PassThru = $true
}
$scrcpyProcess = Start-Process @launchOptions
Write-Output "ProcessId=$($scrcpyProcess.Id)"
Write-Output "LogDirectory=$taskLogDir"
```

`Start-Process` 启动独立进程；`PassThru` 返回进程对象，便于后续检查。两个日志文件分别保存标准输出和标准错误，后者也可能包含正常的服务推送提示，不应仅凭文件非空判断失败。日志存放在 Windows 临时目录，每次按时间新建文件夹。

**第三步：等待窗口出现，再在同一个 PowerShell 会话中检查。** 以下命令用于上面的日志启动方式：

```powershell
Get-Process -Id $scrcpyProcess.Id -ErrorAction SilentlyContinue |
    Select-Object Id,ProcessName,MainWindowTitle,Responding
Get-Content -LiteralPath (Join-Path $taskLogDir 'stdout.log'),(Join-Path $taskLogDir 'stderr.log')
```

若刚启动时尚无窗口标题或日志，可待窗口出现后再次执行检查。变量仅在当前 PowerShell 会话中有效；新开终端后需要重新获取进程或使用实际日志路径。

本次实际检查结果：

```text
Id     ProcessName    MainWindowTitle    Responding
26088  scrcpy         V2502A             True
```

关键日志摘录：

```text
scrcpy 4.1 <https://github.com/Genymobile/scrcpy>
INFO: ADB device found:
INFO:     -->   (usb)  10AG1Z064A004P3                 device  V2502A
[server] INFO: Device: [vivo] vivo V2502A (Android 16)
INFO: Renderer: direct3d11
INFO: Texture: 576x1280
```

这说明已选中 USB 手机、启动手机端服务，并建立电脑端画面渲染；投屏窗口标题为 `V2502A`，当时处于响应状态。`576x1280` 是此次投屏纹理尺寸，不是对手机原生分辨率的测量。进程号 `26088` 仅是历史记录，重新启动会变化。

本次日志位置如下，临时文件可能被系统清理：

```text
C:\Users\sunqi\AppData\Local\Temp\scrcpy-session-20260924-154711\stdout.log
C:\Users\sunqi\AppData\Local\Temp\scrcpy-session-20260924-154711\stderr.log
```

**第四步：人工验证交互。** 在 `V2502A` 窗口中，先对普通手机界面进行单击、长按和拖动，观察手机是否同步响应。此次操作止于识别设备、启动窗口和检查日志，没有代替用户点击手机，因此“点击已验证”“持续运行已验证”和“小车联调已完成”仍不能勾选。实际交互测试方法见第五、六节。

## 五、电脑端如何操作手机

| 电脑操作 | 对应手机操作 |
| --- | --- |
| 鼠标左键单击 | 点击当前位置 |
| 鼠标左键按住 | 长按当前位置 |
| 按住左键并移动 | 拖动、滑动或操作部分虚拟摇杆 |
| 键盘输入 | 在已选中的文本框输入；中文输入行为取决于输入法与应用 |
| 复制文字后 Ctrl+V | 通常可粘贴到手机文本框，具体受应用支持情况影响 |

默认即可控制，不需要额外打开控制开关；不要添加 `--no-control`，它会禁止交互。

首次先在手机设置页测试点击、长按和滑动，再进入小车 App。小车方向键若要求“按住持续运动、松开停止”，需要单独测试按下与释放是否都能正确传递。

普通鼠标不等于两根独立手指。若 App 要求同时按住油门和方向，或操作两个虚拟摇杆，默认单指操作可能不能完整替代手机多点触控，不能直接假定可用。

## 六、与小车联调的建议步骤

1. 手机连接小车，在手机上确认画面、方向键和停止功能正常。
2. 手机通过 USB 连接电脑，启动 scrcpy。
3. 先将小车架起使驱动轮离地，测试电脑端点按、长按、松开及停止。
4. 检查松开按钮、鼠标拖出按钮区域、窗口失去焦点后是否仍持续运动。
5. 在有人随时能够停止小车的条件下，验证关闭投屏或 USB 断开后的实际行为。
6. 确认停止行为可靠后，在有边界的场地进行低速地面测试。

**关闭 scrcpy 或拔掉 USB，不保证小车会停止。** 手机 App 与小车可能仍保持连接，必须保留可直接操作的手机停止按钮或设备电源开关。

## 七、可选：制作一键启动入口

在 scrcpy 解压目录中创建 `启动手机投屏.bat`，内容如下。此处仅给出模板，本文不自动创建或执行脚本。

```bat
@echo off
cd /d "%~dp0"
scrcpy.exe --select-usb --no-audio --max-size=1280 --max-fps=30 --video-bit-rate=4M --stay-awake
pause
```

之后连接并授权手机，双击即可启动。`pause` 用于在退出后保留错误信息，便于排查。

## 八、常见问题排查

### 8.1 ADB 找不到手机

- 更换确认支持数据传输的数据线，优先直连电脑 USB 接口，暂时移除扩展坞。
- 保持手机解锁，确认 USB 调试已打开，尝试切换到文件传输模式。
- 查看 Windows 设备管理器是否存在异常的 Android/ADB 设备；必要时安装手机厂商官方驱动。
- 若需要重启 ADB，运行以下命令。重启会暂时影响其他正在使用 ADB 的工具或设备会话。

```powershell
.\adb.exe kill-server
.\adb.exe start-server
.\adb.exe devices
```

### 8.2 显示 unauthorized

解锁手机并确认调试授权。如果没有弹窗，可在手机开发者选项中撤销 USB 调试授权，然后重新插拔并授权；撤销会清除此前信任的电脑记录。

### 8.3 有画面但不能点击

- 确认启动时没有使用 `--no-control`。
- 检查厂商的“USB 调试（安全设置）”或输入模拟权限。
- 先在系统设置页测试：若系统页可以操作而某个 App 不行，检查该 App 的输入限制或特殊控件。
- 安全确认、支付等界面可能限制外部输入，不能保证所有页面都可控制。

### 8.4 画面卡顿或延迟明显

先降低分辨率和码率：

```powershell
.\scrcpy.exe --select-usb --no-audio --max-size=1024 --max-fps=30 --video-bit-rate=2M
```

同时检查 USB 连接、手机温度及后台负载。小车相机到手机 App 本身也可能存在延迟；若手机原画面已卡顿，单独调整 scrcpy 不能消除上游问题。降低码率可能损失障碍物细节，需要在清晰度和流畅度之间实测取舍。

### 8.5 手机界面可见，但 App 视频区域黑屏

先比较手机本机画面与电脑投屏：若仅电脑上的某个区域黑屏，可能涉及受保护内容、安全窗口或设备编码兼容性。可升级官方 scrcpy 并排查编码器兼容性；受保护界面不保证能投屏。若小车视频无法显示，应回到原生视频接口方案评估。

### 8.6 同时连接多台手机

先通过 `.\adb.exe devices` 查到序列号，然后指定目标设备，替换示例值：

```powershell
.\scrcpy.exe --serial=ABC123456 --no-audio
```

指定序列号时使用这条命令即可，不再叠加 `--select-usb`。

## 九、后续若需程序自动点击

当前需求使用 scrcpy 的鼠标控制即可。若后续需要：

```powershell
.\adb.exe shell input tap 500 1000
.\adb.exe shell input swipe 500 1400 500 600 400
```

以上坐标只是示例，不对应实际小车 App。坐标依据手机当前显示方向的屏幕坐标，而不是电脑投屏窗口的像素；第二条命令末尾的 `400` 表示滑动时长，单位毫秒。手机分辨率、横竖屏和 App 布局改变后需要重新校准。

单次 `tap` 是一次点击，不是持续按住，也不适合直接构成高频运动控制。若需要稳定的自动操作，可进一步评估基于界面控件的自动化；自绘摇杆或视频控件未必能被识别为可操作控件。课程的自主导航仍优先考虑直接获取视频和调用小车控制协议。

## 十、验收清单

- [x] ADB 列表中目标手机状态为 `device`（2026-09-24，vivo V2502A）。
- [x] scrcpy 投屏窗口已启动，日志确认画面渲染建立（2026-09-24）。
- [ ] 电脑可持续显示手机画面，建议至少观察 10 分钟并记录卡顿情况。
- [ ] 在普通界面完成点击、长按和滑动测试。
- [ ] 手机连接小车时，USB 投屏与控制仍可同时使用。
- [ ] 小车 App 视频区域能在电脑显示。
- [ ] 电脑可发出所需操作，且松开方向键后小车按预期停止。
- [ ] 已验证断线、失焦和退出时的行为，并明确人工停止方式。

## 参考资料

- scrcpy 官方项目与前置要求：https://github.com/Genymobile/scrcpy
- Windows 安装：https://github.com/Genymobile/scrcpy/blob/master/doc/windows.md
- 交互控制：https://github.com/Genymobile/scrcpy/blob/master/doc/control.md
- 视频参数：https://github.com/Genymobile/scrcpy/blob/master/doc/video.md
- Android 设备调试：https://developer.android.com/studio/run/device

建议实施顺序：先完成普通手机界面的 USB 投屏与点击验证，再联调小车 App，最后评估程序自动点击或原生接口开发。
