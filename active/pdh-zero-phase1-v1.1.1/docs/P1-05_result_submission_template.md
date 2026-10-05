# P1-05 发回分析的结果格式

每完成一个实验编号，发送以下内容：

```text
实验编号：A01_AMD_SYSTEM
执行时间：
使用环境：GPU
镜像名称：
额度：100h00m -> 99h52m

执行命令：
<完整命令>

退出码：

结果文件：
- <run>.zip
- <run>.zip.sha256

关键输出：
<粘贴 summary.json；不要只发截图>

stderr：
<无则写 empty>

页面截图：
<额度和实例状态截图>

个人观察：
<只写观察，不写未经验证的结论>
```

必须上传结果 ZIP。若平台限制上传，至少提供：

- `manifest.json`；
- `summary.json`；
- `stdout.log`；
- `stderr.log`；
- `checksums.sha256`。
