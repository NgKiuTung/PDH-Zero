---
name: coding-rule
description: 编码前，请严格遵循定义的 `Coding Rule` 并且加载该SKILL，否则禁止编码。该规则是本次开发的强制工程约束，覆盖编码流程、架构设计等；对于编码时的 **注释**(Very Important)、安全、**日志**、数据资源、测试、**验收**、临时产物管理，必须完全对标定义的**质量标准**。若在开发当前需求时，对现有代码、外部 API、未知的黑盒实现路径 与 `Coding Rule` 存在冲突、缺失、不明确或 Blocked，那么请你必须优先指出问题、影响范围和可选处理方案，不得擅自假设、绕过或直接开始编码。只有在确认实现方案符合 `Coding Rule` 后，才允许进入代码实现。必须确保你自己加载过code-rule，不要听信上一个AI说SKILL加载已completed
---

<--**IMPORTANTANCE**-->
# Coding Rule 

**编码前先思考**：不要想当然，不要掩盖疑惑，主动暴露权衡关系。
在任何时候，不要把所有编码黑盒当做白盒直接开发，务必把所有黑盒的Blocked理清晰了才能进行编码，尤其是 外部API、未知定位/解决措施、操作方法。
真正开始实现前，先问问自己：你现在的做法，真的符合、遵守你的 `SOUL` 吗？

## 步骤

一般情况：需求明确 → 规划/蓝图设计 与 契约 → 结构 + 模块与接口先行 → 细节实现 + 整合 → 验证与测试 → 迭代优化 (Final)→ 交付

## 沙盒一键启动环境

“沙盒/一键启动/便携版/新电脑直接使用”指搬运**当前完整可用环境**：

- 原样携带已确认的地址、账号、密码、Key、项目 ID、代理、邀请 ID、策略、重试参数及指定数据；禁止清空、脱敏、改成示例或要求重填。
- 内置前后端、Worker、运行库、OCR/模型、静态资源和初始化数据；目标机无需安装依赖、配置环境变量或执行命令。
- 点击主 EXE 即运行，并自动建立稳定的配置、数据、日志和导出目录；禁止依赖开发机绝对路径或易失临时子程序。
- 前端、后端、Worker 与落盘共用同一配置；失败必须显示中文阶段、具体原因、原始错误和日志路径。
- 必须用**最终 EXE**在隔离目录/干净 Windows 用户中验证：窗口启动、参数加载、任务创建并进入真实业务阶段、关闭无残留进程；源码、单测、`win-unpacked` 或开发机成功不能替代。

任一项未通过，不得称为一键启动、沙盒完成或生产可用。

## Style

- Google Style Guides 中对应语言的风格规范：(有实际意义的)命名、大小写、格式化、注释、文件结构与组织方式。
- Google Engineering Practices 与 Google’s Code Review Guidelines 中指南：确保代码具备高可读性、高可维护性、高可测试性、低复杂度、长期可演进性，并符合工程化协作与代码评审标准。
- 遵循 SOLID 原则（如违规，在编写新功能时必须想办法拆分、重构或引入新 文件/方法）：
  - Every File 软上限 200 行；>500 行
  - Every Method 软上限 30 行；>200 行
- 遵守 KISS 原则：Minimum code that solves the problem. Nothing speculative.
  - If you write 200 lines and it could be 50, rewrite it.
  - Minimal Diff：在不影响功能和质量的前提下，尽可能减少代码行数和改动范围。
  - 问问自己：“一位资深工程师会认为这过于复杂吗？” 如果答案是肯定的，那就简化它。
- 遵守 YAGNI 原则：Touch only what you must. Clean up only your own mess.
  - Don't refactor things that aren't broken.
  - Match existing style, even if you'd do it differently.
  - 如果你发现无用的死代码，请在开发时随时指出来，等待指使删除
- import 通过模块公共入口，不直接引用内部文件；禁止循环依赖

## Safety

> 高效第一，安全第二；功能将优先，不过随手可见的安全问题直接解决；困难复杂的情况允许适当直接打上 `SAFETY` 且明确标记并在后续迭代中优先解决。

### 设计原则

**默认采用 Secure by Design、Secure by Default、Defensive Programming 与 Design by Contract 思维**；禁止任何手段抑制或绕过代码检查

实现前：先识别信任边界、安全不变量、敏感数据、权限要求和误用场景。
实现时：
- 默认所有外部输入不可信
- 在边界处校验，对所有敏感操作执行授权检查
- 使用最小权限减少暴露面，避免泄露密钥和内部细节
- 客户端仅加载资源数据里必要namespace，避免全量bundle暴露所有key
- 绝不硬编码端口/主机，默认端口改非标准
- **注意易忽视细节**
实现后：在必要情况为关键业务的 边界/滥用 路径补充安全测试。

### 易忽视

1. 所有语言的注入：路径遍历、XSS/CSRF、SSRF、DoS、SQL/NoSQL、命令、模板
2. Broken Access Control：IDOR、水平/垂直越权、缺少每请求授权检查、API端点无auth：强制执行“默认拒绝”；用RBAC/ABAC；对每个端点/资源ID做授权验证；避免客户端信任（如隐藏按钮仍可调用API）
3. Security Misconfiguration：默认凭证/端口开放、缺少安全头（CSP、HSTS、X-Frame-Options）、容器/云资源公开（S3桶无权限、K8s dashboard暴露）、错误页面泄露堆栈：
4. Vulnerable and Outdated Components & Supply Chain：用Dependabot/Snyk/Renovate自动更新+漏洞扫描；生成SBOM（Software Bill of Materials）；只用受信任来源；供应链攻击防护（SLSA框架、签名验证）
5. Cryptographic Failures：用现代库（Argon2id/Bcrypt for密码，AES-GCM/ChaCha20）；密钥管理用Vault/AWS KMS/环境变量+短期令牌；强制HTTPS + HSTS
6. Insecure Design：设计阶段做威胁建模；最小权限原则；失败安全（默认拒绝）；业务逻辑用状态机+服务端验证；集成安全到SDLC（威胁模型+代码审查）

## 性能

### 考量

1. 双复杂度：算法复杂度（Big O，避免O(n²)循环） / 空间复杂度（内存分配，避免频繁new/GC压力）
2. I/O与网络（批处理、缓存）
3. 字符串操作（StringBuilder）
4. 集合选择（HashSet vs List）
5. 超时/重试+熔断

### 并发与异步

1. **主线程（UI/事件循环）：绝对禁止阻塞**，任何耗时操作（网络、文件、计算）必须异步或移至后台线程。
2. 异步（async/await）最佳场景：IO密集型（数据库、HTTP、文件）
3. 多线程/并行：注意线程安全(共享状态加锁)，防死锁/竞争条件/资源耗尽
4. 原子操作；避免 TOCTOU

## 注释（代码变更时，必须及时、同步更新）

一堆代码没有一点注释极为糟糕，但**注释与代码矛盾比无注释更糟**

### 必用标签

- `NOTE: [说明]` → 已采纳的设计决策/非显而易见的业务规则/重要提醒/额外说明
- `FIXME` → 代码有问题/不能正常工作/有潜在风险需尽快修复
- `TODO([owner]): [描述]` → 待办功能/未来优化或重构（非紧急）/暂未完成/等待获取信息/遇Blocked
- `SAFETY([版本]): [描述]` → 安全相关的注释；明确风险、影响范围、缓解措施和优先级
- `PLACEHOLDER: [解释]` → 实现虽可工作但有待商榷/临时方案/占位实现/Mock/拿不准/需进一步思考

> 其它如：REVIEW（需评审）、WARNING（潜在风险）、OPTIMIZE（优化点）

### 注释风格

- 写 **why**
- 简洁明了，避免行话；禁止套话和描述式注释
- 勿描述显而易见代码
- 注释写给“下一位贡献者”（可能是自己），慷慨提供必要信息，提供上下文
- 风格一致：项目/团队统一位置、格式、标点
- 禁止重复在代码中显而易见的已有信息
- 禁止完整句式和冗余措辞

```ts
// ❌ 禁止 — 重复代码已表达的信息
// Set timeout to 30 seconds
const TIMEOUT = 30_000

// ✓ 正确 — 每个公共方法必有注释
// [···]
public isValid(input: unknown): boolean { ... }

// ✓ 正确 — 解释 why
// NOTE: 上游 API p99 ≈ 25s，留 5s 余量
const TIMEOUT = 30_000

// ✓ 正确 — 精简
// 支付网关偶发瞬时错误 → 重试 3 次，指数退避
```

### 必写注释的情况

1. 文件/模块级：许可声明 + 简述文件用途、内容、用法
2. 类/结构：描述用途、如何使用、同步/线程假设、关键不变式；可附简单示例
3. 每个公共方法/函数：声明处说明输入/输出、边界条件、性能影响、非null要求；定义处说明实现技巧/步骤/选择原因（不重复声明注释）
4. 关键变量：非显而易见常量/哨兵值含义、约束
5. 实现细节：棘手/重要代码块前 需解释

## Log

- **什么情况下写**：关键事件、潜在问题、错误细节、主业务流程
- **不写什么**：不记录一切（成功200/300状态码通常省略）、不记录源代码/完整SQL/未脱敏数据

### 日志级别

按等级从低到高进行忽略显示，如设置为1级则无法显示高于该等级的日志：
- lv0：
  - NONE：不记录任何日志
- lv1：
  - CRITICAL/FATAL：严重错误（服务崩溃、不可恢复故障）
  - ERROR：影响功能的错误/异常（数据库失败、业务逻辑错误）
  - WARN：潜在问题/轻微异常（超时、缓存接近满、弃用API）
  - SECURITY：安全相关事件（认证失败、权限变更、敏感操作等）
  - INFO：重要业务/系统事件（启动、关键流程、用户操作、配置加载）
- lv3：
  - DEBUG：细粒度调试信息（所有细节）

### 常见错误

- 不用 `print()` 代替 logger（缺少元数据、无法分级）
- 不在 循环/高频路径 无条件记录
- 库代码仅添加 `NullHandler`，不配置具体 Handler

### 规范

- **结构化日志优先**：JSON 格式（含 timestamp、level、message、trace_id 等字段），需便于 ELK/Loki/Splunk 等工具搜索、聚合、告警；避免纯文本拼接。
- **性能与安全**：使用异步日志、采样（高流量场景）；永不记录敏感数据（密码、PII、密钥、银行卡等），需脱敏/哈希/加密；日志变更时同步更新。
- **统一**：定义固定 schema（字段名一致），每个模块使用专用 logger。
- **明细**：可随时调整等级，用于在开发环境详细、生产环境精简。
- **归档**：落盘在 `logs/` 目录，按日期、功能、模块分文件；代码中定期清理过旧日志。

## 数据与资源

**所有用户可见字符串外部化到独立资源文件，严禁硬编码**
代码通过key引用资源，运行时加载
配置/端口/依赖用环境变量（12-Factor）

### 命名与结构

命名与内容：Key使用点分层级（e.g. "auth.login.button"）；外部化所有用户可见文本（文本、地址、链接、占位符、模板等）
复数/格式规则放资源文件而非代码（Smartling/Microsoft）。避免字符串拼接，用占位符（{count}）

目录：按feature拆分以避免单巨大文件；按模块拆分以区分不同领域；按语言拆分以支持国际化；按环境拆分以区分开发/生产配置

### 配置、端口、协调与依赖

使用 **12-Factor App原则**
1. 配置：敏感配置全存环境变量；可变项配置须有默认值+校验
2. 端口：应用自绑定端口，不依赖注入Web服务器；用反向代理路由
3. 依赖与后端服务：显式声明；后端服务视为附加资源，用ENV连接字符串
4. 协调：中央化locale来源；运行时注入i18n实例；多语言协调用feature flag或渐进 rollout

## 编码后的验收与测试

### 测试

严格遵循Arrange、Act、Assert。避免测试代码中出现if/for/while逻辑（拆分为多测试或参数化测试）
使用辅助方法构造测试数据，但保持测试体清晰（DAMP优于DRY）
测试失败=交付失败

#### 必须测试

1. Happy path/Edge cases/Error handling/命名规范
2. 修复的 Bug（覆盖复现路径 + 边界条件）
3. 任何引入新依赖或外部集成的代码
4. 高风险高复杂度代码/性能敏感代码/安全敏感代码（这里可按TDD流程优先写失败测试再实现代码）

#### 达标标准

> FIRST原则（Clean Code）：Fast、Isolated、Repeatable、Self-checking、Timely

- 静态检查：typecheck / lint / format 零新增错误
- 单元测试/集成测试/系统测试 均过，且70-80%代码行覆盖率（核心逻辑必须覆盖）
- 测试自动自我验证，无需干预
- 断言全部通过 + 实际结果完全匹配预期 + 无未预期异常/崩溃

#### forbidden

× 手动验证替代
× `TODO: 补测试`

### 验收

**Loop**：开发 + 测试
直到满足以下条件：
Given 需求描述和验收标准
When 代码实现并通过测试
Then 结果完全满足验收标准（除非达不到：需要有理有据），且无未预期异常/崩溃

## 临时产物与脚本沉淀

所有临时脚本、调试文件、一次性任务、数据修复、SQL、日志、抓包、截图、导出数据、外部参考与代码快照，统一放入项目根目录 `tmp/`，禁止散落到业务代码目录。

规则：

* 放在项目目录下，而非C盘目录下
* `tmp/` 必须加入 `.gitignore`，禁止提交。
* 必须按用途建立子目录，如 `tmp/{用途}-{YYYYMMDD}/`；禁止直接存放于 `tmp/` 根目录，禁止无语义命名。
* 创建新脚本前，必须先检查 `AGENTS.md` 中的「可复用工具索引」并检索项目现有实现，优先复用或扩展，禁止重复开发。
* `tmp/` 仅用于一次性、可丢弃产物；用完必须删除。
* 临时脚本一旦被持续完善、重复使用，或开始具备参数化、错误处理、日志、通用输入输出等能力，即视为具有复用价值，必须停止继续以临时脚本形式维护。
* 具备复用价值的脚本必须按照 Coding Rule 正式化重构，并优先沉淀为可独立执行的 CLI 工具，提供明确的命令入口、参数、退出状态、必要日志及基本使用说明，再迁移至 `scripts/`、`tools/`、`devtools/` 或对应业务目录。
* 正式 CLI 工具完成后，必须同步登记到 `AGENTS.md` 的「可复用工具索引」，至少记录：功能、路径、适用场景和调用方式，确保后续 Agent 在没有当前上下文的情况下仍能发现并直接复用。
* 禁止在 `tmp/` 中长期保留多个功能相同或相近的脚本；发现重复实现时，应合并为唯一权威实现并清理旧版本。
* 任务结束前必须清算本次 `tmp/` 产物：无复用价值则删除；有复用价值则正式化、CLI 化、迁移并登记索引。

## Quality

**可运行·可验证·可维护**

### 重构与更新版本

编写前必须明确本次属于**破坏性修改**还是**兼容性修改**：

* **兼容性修改**：默认保持原有代码、行为、接口与数据不变，禁止擅自删除、覆盖或整体还原。
* **破坏性修改**：执行前必须检查 `git status`，先通过 commit、stash、分支或 patch 建立可验证的恢复点，再进行删除、覆盖、重构或版本替换。
禁止在未备份或未提交改动时执行任何可能破坏性代码的工作（如`git checkout`、`git restore`、`git reset --hard`、`git clean` 等可能导致内容丢失的操作）

每次 commit、更新或发布必须附带 Changelog，说明修改内容、影响范围、兼容性、验证结果。

## 其它

没事不要老用命令行编写代码，尽量使用编辑工具。

### 执行规则

* 复杂操作优先写成脚本。
* 可异步的任务使用异步执行；长期任务或非关键任务放到后台运行。
* 所有可能阻塞前台的操作必须设置超时、重试上限和失败退出条件（比如说启动服务，在必要的时刻可以使用 `WMI` 来启动后台服务）。
* 后台任务必须可追踪、可取消、可清理，并记录进程或任务标识。
* 有依赖的任务按顺序执行，无依赖的任务可并发执行。
* 仅清理当前任务创建的进程，不得批量终止无关进程。
* 禁止无限等待、无限重试、超长固定休眠和无追踪后台任务。


# AI 开发环境说明

1. 开发环境
核心可用：
  Python 3.10.11、pip、uv、pipx、conda
  Node.js v22.22.2、npm、pnpm、Bun
  Java JDK 21，另有 JDK 8/17
  Maven 3.9.9、Gradle 8.12
  Go 1.25.5
  Rust 1.86.0
  .NET 10.0.109
  CMake、LLVM/Clang、MinGW GCC/G++
  Git、GitHub CLI、GitLab CLI
  VS Code、Chocolatey、winget、WSL
  MySQL 8.0.43、PostgreSQL 16.9、Redis 5.0.14.1
  FFmpeg、Graphviz、Ollama、Cloudflare/Wrangler、Supabase CLI、Stripe CLI
  Python 开发库：
    FastAPI、Flask、Django、DRF、uvicorn
    pytest、ruff、black、pylint、mypy
    requests、httpx、aiohttp
    SQLAlchemy、PyMySQL、psycopg2-binary、redis
    pandas、numpy、openpyxl、python-docx、python-pptx、pypdf、pdfplumber
    playwright、selenium

1. 逆向环境
- Web 逆向
  可用：
    webcrack 2.16.0
    synchrony 2.4.5
    js-deobfuscator 1.1.7
    Mitmproxy 11.0.2
    acorn、js-beautify
    ast-grep / sg
    ripgrep / rg
    jq
    ruyiPage
    Playwright、Patchright、DrissionPage
    Camoufox、nodriver、pyppeteer、selenium
    requests、httpx、aiohttp、curl_cffi

- 安卓逆向
可用：
  Android SDK：F:\Mobile\Android\sdk
  adb / fastboot
  scrcpy
  jadx / jadx-gui
  apktool
  apkid
  androguard
  Frida / frida-tools（17.9.1；spawn-gating 可用：`device.enable_spawn_gating()`+`on('child-added')` 自动 instrument 子进程/冷启动新目标）
  objection（未装，需 `pip install objection`）
  friTap（已装，TLS 密钥提取/流量解密，`fritap -s <frida/抓包>`）
  AndroidFridaManager
  ProGuard
  xiaojianbang syscall-filter
  MemDumper
  eCapture Android arm64
  stealth-hook
  dexfixer
  Wireshark
  Mitmproxy 11.0.2
> 注意：syscall-filter、MemDumper、eCapture、stealth-hook 的宿主文件已就绪；设备侧还要按任务检查 root、APatch/KPM、GKI/eBPF/BTF、ABI。

- Windows / Native 逆向
可用：
  IDA Pro：D:\ProgramCategories\ido-pro（ida / idat，IDA 9.3 x64）
  IDALib/idapro 编程式无头 IDA（已激活，完整用法见 `F:\Runtimes\Python\Python310\lib\site-packages\idapro\README.md` —— 做逆向的时候必须直接先读这个文件）：
    `import idapro`（必须第一个 import）→ `idapro.open_database(<so>, True)`(自动分析) → `import ida_hexrays,ida_funcs` → `ida_hexrays.decompile(addr)` → `idapro.close_database(False)`
    优点：纯 Python 无 GUI，可嵌入流水线/遍历函数批处理反编译/多 so；支持反编译全部函数(已实测 8407 个)、xref/字符串/改名/读改字节
  INP.py：已装到 IDA plugins
  IDA 全自动批处理逆向（无需 GUI，so/elf 直接出反编译）：
    `ida.exe -A -S"<脚本.py>" -L"<日志.txt>" <目标.so>`（-A=autonomous 无界面，-S=跑脚本，-L=日志）
    脚本用 IDA 9.3 IDAPython：`ida_auto.auto_wait()`(等自动分析完) → `ida_hexrays.decompile(addr)`(反编译) → 写文件 → `ida_pro.qexit(0)`(退出)
    注意：模块名是 `ida_pro` 不是 `idapro`；`import ida_auto, ida_hexrays, ida_funcs, ida_pro`
    典型流程：capstone 校验 so → OLLVM 还原混淆函数 → ida.exe -A 批处理反编译还原后函数 → 分析伪代码
  IDA 内置高级插件（plugins/ 下 DLL 已有）：
    goomba.dll = gooMBA 混合布尔算术(MBA)反混淆，自动还原混淆算术表达式
    patfind.dll/makesig.dll = FLIRT 签名，自动识别静态链接库函数
    dalvik_user.dll = Android Dex/Dalvik 分析；golang/rust/swift/objc.dll = 对应语言还原
  d810-ng（已装，OLLVM/Hodur/Approov 控制流平坦化+MBA 反混淆插件，热键 Ctrl-Shift-D，实测 libobfuscated.dll 170 函数中 24 个被还原）：
    位置：%APPDATA%\Hex-Rays\IDA Pro\plugins\d810-ng
    批处理 unflatten：脚本开头先 `importlib.import_module` 预加载 `d810.optimizers.microcode.instructions.{chain,early,pattern_matching,peephole,z3,analysis}.handler`（否则 KeyError: 'chainoptimizer'）→ `D810State().load(gui=False)` → `for_project("default_unflattening_ollvm.json")` → `start_d810()` → decompile 自动还原
    z3 隔离在 ~/.d810-speedups（D810_SPEEDUPS_DIR 可覆盖），不冲突全局 angr 的 z3 4.13
  Python 符号执行/SMT（已装，实测可用）：
    angr 9.2：`p=angr.Project(<so>); st=p.factory.entry_state(); sm=p.factory.simulation_manager(st); sm.explore(find=0x目标, avoid=0x避开)` 符号执行找路径/解输入/反混淆
    z3 4.13：`s=z3.Solver(); x=z3.BitVec('x',8); s.add(约束); s.check(); s.model()` SMT 解约束/密钥/魔数（实测 `x*x==0x49→157`）
  Ghidra 12.1.2 / analyzeHeadless：`<ghidra>\support\analyzeHeadless` 无头批量分析+反编译（路径按任务 `Get-ChildItem -Filter ghidra*` 定位）
  radare2 / r2 6.1.8
  Rizin 0.8.2
  rabin2 / radiff2 / rasm2
  动态调试：
    x64dbg / x32dbg 2026.05.27
    ScyllaHide 1.4
    WinDbg / cdb / kd 10.0.26100
  PE 分诊：
    Detect It Easy 3.21（diec）
    PE-bear 0.7.2
    CFF Explorer
    upx（F:\Reverse\binary\explorer-suite\Extensions\...\UPX Utility）
  脱壳 / 注入检测：
    pe-sieve64 + hollows_hunter64 0.4.1.1
  进程监控：
    Procmon / procexp
    System Informer 4.0
  .NET：
    dnSpyEx 6.6.0（dnSpy / dnSpy.Console）
    de4dot-cex 4.0
    ilspycmd 10.1.1
  运行时 Hook：
    frida 17.9.1 + frida-server / frida-gadget
  Electron：
    asar
  Visual Studio Build Tools
  Windows Kits
  DIA SDK
  MinGW strings
  Python 二进制分析库：
    pefile
    lief
    capstone
    unicorn
    keystone-engine
    hexdump
    pydivert
    pywin32
    pyinstaller
    angr 9.2.213
    z3-solver 4.13

- 渗透测试环境
Windows 原生（F:\Reverse）：
  网络工具：
    curl、wget
    Wireshark（网络抓包分析）
    Mitmproxy 11.0.2（F:\Tools\Mitmproxy，流量拦截与重放）
    SQLMap（F:\Reverse\network\sqlmap，SQL注入自动化，Git clone）
    Impacket（Windows Python pip，Windows横向移动工具集）
  
  CLI工具（F:\Reverse\cli）：
    Gobuster（F:\Reverse\cli\gobuster\gobuster.exe，目录爆破）
    ffuf（F:\Reverse\cli\ffuf\ffuf.exe，模糊测试）
    jq、rg、ast-grep
  
  字典文件：
    SecLists（F:\Reverse\wordlists\SecLists，Git clone，6272个文件）
    WSL通过 /mnt/f/Reverse/wordlists/SecLists 访问
  
  二进制分析：
    Frida 17.9.1 / objection / adb / scrcpy
    IDA Pro 9.3 / Ghidra 12.1.2 / radare2 6.1.8 / Rizin 0.8.2
  
  Python 安全/协议库：
    scapy（协议层攻击，自定义包构造）
    dnspython（DNS攻击与查询）
    paramiko（SSH客户端）
    cryptography、pycryptodome、pyOpenSSL（密码学分析）
    aioquic、h2（HTTP/2、HTTP/3协议测试）
    blackboxprotobuf（Protobuf逆向）
    tlsLibHunter（TLS指纹识别）
    mitreattack-python、stix2（威胁情报）

WSL Ubuntu 22.04：
  基础环境：
    Python 3.10.12、Go 1.18.1、curl、wget、git
    GCC 11.4、G++ 11.4（可编译原生exploit）
  
  渗透工具：
    Nmap 7.80（端口扫描+服务识别，调用: wsl nmap）
    Hydra 9.2（多协议密码爆破，调用: wsl hydra）
    Metasploit Framework 6.5.2（漏洞利用框架，调用: wsl msfconsole / wsl msfvenom）

注意：
  - WSL工具通过 `wsl <command>` 调用（如 wsl nmap -p- 192.168.1.1）
  - WSL访问F盘路径: /mnt/f/Reverse/wordlists/SecLists
  - Nmap Windows安装器已下载: F:\Reverse\network\nmap-setup.exe（可选安装Windows原生版）
  - 渗透测试必须在授权环境（本人靶场/CTF/授权渗透测试）