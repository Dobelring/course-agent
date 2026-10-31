# Python 文件与模块组织

## 文件读写

`open()` 打开文件，常用模式：`"r"` 读（默认）、`"w"` 写（覆盖）、`"a"` 追加、`"b"` 二进制。文本模式务必显式指定 `encoding="utf-8"`，否则结果依赖操作系统默认编码，在 Windows 上中文容易乱码。

```python
with open("data.txt", encoding="utf-8") as f:
    text = f.read()
```

推荐用 `pathlib.Path` 替代 `os.path` 拼接：`Path("data") / "out.txt"` 跨平台安全，`.read_text()` / `.write_text()` 直接带编码。

## 结构化数据读写

JSON 与文件互转是常见需求：

```python
import json

# 字典 → 文件
with open("config.json", "w", encoding="utf-8") as f:
    json.dump(info, f, ensure_ascii=False, indent=2)

# 文件 → 字典
with open("config.json", encoding="utf-8") as f:
    info = json.load(f)
```

- `dump`/`load` 对**文件对象**，`dumps`/`loads` 对**字符串**
- `ensure_ascii=False` 让中文按原样写出，否则会被转成 `\uXXXX`
- CSV 用标准库 `csv` 模块：`csv.reader` / `csv.writer`，自动处理逗号转义

## 模块导入机制

- `import math` 导入整个模块；`from math import sqrt` 导入单个名字
- `import numpy as np` 起别名，命名冲突时的常规做法
- 模块查找顺序：当前目录 → 标准库 → 第三方包（`sys.path` 可查看）
- 模块顶层代码在**首次导入时**执行一次，之后导入复用 `sys.modules` 中的缓存

## 入口文件与包结构

每个 Python 文件都有内置变量 `__name__`：
- 被导入时为模块名
- 被直接执行时为 `"__main__"`

因此脚本惯用入口写法：

```python
def main():
    ...

if __name__ == "__main__":
    main()
```

作用：被 `import` 时不自动执行主体逻辑，被 `python app.py` 直接运行时才启动。

包（package）是含 `__init__.py` 的目录，支持 `from package.sub import module` 的层级导入。工程上按职责拆分子模块，避免单文件膨胀。

## 依赖与环境

- `pip install requests` 安装第三方库，`requirements.txt` 锁定版本
- 虚拟环境（`venv`）为每个项目隔离依赖，避免全局污染
- 原则：标准库优先，第三方依赖引入前评估维护活跃度与安全记录
