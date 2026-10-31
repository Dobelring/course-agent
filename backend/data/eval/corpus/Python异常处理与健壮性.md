# Python 异常处理与程序健壮性

## 异常体系

Python 用异常对象表达运行时错误，所有异常继承自 `BaseException`，业务异常应继承 `Exception`。

常见内置异常：
- `ValueError`：值的类型正确但取值不合法（如 `int("abc")`）
- `TypeError`：类型不匹配（如字符串与整数相加）
- `KeyError` / `IndexError`：字典键不存在、列表下标越界
- `ZeroDivisionError`：除数为零
- `StopIteration`：迭代器耗尽（生成器协议依赖它）

自定义异常：继承 `Exception` 后 `raise` 抛出，可携带结构化字段供上层统一处理。

## try / except / else / finally

```python
try:
    data = int(raw)
except ValueError:
    print("输入不是合法整数")
except (TypeError, KeyError) as e:
    print(f"类型或键错误：{e}")
else:
    print("没有异常时才执行，可放成功分支")
finally:
    print("无论是否异常都执行，用于释放资源")
```

要点：
- `except` 应捕获**具体类型**，裸 `except:` 会把 `KeyboardInterrupt` 一并吞掉，掩盖真实错误
- `else` 只在 try 块**没有**异常时执行，常用来把"可能出错的语句"与"依赖其成功的语句"分开
- `finally` 无论正常、异常、甚至 try 中有 `return` 都会执行，是清理资源的兜底

## 主动抛出与异常传播

- `raise ValueError("半径不能为负")`：主动触发校验，错误信息写清楚上下文
- `raise`：在 `except` 中不带参数重抛，保留原始 traceback，避免吞掉调用链信息
- 异常传播：函数内未捕获的异常沿调用栈向上传，直到某个 `except` 接住或进程终止
- 设计原则：谁能把错误转换成有意义的决策，谁来捕获；只捕获能处理的，处理不了就抛给上层

## 断言与防御性检查

`assert condition, message` 用于开发期不变量检查，被 `python -O` 优化时移除——**不要用 assert 做用户输入校验**，正式校验要用 `if + raise`。

## 上下文管理与资源安全

异常发生时若跳过清理代码，文件句柄、网络连接会泄漏。两条防线：
1. `try/finally`：显式保证清理段执行
2. `with` 语句：进入时获取资源、退出时（无论是否异常）自动关闭，写法更短且不易遗漏

异常处理的常见反模式：
- 捕获后什么都不做（`except: pass`）——故障被静默隐藏，排查成本翻倍
- 捕获后包装成更泛的异常重新抛出却不保留原因链
- 用异常做普通控制流——`if` 判断比抛异常快几个数量级
