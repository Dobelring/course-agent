"""评测结果持久化：按 suite 合并、坏文件恢复、覆盖写。"""
import json
import sys
from pathlib import Path

# scripts/ 不是包，手动入 path（eval_report 无副作用）
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from eval_report import merge_results  # noqa: E402


def test_single_suite_write(tmp_path: Path):
    out = tmp_path / "results.json"
    merge_results(out, {"retrieval": {"hit_at_1": 1.0}})
    assert json.loads(out.read_text(encoding="utf-8")) == {"retrieval": {"hit_at_1": 1.0}}


def test_second_suite_does_not_overwrite_first(tmp_path: Path):
    """核心语义：单跑 suite 不覆盖其他 suite 的历史结果（修复前的 bug）。"""
    out = tmp_path / "results.json"
    merge_results(out, {"retrieval": {"hit_at_1": 0.9667}})
    merge_results(out, {"tool": {"tool_accuracy": 1.0}})
    merged = json.loads(out.read_text(encoding="utf-8"))
    assert merged["retrieval"]["hit_at_1"] == 0.9667   # 保留
    assert merged["tool"]["tool_accuracy"] == 1.0


def test_same_suite_updates_in_place(tmp_path: Path):
    out = tmp_path / "results.json"
    merge_results(out, {"retrieval": {"hit_at_1": 0.5}})
    merge_results(out, {"retrieval": {"hit_at_1": 1.0}})
    merged = json.loads(out.read_text(encoding="utf-8"))
    assert merged == {"retrieval": {"hit_at_1": 1.0}}  # 同 suite 刷新而非累积


def test_corrupt_file_recovers(tmp_path: Path):
    """results.json 损坏 → 重建而不是崩溃。"""
    out = tmp_path / "results.json"
    out.write_text("{{{ broken", encoding="utf-8")
    merge_results(out, {"plan": {"completion_rate": 1.0}})
    assert json.loads(out.read_text(encoding="utf-8")) == {"plan": {"completion_rate": 1.0}}


def test_creates_missing_parent_dir(tmp_path: Path):
    out = tmp_path / "deep" / "nested" / "results.json"
    merge_results(out, {"x": 1})
    assert out.exists()
