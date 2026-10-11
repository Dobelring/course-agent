"""向量存储格式：float32 BLOB 编解码往返、脏数据容错、旧 JSON 兼容。"""
import json

import numpy as np

from app.rag.vecfmt import decode_json_vec, decode_vec, encode_vec


def test_roundtrip_preserves_values():
    vec = [0.1, -0.25, 3.75, 1e-6]
    restored = decode_vec(encode_vec(vec))
    assert np.allclose(restored, vec, atol=1e-6)


def test_blob_size_is_smaller_than_json():
    """float32 BLOB 体积显著小于 JSON 文本（存储收益的回归防线）。"""
    vec = [0.123456] * 512
    assert len(encode_vec(vec)) == 512 * 4          # 2048 字节
    assert len(json.dumps(vec)) > 2 * 2048          # JSON 文本 > 2 倍


def test_decode_dirty_data_returns_none():
    assert decode_vec(None) is None
    assert decode_vec(b"") is None
    assert decode_vec(b"not-bytes") is None          # 长度非 4 倍数
    assert decode_json_vec(None) is None
    assert decode_json_vec("") is None
    assert decode_json_vec("{bad json") is None
    assert decode_json_vec("[]") is None


def test_decode_json_vec_legacy_format():
    assert decode_json_vec("[0.5, 0.5]") == [0.5, 0.5]
