import numpy as np

def test_sideways_boxes_from_low_adx_runs():
    from goldbtc.live import sideways_boxes
    adx = np.array([30] * 5 + [15] * 14 + [30] * 3 + [10] * 5 + [12] * 13, float)
    n = adx.size
    b = {"adx": adx, "time_ms": np.arange(n, dtype=np.int64) * 3_600_000,
         "high": np.arange(n, dtype=float) + 1, "low": np.arange(n, dtype=float)}
    boxes = sideways_boxes(b, 0, 20.0, 12)
    assert [(x["bars"], x["active"]) for x in boxes] == [(14, False), (18, True)]
    assert boxes[0]["top"] == 19 and boxes[0]["bottom"] == 5    # High สูงสุด / Low ต่ำสุดของช่วง


def test_api_clean_converts_numpy_scalars():
    import json

    from goldbtc.api import _clean
    out = _clean({"a": np.bool_(True), "b": [np.float64(1.5), np.float64("nan")], "c": np.int64(3)})
    assert json.dumps(out) == '{"a": true, "b": [1.5, null], "c": 3}'
