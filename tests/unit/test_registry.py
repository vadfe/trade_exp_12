from trade_exp_12.registry import DatasetRecord, Registry


def test_registry_round_trip(tmp_path) -> None:
    registry = Registry(tmp_path / "registry.db")
    registry.initialize()
    record = DatasetRecord(
        symbol="ADAUSDT",
        timeframe="15",
        version="v1",
        path="data.parquet",
        first_open_time="2026-01-01T00:00:00+00:00",
        last_open_time="2026-01-01T00:15:00+00:00",
        row_count=2,
        gap_count=0,
        checksum="abc",
        created_at="2026-01-01T01:00:00+00:00",
    )

    registry.add_dataset(record)

    loaded = registry.latest_dataset("ADAUSDT", "15")
    assert loaded is not None
    assert loaded["row_count"] == 2

    processed = DatasetRecord(
        symbol="ADAUSDT",
        timeframe="15",
        version="processed-v1",
        path="processed.parquet",
        first_open_time=record.first_open_time,
        last_open_time=record.last_open_time,
        row_count=2,
        gap_count=0,
        checksum="def",
        created_at="2026-01-01T02:00:00+00:00",
    )
    registry.add_processed(processed, source_version="v1", feature_set="relative_v1")

    loaded_processed = registry.latest_processed("ADAUSDT", "15")
    assert loaded_processed is not None
    assert loaded_processed["source_version"] == "v1"
    assert loaded_processed["feature_set"] == "relative_v1"
