import pandas as pd
from energy_modeling.features import add_calendar_features, aggregate_to_hourly, complete_hourly_index


def test_hourly_aggregation_and_completion():
    df = pd.DataFrame(
        {
            "localtime_hour": pd.to_datetime([
                "2022-01-01 00:00:00",
                "2022-01-01 00:00:00",
                "2022-01-01 02:00:00",
            ]),
            "kilowatt_hours": [1.0, 2.0, 4.0],
            "voltage_avg": [230.0, 231.0, 235.0],
        }
    )
    hourly = aggregate_to_hourly(df, "localtime_hour", "kilowatt_hours")
    complete = complete_hourly_index(hourly, "kilowatt_hours")
    assert len(complete) == 3
    assert complete.loc[0, "kilowatt_hours"] == 3.0
    assert complete.loc[1, "kilowatt_hours"] == 0.0


def test_calendar_features():
    df = pd.DataFrame({"timestamp": pd.to_datetime(["2022-01-01 12:00:00"]), "kilowatt_hours": [5.0]})
    out = add_calendar_features(df)
    assert out.loc[0, "hour"] == 12
    assert out.loc[0, "is_weekend"] == 1
    assert "hour_sin" in out.columns
