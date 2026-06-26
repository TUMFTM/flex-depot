import pandas as pd

from flex_dep_opt.market.fcr import droop_signal, fcr_gate_closure_timestamp

_KW = dict(nominal_hz=50.0, deadband_hz=0.010, full_activation_hz=0.200)


def test_droop_sign_and_deadband():
    # delta_f:        0     +0.005   -0.1    +0.1    -0.02
    freq = pd.Series([50.0, 50.005, 49.9, 50.1, 49.98])
    d = droop_signal(freq, **_KW)

    assert d.iloc[0] == 0.0  # at nominal -> 0
    assert d.iloc[1] == 0.0  # inside deadband -> 0
    assert abs(d.iloc[2] - 0.5) < 1e-9  # low freq  -> upward FCR -> droop > 0
    assert abs(d.iloc[3] + 0.5) < 1e-9  # high freq -> downward FCR -> droop < 0
    assert abs(d.iloc[4] - 0.1) < 1e-9  # past deadband -> kept (0.02 / 0.2)


def test_droop_clipped_to_unit_interval():
    freq = pd.Series([49.0, 51.0])  # delta -1.0 / +1.0, far past full activation
    d = droop_signal(freq, **_KW)
    assert d.iloc[0] == 1.0
    assert d.iloc[1] == -1.0


def test_gate_closure_d_minus_1_default():
    slot = pd.Timestamp("2025-10-02 00:00", tz="Europe/Berlin")
    gc = fcr_gate_closure_timestamp(slot)  # default 08:00, closes previous day
    assert gc == pd.Timestamp("2025-10-01 08:00", tz="Europe/Berlin")


def test_gate_closure_same_day_and_custom_hour():
    slot = pd.Timestamp("2025-10-02 00:00", tz="Europe/Berlin")
    gc = fcr_gate_closure_timestamp(slot, hour="06:30", closes_previous_day=False)
    assert gc == pd.Timestamp("2025-10-02 06:30", tz="Europe/Berlin")


if __name__ == "__main__":
    test_droop_sign_and_deadband()
    test_droop_clipped_to_unit_interval()
    test_gate_closure_d_minus_1_default()
    test_gate_closure_same_day_and_custom_hour()
    print("ok")
