from src.env.powertrain import Battery, _Q_BT_IC

b = Battery()
b.reset()
print(f"q_bt before: {b.q_bt}")

out = b.step(p_bt=5000.0, x_tot=10.0)  # discharge: p_bt > 0
print(f"i_bt = {out['i_bt']:.4f}")
print(f"q_bt after (discharge, p_bt=+5000): {out['q_bt']:.4f}")
print(f"delta = {out['q_bt'] - _Q_BT_IC:+.4f}  (expected: NEGATIVE)")