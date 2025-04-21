import streamlit as st
import random
import numpy as np
from collections import Counter
import pandas as pd

# Page configuration
st.set_page_config(page_title="Camel Up Predictor", layout="wide")
st.title("Camel Up Predictor")

# Camel setup
dice_colors = ["Purple", "Blue", "Red", "Green", "Yellow", "Black", "White"]
backward_camels = {"Black", "White"}
camel_emojis = {
    "Purple": "🟣", "Blue": "🔵", "Red": "🔴",
    "Green": "🟢", "Yellow": "🟡", "Black": "⚫", "White": "⚪"
}
track_len = 16

# Load prediction tensor
@st.cache_data
def load_tensor(path="permutation_face_tensor.npz"):
    data = np.load(path)
    return data["tensor"]
tensor = load_tensor()

# Predict leg-end winning odds
def predict_probs():
    ss = st.session_state
    rolled = ss.roll_count
    total = 7 if ss.current_round == 1 else 5
    # uniform if no rolls yet
    if rolled == 0:
        return {c: 1/len(dice_colors) for c in dice_colors}
    mask = np.ones(tensor.shape[:2], dtype=bool)
    # filter by history
    for k in range(rolled):
        _, _, col, steps = ss.dice_history[k]
        die_id = dice_colors.index(col) + 1
        mask &= (tensor[:, :, k, 0] == die_id) & (tensor[:, :, k, 1] == steps)
    winners = []
    for i, j in np.argwhere(mask):
        # simulate scenario
        sim_pos = {c: ss.positions.get(c, 1) for c in dice_colors}
        sim_stack = ss.camel_stack.copy()
        for k in range(rolled, total):
            did = int(tensor[i, j, k, 0]); fval = int(tensor[i, j, k, 1])
            color = dice_colors[did - 1]
            mv = fval if fval <= 3 else fval - 3
            step = -mv if color in backward_camels else mv
            cur = sim_pos[color]; tgt = cur + step
            if color in backward_camels and tgt < 1:
                tgt += track_len
            if color not in backward_camels and tgt > track_len:
                tgt = track_len
            same = [c for c in sim_stack if sim_pos[c] == cur]
            if color not in same:
                continue
            idx0 = same.index(color)
            moving = same[idx0:]; staying = [c for c in sim_stack if c not in moving]
            for c in moving:
                sim_pos[c] = tgt
            sim_stack = [c for c in staying if sim_pos[c] != tgt] + [c for c in staying if sim_pos[c] == tgt] + moving
        # determine winner
        maxp = max(sim_pos[c] for c in sim_stack if c not in backward_camels)
        cands = [c for c in sim_stack if sim_pos[c] == maxp and c not in backward_camels]
        for c in reversed(sim_stack):
            if c in cands:
                winners.append(c)
                break
    cnt = Counter(winners); N = len(winners)
    return {c: (cnt[c] / N if N > 0 else 0.0) for c in dice_colors}

# Callbacks
def reset_game():
    for k in list(st.session_state.keys()):
        del st.session_state[k]

def add_trap():
    ss = st.session_state
    ss.traps[ss.trap_pos] = 1 if ss.trap_type == "+1" else -1

def confirm_roll():
    ss = st.session_state
    color = ss.selected_color; steps = ss.selected_steps
    if ss.current_round == 1:
        pos = 17 - steps if color in backward_camels else steps
        ss.positions[color] = pos; ss.camel_stack.append(color)
    else:
        cur = ss.positions[color]
        if color in backward_camels:
            tmp = cur - steps; pos = track_len + tmp if tmp < 1 else tmp
        else:
            pos = cur + steps
        trap = ss.traps.get(pos, 0)
        if color in backward_camels:
            pos = pos - trap
            if pos < 1: pos += track_len
        else:
            pos = pos + trap
            if pos > track_len: pos = track_len
        same = [c for c in ss.camel_stack if ss.positions[c] == cur]
        idx0 = same.index(color); moving = same[idx0:]; staying = [c for c in ss.camel_stack if c not in moving]
        for c in moving: ss.positions[c] = pos
        ss.camel_stack = [c for c in staying if ss.positions[c] != pos] + [c for c in staying if ss.positions[c] == pos] + moving
    ss.dice_history.append((ss.current_round, ss.roll_count + 1, color, steps))
    ss.used_this_round.append(color); ss.roll_count += 1
    finishers = [c for c in ss.camel_stack[::-1] if ss.positions[c] > track_len and c not in backward_camels]
    if finishers and not ss.final_winner: ss.final_winner = finishers[0]

def end_leg():
    ss = st.session_state; pm = ss.positions
    candidates = [c for c in dice_colors if c not in backward_camels and c in pm]
    if candidates:
        maxp = max(pm[c] for c in candidates)
        winners = [c for c in candidates if pm[c] == maxp]
        for c in ss.camel_stack[::-1]:
            if c in winners: ss.round_winners.append((ss.current_round, c)); break
    ss.current_round += 1; ss.roll_count = 0; ss.remaining_dice = []

# UI: Start New Race
st.button("Start New Race", on_click=reset_game)

# Initialize state
if "positions" not in st.session_state:
    st.session_state.update({
        "positions": {}, "camel_stack": [], "dice_history": [],
        "round_winners": [], "current_round": 1, "roll_count": 0,
        "remaining_dice": [], "used_this_round": [], "traps": {}, "final_winner": None
    })

ss = st.session_state
# Setup leg start
total_rolls = 7 if ss.current_round == 1 else 5
if not ss.remaining_dice:
    ss.remaining_dice = dice_colors.copy() if ss.current_round == 1 else random.sample(dice_colors, 5)
    ss.used_this_round = []
    if ss.current_round > 1: ss.traps.clear()

# UI: Dice Roll
st.subheader("Dice Roll")
if ss.final_winner:
    st.success(f"Race Completed! Winner: {ss.final_winner}")
elif ss.current_round > 10:
    st.warning("10 legs completed without finish line crossed.")
else:
    st.write(f"Leg {ss.current_round}: Roll {ss.roll_count+1} of {total_rolls}")
    if ss.current_round > 1:
        st.subheader("Place Trap")
        st.number_input("Trap position (1-16)", 1, track_len, key="trap_pos")
        st.selectbox("Trap type", ["+1", "-1"], key="trap_type")
        st.button("Place Trap", on_click=add_trap)
    valid = [c for c in dice_colors if c not in ss.used_this_round]
    options = [""] + valid
    st.selectbox("Select camel", options, key="selected_color")
    st.number_input("Steps (1-3)", 1, 3, key="selected_steps")
    disable_roll = (ss.selected_color == "") or (ss.roll_count >= total_rolls)
    st.button("Submit Roll", on_click=confirm_roll, disabled=disable_roll)
    if ss.roll_count >= total_rolls:
        st.subheader("Conclude Leg")
        st.button("Record Leg Winner & Start Next Leg", on_click=end_leg)

# UI: Estimate Leg Winner Odds
st.subheader("Estimate Leg Winner Odds")
disable_predict = ss.current_round == 1
if st.button("Estimate Odds", disabled=disable_predict):
    probs = predict_probs()
    for c in dice_colors:
        st.write(f"{c}: {probs[c]*100:.1f}%")

# UI: Camel Positions & Traps on Track
st.subheader("Camel Positions & Traps on Track")
grid = {i: [] for i in range(1, track_len+1)}
for c in ss.camel_stack:
    p = ss.positions.get(c, 0)
    if 1 <= p <= track_len:
        grid[p].append(camel_emojis[c])
rows = []
height = max((len(v) for v in grid.values()), default=1)
for lvl in range(height-1, -1, -1):
    row = []
    for i in range(1, track_len+1):
        if lvl < len(grid[i]):
            cell = grid[i][lvl]
        elif lvl == 0 and i in ss.traps:
            cell = "+1" if ss.traps[i] == 1 else "-1"
        else:
            cell = ""
        row.append(cell)
    rows.append(row)
st.table(pd.DataFrame(rows, columns=[str(i) for i in range(1, track_len+1)]))

# UI: Dice Roll History
st.subheader("Dice Roll History")
hist = [["" for _ in range(10)] for _ in range(7)]
for rnd, roll, col, stp in ss.dice_history:
    if 1 <= rnd <= 10 and 1 <= roll <= 7:
        hist[roll-1][rnd-1] = f"{col}({stp})"
st.dataframe(pd.DataFrame(hist, index=[f"Roll{r}" for r in range(1,8)], columns=[f"Leg{c}" for c in range(1,11)]), use_container_width=True)

# UI: Leg Winners
if ss.round_winners:
    st.subheader("Leg Winners")
    for r, w in ss.round_winners:
        st.write(f"Leg {r}: {w}")
