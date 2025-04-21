import streamlit as st
import random
import numpy as np
import copy
from collections import Counter
import pandas as pd

# Page configuration
st.set_page_config(page_title="Camel Race Simulator", layout="wide")
st.title("🎲 Camel Race Simulator")

# Camel colors and emoji
camel_colors = ["Purple", "Blue", "Red", "Green", "Yellow", "Black", "White"]
backward_camels = {"Black", "White"}
camel_emojis = {
    "Purple": "🟣", "Blue": "🔵", "Red": "🔴",
    "Green": "🟢", "Yellow": "🟡", "Black": "⚫", "White": "⚪"
}
track_len = 16

# Load precomputed tensor for prediction (cached)
@st.cache_data
def load_tensor(path="permutation_face_tensor.npz"):
    data = np.load(path)
    return data["tensor"]
tensor = load_tensor()

# Predict round-end winning probabilities
def predict_probs():
    ss = st.session_state
    rolled = ss.roll_count
    total_rolls = 7 if ss.current_round == 1 else 5
    # uniform if no rolls yet
    if rolled == 0:
        return {c: 1/len(camel_colors) for c in camel_colors}
    mask = np.ones(tensor.shape[:2], dtype=bool)
    # filter scenarios by history
    for k in range(rolled):
        _,_,col,steps = ss.dice_history[k]
        die_id = camel_colors.index(col) + 1
        mask &= (tensor[:,:,k,0] == die_id) & (tensor[:,:,k,1] == steps)
    winners = []
    for i,j in np.argwhere(mask):
        # simulate scenario (i,j)
        sim_pos = {c: ss.positions.get(c, 1) for c in camel_colors}
        sim_stack = ss.camel_stack.copy()
        for k in range(rolled, total_rolls):
            die_id = int(tensor[i,j,k,0])
            face_val = int(tensor[i,j,k,1])
            color = camel_colors[die_id-1]
            mv = face_val if face_val <= 3 else face_val - 3
            step = -mv if color in backward_camels else mv
            cur = sim_pos[color]
            tgt = cur + step
            if color in backward_camels and tgt < 1:
                tgt += track_len
            if color not in backward_camels and tgt > track_len:
                tgt = track_len
            # move stack
            same = [c for c in sim_stack if sim_pos[c] == cur]
            if color not in same:
                continue
            idx0 = same.index(color)
            moving = same[idx0:]
            staying = [c for c in sim_stack if c not in moving]
            for c in moving:
                sim_pos[c] = tgt
            below = [c for c in staying if sim_pos[c] != tgt]
            above = [c for c in staying if sim_pos[c] == tgt]
            sim_stack = below + above + moving
        # determine winner
        maxp = max(sim_pos[c] for c in sim_stack if c not in backward_camels)
        cands = [c for c in sim_stack if sim_pos[c] == maxp and c not in backward_camels]
        for c in reversed(sim_stack):
            if c in cands:
                winners.append(c)
                break
    cnt = Counter(winners)
    total = len(winners)
    return {c: (cnt[c]/total if total>0 else 0.0) for c in camel_colors}

# Callbacks for buttons
def reset_game():
    for key in list(st.session_state.keys()):
        del st.session_state[key]

def add_trap():
    ss = st.session_state
    ss.traps[ss.trap_pos] = 1 if ss.trap_type == "+1" else -1

def confirm_roll():
    ss = st.session_state
    color = ss.selected_color
    steps = ss.selected_steps
    # handle roll
    if ss.current_round == 1:
        pos = 17 - steps if color in backward_camels else steps
        ss.positions[color] = pos
        ss.camel_stack.append(color)
    else:
        cur = ss.positions[color]
        if color in backward_camels:
            tmp = cur - steps
            pos = track_len + tmp if tmp < 1 else tmp
        else:
            pos = cur + steps
        # trap adjust
        trap = ss.traps.get(pos, 0)
        if color in backward_camels:
            pos = pos - trap
            if pos < 1: pos += track_len
        else:
            pos = pos + trap
            if pos > track_len: pos = track_len
        same = [c for c in ss.camel_stack if ss.positions[c] == cur]
        idx0 = same.index(color)
        moving = same[idx0:]
        staying = [c for c in ss.camel_stack if c not in moving]
        for c in moving:
            ss.positions[c] = pos
        below = [c for c in staying if ss.positions[c] != pos]
        above = [c for c in staying if ss.positions[c] == pos]
        ss.camel_stack = below + above + moving
    ss.dice_history.append((ss.current_round, ss.roll_count+1, color, steps))
    ss.used_this_round.append(color)
    ss.roll_count += 1
    finishers = [c for c in ss.camel_stack[::-1] if ss.positions[c] > track_len and c not in backward_camels]
    if finishers and not ss.final_winner:
        ss.final_winner = finishers[0]

def end_round():
    ss = st.session_state
    pm = ss.positions
    valids = [c for c in camel_colors if c not in backward_camels and c in pm]
    if valids:
        mx = max(pm[c] for c in valids)
        cands = [c for c in valids if pm[c] == mx]
        for c in ss.camel_stack[::-1]:
            if c in cands:
                ss.round_winners.append((ss.current_round, c))
                break
    ss.current_round += 1
    ss.roll_count = 0
    ss.remaining_dice = []

# UI: Reset game
st.button("🔄 Reset Game", on_click=reset_game)

# Initialize state
if "positions" not in st.session_state:
    st.session_state.update({
        "positions": {},
        "camel_stack": [],
        "dice_history": [],
        "round_winners": [],
        "current_round": 1,
        "roll_count": 0,
        "remaining_dice": [],
        "used_this_round": [],
        "traps": {},
        "final_winner": None
    })

# Start of round setup
ss = st.session_state
if not ss.remaining_dice:
    ss.remaining_dice = camel_colors.copy() if ss.current_round == 1 else random.sample(camel_colors, 5)
    ss.used_this_round = []
    if ss.current_round > 1:
        ss.traps.clear()

# UI: Roll Input
st.subheader("🌀 Roll Input")
if ss.final_winner:
    st.success(f"🏁 Game Over! Winner: {ss.final_winner}")
elif ss.current_round > 10:
    st.warning("10 rounds completed without finish.")
else:
    total = 7 if ss.current_round == 1 else 5
    st.write(f"Round {ss.current_round}, Roll {ss.roll_count+1} of {total}")
    # Trap UI
    if ss.current_round > 1:
        st.subheader("🪤 Set Trap")
        st.number_input("Trap position (1-16)", 1, track_len, key="trap_pos")
        st.selectbox("Trap type", ["+1", "-1"], key="trap_type")
        st.button("Add Trap", on_click=add_trap)
    # Roll controls
    st.selectbox("Select camel color", [c for c in camel_colors if c not in ss.used_this_round], key="selected_color")
    st.number_input("Steps (1-3)", 1, 3, key="selected_steps")
    st.button("Confirm Roll", on_click=confirm_roll)
    # End round
    if ss.roll_count >= total:
        st.subheader("🏁 End Round")
        st.button("Record Winner & Next Round", on_click=end_round)

# UI: Prediction
st.subheader("🔮 Round-end Winning Probability")
# Only show predict button from round 2 onward
if st.session_state.current_round > 1:
    if st.button("Predict Probability"):
        probs = predict_probs()
        for c in camel_colors:
            st.write(f"{c}: {probs[c]*100:.1f}%")

# UI: Track display
st.subheader("🐪 Current Camel Positions & Traps")
# Support Chinese-to-English translation for any leftover Chinese color keys
translation_map = {"紫":"Purple","蓝":"Blue","红":"Red","绿":"Green","黄":"Yellow","黑":"Black","白":"White"}
# Build positions grid
grid = {i: [] for i in range(1, track_len+1)}
for orig_color in st.session_state.camel_stack:
    color = translation_map.get(orig_color, orig_color)
    pos = st.session_state.positions.get(orig_color, 0)
    if 1 <= pos <= track_len:
        grid[pos].append(camel_emojis.get(color, ""))
# Render grid rows including traps
rows = []
height = max((len(v) for v in grid.values()), default=1)
for lvl in range(height-1, -1, -1):
    row = []
    for i in range(1, track_len+1):
        if lvl < len(grid[i]):
            cell = grid[i][lvl]
        elif lvl == 0 and i in st.session_state.traps:
            cell = "+1" if st.session_state.traps[i] == 1 else "-1"
        else:
            cell = ""
        row.append(cell)
    rows.append(row)
st.table(pd.DataFrame(rows, columns=[str(i) for i in range(1, track_len+1)]))

# UI: Roll History
st.subheader("🎯 Roll History")
hist = [["" for _ in range(10)] for _ in range(7)]
for rnd, roll, col, stp in ss.dice_history:
    if 1 <= rnd <= 10 and 1 <= roll <= 7:
        hist[roll-1][rnd-1] = f"{col}({stp})"
st.dataframe(pd.DataFrame(hist, index=[f"Roll{r}" for r in range(1,8)], columns=[f"Round{c}" for c in range(1,11)]), use_container_width=True)

# UI: Round Winners
if ss.round_winners:
    st.subheader("🏅 Round Winners")
    for r, w in ss.round_winners:
        st.write(f"Round {r}: {w}")
