import streamlit as st
import random
import numpy as np
import copy
from collections import defaultdict, Counter
import pandas as pd

# Page configuration
st.set_page_config(page_title="Camel Race Simulator", layout="wide")
st.title("🎲 Camel Race Simulator")

# Camel colors and directions
camel_colors = ["Purple", "Blue", "Red", "Green", "Yellow", "Black", "White"]
backward_camels = ["Black", "White"]
camel_emojis = {"Purple": "🟣", "Blue": "🔵", "Red": "🔴", "Green": "🟢", "Yellow": "🟡", "Black": "⚫", "White": "⚪"}
track_length = 16

# Load 4D tensor for prediction (cached)
@st.cache_data
def load_tensor(path="permutation_face_tensor.npz"):
    return np.load(path)["tensor"]
tensor = load_tensor()

# Function to predict round-end probabilities
def predict_round_winner():
    ss = st.session_state
    rolled = ss.roll_count
    total = 7 if ss.current_round == 1 else 5
    # If no rolls yet, return uniform probabilities
    if rolled == 0:
        return {c: 1/len(camel_colors) for c in camel_colors}
    # Build mask of matching scenarios
    mask = np.ones(tensor.shape[:2], dtype=bool)
    for k in range(rolled):
        _, _, color, steps = ss.dice_history[k]
        die_id = camel_colors.index(color) + 1
        face_val = steps
        mask &= (tensor[:, :, k, 0] == die_id) & (tensor[:, :, k, 1] == face_val)
    winners = []
    indices = np.argwhere(mask)
    # Simulate each matching scenario
    for i, j in indices:
        # Initialize simulation state
        sim_pos = {c: ss.positions.get(c, 1) for c in camel_colors}
        # Place any unused camels at starting if needed\#
        sim_stack = [c for c in camel_colors if c not in ss.camel_stack] + ss.camel_stack.copy()
        # Simulate remaining rolls
        for k in range(rolled, total):
            die_id = int(tensor[i, j, k, 0])
            face_val = int(tensor[i, j, k, 1])
            color = camel_colors[die_id - 1]
            move_val = face_val if face_val <= 3 else face_val - 3
            direction = -1 if color in backward_camels else 1
            step = direction * move_val
            current_pos = sim_pos[color]
            target = current_pos + step
            if color in backward_camels:
                if target < 1:
                    target += track_length
            else:
                if target > track_length:
                    target = track_length
            # Stack movement
            same_stack = [c for c in sim_stack if sim_pos[c] == current_pos]
            if color not in same_stack:
                continue
            idx0 = same_stack.index(color)
            moving = same_stack[idx0:]
            staying = [c for c in sim_stack if c not in moving]
            for c in moving:
                sim_pos[c] = target
            below = [c for c in staying if sim_pos[c] != target]
            above = [c for c in staying if sim_pos[c] == target]
            sim_stack = below + above + moving
        # Determine winner: highest non-backward camel, top of stack if tied
        max_pos = max(sim_pos[c] for c in sim_stack if c not in backward_camels)
        candidates = [c for c in sim_stack if sim_pos[c] == max_pos and c not in backward_camels]
        for c in reversed(sim_stack):
            if c in candidates:
                winners.append(c)
                break
    count = Counter(winners)
    total_cases = len(winners)
    return {c: (count[c]/total_cases if total_cases>0 else 0.0) for c in camel_colors}

# Reset button
if st.button("🔄 Reset Game"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.experimental_rerun()

# Initialize game state
if "positions" not in st.session_state:
    st.session_state.positions = {}
    st.session_state.camel_stack = []
    st.session_state.dice_history = []
    st.session_state.round_winners = []
    st.session_state.current_round = 1
    st.session_state.roll_count = 0
    st.session_state.remaining_dice = []
    st.session_state.used_this_round = []
    st.session_state.traps = {}
    st.session_state.final_winner = None

# Start of round: setup dice and clear traps for round >1
if len(st.session_state.remaining_dice) == 0:
    ss = st.session_state
    ss.remaining_dice = camel_colors.copy() if ss.current_round == 1 else random.sample(camel_colors, 5)
    ss.used_this_round = []
    if ss.current_round > 1:
        ss.traps = {}

# Roll handler
def handle_roll(color, steps):
    ss = st.session_state
    if ss.current_round == 1:
        # Initial placement
        pos = 17 - steps if color in backward_camels else steps
        ss.positions[color] = pos
        ss.camel_stack.append(color)
    else:
        # Movement with stacking
        cur = ss.positions[color]
        if color in backward_camels:
            tmp = cur - steps
            pos = track_length + tmp if tmp < 1 else tmp
        else:
            pos = cur + steps
        # Trap adjustment: reverse for black/white
        trap_val = ss.traps.get(pos, 0)
        if color in backward_camels:
            pos = max(1, min(track_length, pos - trap_val))
        else:
            pos = max(1, min(track_length, pos + trap_val))
        full = ss.camel_stack
        same_stack = [c for c in full if ss.positions[c] == cur]
        idx0 = same_stack.index(color)
        moving = same_stack[idx0:]
        staying = [c for c in full if c not in moving]
        for c in moving:
            ss.positions[c] = pos
        below = [c for c in staying if ss.positions[c] != pos]
        above = [c for c in staying if ss.positions[c] == pos]
        ss.camel_stack = below + above + moving
    # Record roll
    ss.dice_history.append((ss.current_round, ss.roll_count+1, color, steps))
    ss.used_this_round.append(color)
    ss.roll_count += 1
    # Check final winner
    finishers = [c for c in ss.camel_stack[::-1] if ss.positions[c] > track_length and c not in backward_camels]
    if finishers and not ss.final_winner:
        ss.final_winner = finishers[0]

# Roll input UI
st.subheader("🌀 Roll Input")
if st.session_state.final_winner:
    st.success(f"🏁 Game Over! Final winner: {st.session_state.final_winner}")
elif st.session_state.current_round > 10:
    st.warning("10 rounds completed without reaching finish line.")
else:
    ss = st.session_state
    total = 7 if ss.current_round == 1 else 5
    st.write(f"Round {ss.current_round}: Roll {ss.roll_count+1} of {total}")
    # Trap setup
    if ss.current_round > 1:
        st.subheader("🪤 Set Trap")
        col1, col2 = st.columns(2)
        with col1:
            trap_pos = st.number_input("Trap position (1-16)", 1, track_length, 1)
        with col2:
            trap_type = st.selectbox("Trap type", ["+1", "-1"])
        if st.button("Add Trap"):
            ss.traps[trap_pos] = 1 if trap_type == "+1" else -1
    # Roll controls
    col1, col2 = st.columns(2)
    valid = [c for c in camel_colors if c not in ss.used_this_round]
    with col1:
        selected_color = st.selectbox("Select camel color", valid)
    with col2:
        selected_steps = st.number_input("Steps (1-3)", 1, 3, 1)
    if st.button("Confirm Roll"):
        if selected_color in valid:
            handle_roll(selected_color, selected_steps)
        st.experimental_rerun()
    # End-round button
    if ss.roll_count >= total:
        st.subheader("🏁 End Round")
        if st.button("Record winner and next round"):
            posmap = ss.positions
            valids = [c for c in camel_colors if c not in backward_camels and c in posmap]
            if valids:
                maxp = max(posmap[c] for c in valids)
                cands = [c for c in valids if posmap[c] == maxp]
                winner = None
                for c in ss.camel_stack[::-1]:
                    if c in cands:
                        winner = c
                        break
                if winner:
                    ss.round_winners.append((ss.current_round, winner))
            ss.current_round += 1
            ss.roll_count = 0
            ss.remaining_dice = []
            st.experimental_rerun()

# Prediction UI
st.subheader("🔮 Round-end Winning Probability")
if st.button("Predict Probability"):
    probs = predict_round_winner()
    for c in camel_colors:
        st.write(f"{c}: {probs.get(c,0)*100:.1f}%")

# Track display (inc. traps)
st.subheader("🐪 Current Camel Positions & Traps")
position_grid = {i: [] for i in range(1, track_length+1)}
for c in st.session_state.camel_stack:
    p = st.session_state.positions.get(c, 0)
    if 1 <= p <= track_length:
        position_grid[p].append(camel_emojis[c])
rows = []
height = max((len(v) for v in position_grid.values()), default=1)
for lvl in range(height-1, -1, -1):
    row = []
    for i in range(1, track_length+1):
        if lvl < len(position_grid[i]):
            cell = position_grid[i][lvl]
        elif lvl == 0 and i in st.session_state.traps:
            cell = "+1" if st.session_state.traps[i] == 1 else "-1"
        else:
            cell = ""
        row.append(cell)
    rows.append(row)
st.table(pd.DataFrame(rows, columns=[str(i) for i in range(1, track_length+1)]))

# Roll history UI
st.subheader("🎯 Roll History")
history_matrix = [["" for _ in range(10)] for _ in range(7)]
for rnd, roll, col, stp in st.session_state.dice_history:
    if 1 <= rnd <= 10 and 1 <= roll <= 7:
        history_matrix[roll-1][rnd-1] = f"{col}({stp})"
history_df = pd.DataFrame(history_matrix, index=[f"Roll{r}" for r in range(1,8)], columns=[f"Round{c}" for c in range(1,11)])
st.dataframe(history_df, use_container_width=True)

# Round winners UI
if st.session_state.round_winners:
    st.subheader("🏅 Round Winners")
    for r, w in st.session_state.round_winners:
        st.write(f"Round {r}: {w}")
