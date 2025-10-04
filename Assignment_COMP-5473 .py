#!/usr/bin/env python
# coding: utf-8

# In[11]:


# DES Tkinter GUI (no CSV export)
import tkinter as tk
from tkinter import ttk, messagebox
from tabulate import tabulate

# --- Utility functions ---
def hex_to_bin(hex_str):
    """Convert hex string to binary string with proper padding."""
    return bin(int(hex_str, 16))[2:].zfill(len(hex_str) * 4)

def bin_to_hex(bin_str):
    """Convert binary string to hex string with proper padding."""
    return hex(int(bin_str, 2))[2:].zfill(len(bin_str) // 4)

def permute(block, table):
    """Permute bits of input block according to permutation table (1-based indices)."""
    return ''.join(block[i - 1] for i in table)

def left_shift(bits, n):
    """Circular left shift on bit string."""
    return bits[n:] + bits[:n]

def xor(bits1, bits2):
    """Bitwise XOR of two equal-length binary strings."""
    return ''.join('0' if b1 == b2 else '1' for b1, b2 in zip(bits1, bits2))

# --- DES standard tables ---
IP = [58,50,42,34,26,18,10,2,
      60,52,44,36,28,20,12,4,
      62,54,46,38,30,22,14,6,
      64,56,48,40,32,24,16,8,
      57,49,41,33,25,17,9,1,
      59,51,43,35,27,19,11,3,
      61,53,45,37,29,21,13,5,
      63,55,47,39,31,23,15,7]

IP_INV = [40,8,48,16,56,24,64,32,
          39,7,47,15,55,23,63,31,
          38,6,46,14,54,22,62,30,
          37,5,45,13,53,21,61,29,
          36,4,44,12,52,20,60,28,
          35,3,43,11,51,19,59,27,
          34,2,42,10,50,18,58,26,
          33,1,41,9,49,17,57,25]

E = [32,1,2,3,4,5,4,5,6,7,8,9,
     8,9,10,11,12,13,12,13,14,15,16,17,
     16,17,18,19,20,21,20,21,22,23,24,25,
     24,25,26,27,28,29,28,29,30,31,32,1]

P = [16,7,20,21,29,12,28,17,
     1,15,23,26,5,18,31,10,
     2,8,24,14,32,27,3,9,
     19,13,30,6,22,11,4,25]

PC1 = [57,49,41,33,25,17,9,
       1,58,50,42,34,26,18,
       10,2,59,51,43,35,27,
       19,11,3,60,52,44,36,
       63,55,47,39,31,23,15,
       7,62,54,46,38,30,22,
       14,6,61,53,45,37,29,
       21,13,5,28,20,12,4]

PC2 = [14,17,11,24,1,5,
       3,28,15,6,21,10,
       23,19,12,4,26,8,
       16,7,27,20,13,2,
       41,52,31,37,47,55,
       30,40,51,45,33,48,
       44,49,39,56,34,53,
       46,42,50,36,29,32]

ROTATIONS = [1,1,2,2,2,2,2,2,1,2,2,2,2,2,2,1]

SBOXES = [
  [[14,4,13,1,2,15,11,8,3,10,6,12,5,9,0,7],
   [0,15,7,4,14,2,13,1,10,6,12,11,9,5,3,8],
   [4,1,14,8,13,6,2,11,15,12,9,7,3,10,5,0],
   [15,12,8,2,4,9,1,7,5,11,3,14,10,0,6,13]],

  [[15,1,8,14,6,11,3,4,9,7,2,13,12,0,5,10],
   [3,13,4,7,15,2,8,14,12,0,1,10,6,9,11,5],
   [0,14,7,11,10,4,13,1,5,8,12,6,9,3,2,15],
   [13,8,10,1,3,15,4,2,11,6,7,12,0,5,14,9]],

  [[10,0,9,14,6,3,15,5,1,13,12,7,11,4,2,8],
   [13,7,0,9,3,4,6,10,2,8,5,14,12,11,15,1],
   [13,6,4,9,8,15,3,0,11,1,2,12,5,10,14,7],
   [1,10,13,0,6,9,8,7,4,15,14,3,11,5,2,12]],

  [[7,13,14,3,0,6,9,10,1,2,8,5,11,12,4,15],
   [13,8,11,5,6,15,0,3,4,7,2,12,1,10,14,9],
   [10,6,9,0,12,11,7,13,15,1,3,14,5,2,8,4],
   [3,15,0,6,10,1,13,8,9,4,5,11,12,7,2,14]],

  [[2,12,4,1,7,10,11,6,8,5,3,15,13,0,14,9],
   [14,11,2,12,4,7,13,1,5,0,15,10,3,9,8,6],
   [4,2,1,11,10,13,7,8,15,9,12,5,6,3,0,14],
   [11,8,12,7,1,14,2,13,6,15,0,9,10,4,5,3]],

  [[12,1,10,15,9,2,6,8,0,13,3,4,14,7,5,11],
   [10,15,4,2,7,12,9,5,6,1,13,14,0,11,3,8],
   [9,14,15,5,2,8,12,3,7,0,4,10,1,13,11,6],
   [4,3,2,12,9,5,15,10,11,14,1,7,6,0,8,13]],

  [[4,11,2,14,15,0,8,13,3,12,9,7,5,10,6,1],
   [13,0,11,7,4,9,1,10,14,3,5,12,2,15,8,6],
   [1,4,11,13,12,3,7,14,10,15,6,8,0,5,9,2],
   [6,11,13,8,1,4,10,7,9,5,0,15,14,2,3,12]],

  [[13,2,8,4,6,15,11,1,10,9,3,14,5,0,12,7],
   [1,15,13,8,10,3,7,4,12,5,6,11,0,14,9,2],
   [7,11,4,1,9,12,14,2,0,6,10,13,15,3,5,8],
   [2,1,14,7,4,10,8,13,15,12,9,0,3,5,6,11]]
]

# --- DES core helper functions ---
def generate_subkeys(key_64bit):
    """Generate 16 round subkeys (48-bit each) from 64-bit key."""
    key_56bit = permute(key_64bit, PC1)
    C = key_56bit[:28]
    D = key_56bit[28:]
    subkeys = []
    for rot in ROTATIONS:
        C = left_shift(C, rot)
        D = left_shift(D, rot)
        subkey = permute(C + D, PC2)
        subkeys.append(subkey)
    return subkeys

def format_ki_6bit_chunks(subkey_bin):
    """Return a readable hex representation of 48-bit subkey split into 8 chunks."""
    chunks = [subkey_bin[i*6:(i+1)*6] for i in range(8)]
    return ''.join("{:02x}".format(int(chunk, 2)) for chunk in chunks)

def sbox_substitution(bits48):
    """Apply all 8 S-boxes to 48-bit input and return 32-bit output."""
    out = ''
    for i in range(8):
        block = bits48[i*6:(i+1)*6]
        row = int(block[0] + block[5], 2)
        col = int(block[1:5], 2)
        val = SBOXES[i][row][col]
        out += bin(val)[2:].zfill(4)
    return out

def feistel(right32, subkey48):
    """DES Feistel f-function: expand, xor, S-boxes, then P-permute."""
    expanded = permute(right32, E)
    xored = xor(expanded, subkey48)
    sboxed = sbox_substitution(xored)
    return permute(sboxed, P)

# --- DES encryption (round-by-round) ---
def des_encrypt(plaintext_hex, key_hex):
    """Perform DES encryption, returning rows for display and the ciphertext hex."""
    plaintext_bin = hex_to_bin(plaintext_hex)
    key_bin = hex_to_bin(key_hex)
    permuted = permute(plaintext_bin, IP)
    L, R = permuted[:32], permuted[32:]
    subkeys = generate_subkeys(key_bin)

    rows = []
    # initial state
    rows.append(["IP", "", bin_to_hex(L), bin_to_hex(R)])

    for i in range(16):
        f = feistel(R, subkeys[i])
        new_L = R
        new_R = xor(L, f)
        ki_formatted = format_ki_6bit_chunks(subkeys[i])
        rows.append([str(i+1), ki_formatted, bin_to_hex(new_L), bin_to_hex(new_R)])
        L, R = new_L, new_R

    preoutput = R + L  # swap halves
    cipher_bin = permute(preoutput, IP_INV)
    cipher_hex = bin_to_hex(cipher_bin)

    # show final permutation result row (as IP⁻¹)
    rows.append(["IP\u207B\u00B9", "", cipher_hex[:8], cipher_hex[8:]])
    return rows, cipher_hex

def des_rounds_only(plaintext_bin, subkeys):
    """Return list of 64-bit intermediate states after each round (no final IP)."""
    permuted = permute(plaintext_bin, IP)
    L, R = permuted[:32], permuted[32:]
    states = []
    for i in range(16):
        f = feistel(R, subkeys[i])
        new_L = R
        new_R = xor(L, f)
        states.append(new_L + new_R)
        L, R = new_L, new_R
    return states

def diff_bits(bin1, bin2):
    """Count differing bits between two equal-length binaries."""
    return sum(b1 != b2 for b1, b2 in zip(bin1, bin2))

def avalanche_effect(p1_hex, p2_hex, key_hex):
    """Compute avalanche differences for rounds 0..8 between two plaintexts."""
    p1_bin = hex_to_bin(p1_hex).zfill(64)
    p2_bin = hex_to_bin(p2_hex).zfill(64)
    key_bin = hex_to_bin(key_hex).zfill(64)
    subkeys = generate_subkeys(key_bin)

    rounds_p1 = des_rounds_only(p1_bin, subkeys)
    rounds_p2 = des_rounds_only(p2_bin, subkeys)

    rows = []
    # initial (round 0) compare originals
    rows.append(["", f"{p1_hex}\n{p2_hex}", diff_bits(p1_bin, p2_bin)])

    # show rounds 1..8
    for i in range(8):
        hex1 = bin_to_hex(rounds_p1[i]).zfill(16)
        hex2 = bin_to_hex(rounds_p2[i]).zfill(16)
        combined = f"{hex1}\n{hex2}"
        d = diff_bits(rounds_p1[i], rounds_p2[i])
        rows.append([str(i+1), combined, d])

    headers = ["Round", "", "δ"]
    table_str = tabulate(rows, headers=headers, tablefmt="fancy_grid", stralign="center", numalign="center")
    return table_str, headers, rows

# --- GUI Setup ---
root = tk.Tk()
root.title("Assignment - DES Encryption and Avalanche Effect")
root.state('zoomed')

notebook = ttk.Notebook(root)
notebook.pack(fill='both', expand=True)

def create_input_frame(parent, labels_and_defaults):
    frame = tk.Frame(parent)
    entries = []
    for i, (label_text, default_val) in enumerate(labels_and_defaults):
        tk.Label(frame, text=label_text).grid(row=i, column=0, sticky='e', padx=5, pady=5)
        entry = tk.Entry(frame, width=40)
        entry.grid(row=i, column=1, padx=5, pady=5)
        entry.insert(0, default_val)
        entries.append(entry)
    return frame, entries

def create_output_text(parent):
    frame = tk.Frame(parent)
    frame.pack(pady=10, fill='both', expand=True)
    text = tk.Text(frame, wrap='none')
    text.pack(fill='both', expand=True)
    return text

# Tab (a) DES
tab_a = ttk.Frame(notebook)
notebook.add(tab_a, text="(a) DES Encryption")
input_labels_defaults_a = [
    ("Plaintext (16 hex chars):", "02468aceeca86420"),
    ("Key (16 hex chars):", "0f1571c947d9e859")
]
input_frame_a, (plaintext_entry_a, key_entry_a) = create_input_frame(tab_a, input_labels_defaults_a)
input_frame_a.pack(pady=10)
output_text_a = create_output_text(tab_a)
result_label_a = tk.Label(tab_a, text="Ciphertext: ")
result_label_a.pack()

def run_des():
    plaintext = plaintext_entry_a.get().strip()
    key = key_entry_a.get().strip()
    if len(plaintext) != 16 or any(c not in '0123456789abcdefABCDEF' for c in plaintext):
        messagebox.showerror("Error", "Plaintext must be 16 hex characters (0-9, a-f).")
        return
    if len(key) != 16 or any(c not in '0123456789abcdefABCDEF' for c in key):
        messagebox.showerror("Error", "Key must be 16 hex characters (0-9, a-f).")
        return
    rows, ciphertext = des_encrypt(plaintext, key)
    headers = ["Round", "Ki", "Li", "Ri"]
    table_str = tabulate(rows, headers=headers, tablefmt="fancy_grid")
    output_text_a.delete(1.0, tk.END)
    output_text_a.insert(tk.END, table_str)
    result_label_a.config(text=f"Ciphertext: {ciphertext}")

button_frame_a = tk.Frame(tab_a)
button_frame_a.pack(pady=5)
run_button_a = tk.Button(button_frame_a, text="Encrypt and Show Rounds", command=run_des)
run_button_a.grid(row=0, column=0, padx=10)

# Tab (b) Avalanche
tab_b = ttk.Frame(notebook)
notebook.add(tab_b, text="(b) Avalanche Effect")
input_labels_defaults_b = [
    ("Plaintext 1 (16 hex chars):", "02468aceeca86420"),
    ("Plaintext 2 (16 hex chars):", "12468aceeca86420"),
    ("Key (16 hex chars):", "0f1571c947d9e859")
]
input_frame_b, (plaintext1_entry_b, plaintext2_entry_b, key_entry_b) = create_input_frame(tab_b, input_labels_defaults_b)
input_frame_b.pack(pady=10)
output_text_b = create_output_text(tab_b)

def run_avalanche():
    p1 = plaintext1_entry_b.get().strip()
    p2 = plaintext2_entry_b.get().strip()
    key = key_entry_b.get().strip()
    for label, val in [("Plaintext 1", p1), ("Plaintext 2", p2), ("Key", key)]:
        if len(val) != 16 or any(c not in '0123456789abcdefABCDEF' for c in val):
            messagebox.showerror("Error", f"{label} must be 16 hex characters (0-9, a-f).")
            return
    table_str, headers, rows = avalanche_effect(p1, p2, key)
    output_text_b.delete(1.0, tk.END)
    output_text_b.insert(tk.END, table_str)

button_frame_b = tk.Frame(tab_b)
button_frame_b.pack(pady=5)
run_button_b = tk.Button(button_frame_b, text="Show Avalanche Effect", command=run_avalanche)
run_button_b.grid(row=0, column=0, padx=10)

# Footer
footer_label = tk.Label(root, text="Created by [StdID-1276026, Name: Aye Kyi Kyi Cho]", font=("Arial", 10), pady=5)
footer_label.pack(side='bottom')

root.mainloop()


# In[ ]:




