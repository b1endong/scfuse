VIEWS = ("S", "G", "D", "A")
SOURCES = VIEWS + ("J",)
VI = {v: i for i, v in enumerate(VIEWS)}
TEXT_VIEWS = ("S", "D")

MAX_UNITS, CHUNK, MAX_CHUNKS = 64, 508, 4
MAX_BLOCKS, BLOCK_OPS = 512, 32
MAX_AST_NODES = 256
UNITS_SCALE = {"S": 6.0, "D": 5.0, "A": 6.0}
CHARS_SCALE = 12.0

ARRAYS = {
    "S_tok": ("uint16", 1), "S_chunk": ("int64", 2), "S_unit": ("int64", 3),
    "D_tok": ("uint16", 1), "D_chunk": ("int64", 2), "D_unit": ("int64", 3),
    "G_x": ("uint8", BLOCK_OPS), "G_edge": ("int16", 2), "G_etype": ("uint8", 1), "G_fnsel": ("int64", 1),
    "G_mem": ("int32", 2), "G_pc": ("int32", 1),
    "A_node": ("int16", 3), "A_unit": ("int64", 3),
    "SP_pair": ("int32", 2), "SP_span": ("int32", 3),
}

Q_LAYOUT = {
    "S": ("solc_minor", "solc_patch", "optimizer", "matched", "log_units", "log_chars"),
    "G": ("unresolved", "log_blocks", "truncated", "entries_beyond_cap"),
    "D": ("compiles", "fidelity", "sel_agreement", "log_units", "log_chars"),
    "A": ("log_nodes", "log_units", "truncated", "matched"),
}
Q_LAYOUT_V25 = {
    "S": ("solc_minor", "log_units", "sel_share", "log_chars"),
    "G": ("unresolved", "log_blocks", "truncated", "entries_beyond_cap"),
    "D": ("heuristic", "log_units", "sel_recovery", "log_chars"),
    "A": ("log_nodes", "log_units", "truncated", "matched"),
}

LABEL_FIXES = {"solidifi_unchecked_send_acc": ("solidifi_Unchecked-Send_", "ULC", "ACC")}
