from scfuse.utils import ast as A
from scfuse.utils.evm import E_UNKNOWN, build_cfg, edit_distance, selector_agreement, split_metadata
from scfuse.utils.solc import parse_srcmap
from scfuse.utils.solidity import blank_comments, scan, scrub, selector, squash_map

DISPATCH = bytes.fromhex(
    "6000" "35" "60e0" "1c"
    "80" "63a9059cbb" "14" "6011" "57"
    "00"
    "5b" "6001" "6017" "56"
    "5b" "00")


def test_dispatcher_and_blocks():
    g = build_cfg(DISPATCH)
    assert g.sels == [0xA9059CBB]
    assert g.n_blocks == len(g.pcs) and g.x.shape == (len(g.pcs), 32)
    assert len(g.fn_blocks) == 1 and g.fn_blocks[0]
    assert all(0 <= u < len(g.pcs) and 0 <= v < len(g.pcs) for u, v in g.edges.tolist())


def test_unresolved_jump_marked():
    code = bytes.fromhex("3556" "5b00")
    g = build_cfg(code)
    assert E_UNKNOWN in g.etypes.tolist() and g.unresolved == 1.0


def test_metadata_split_and_edit_distance():
    body, meta = split_metadata(bytes.fromhex("6000" "a1" "65" + "00" * 5 + "0007"))
    assert body == bytes.fromhex("6000") and len(meta) == 7
    assert edit_distance([1, 2, 3], [1, 3]) == 1 and edit_distance([], [4, 5]) == 2


def test_selector_agreement():
    agree, prec, cov = selector_agreement([1, 2, None, -1], [2, 3])
    assert (agree, prec, cov) == (1 / 3, 0.5, 0.5)


def test_srcmap_inherits_fields():
    assert parse_srcmap("1:2:0;:3;4::1;;") == [(1, 2, 0), (1, 3, 0), (4, 3, 1), (4, 3, 1), (4, 3, 1)]


def test_comment_blanking_keeps_offsets():
    src = 'contract C { // x\n string s = "//no"; /* y */ function f() public {} }'
    blank = blank_comments(src)
    assert len(blank) == len(src) and "//no" in blank and "x" not in blank.split("\n")[0][12:]


def test_scanner_selector_and_scrub():
    rec = scan("contract T { function transfer(address to, uint v) public returns (bool) { return true; } }")
    assert rec["units"][0]["sel"] == 0xA9059CBB == selector("transfer", ["address", "uint256"])
    names = {}
    assert scrub("bug_tmstmp1 = reentrancyGuard + bug_tmstmp1", names) == "v0 = v1 + v0"


def test_squash_map_points_back():
    text, idx = squash_map("a   b\n\tc")
    assert text == "a b c" and [idx[i] for i in (0, 2, 4)] == [0, 4, 7]


def test_ast_labels_have_no_user_identifiers():
    node = {"nodeType": "FunctionDefinition", "visibility": "public", "stateMutability": "payable", "src": "0:10:0",
            "body": {"nodeType": "Block", "src": "5:5:0", "statements": [
                {"nodeType": "ExpressionStatement", "src": "6:3:0", "expression": {
                    "nodeType": "MemberAccess", "memberName": "call", "src": "6:3:0",
                    "expression": {"nodeType": "Identifier", "name": "secretOwner", "src": "6:1:0"}}}]}}
    seq, trunc = A.sequence(node, False)
    labels = [A.LABELS[i] for i, _, _ in seq]
    assert not trunc and "FunctionDefinition:public/payable" in labels and "MemberAccess:call" in labels
    assert "Identifier" in labels and not any("secretOwner" in s for s in labels)
