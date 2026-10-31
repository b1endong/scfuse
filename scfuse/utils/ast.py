import re

from scfuse.schema import MAX_AST_NODES

NODE_TYPES = (
    "SourceUnit", "PragmaDirective", "ImportDirective", "ContractDefinition", "InheritanceSpecifier",
    "UsingForDirective", "StructDefinition", "EnumDefinition", "EnumValue", "ParameterList", "OverrideSpecifier",
    "FunctionDefinition", "VariableDeclaration", "ModifierDefinition", "ModifierInvocation", "EventDefinition",
    "ErrorDefinition", "ElementaryTypeName", "UserDefinedTypeName", "FunctionTypeName", "Mapping", "ArrayTypeName",
    "InlineAssembly", "Block", "UncheckedBlock", "PlaceholderStatement", "IfStatement", "TryCatchClause",
    "TryStatement", "WhileStatement", "DoWhileStatement", "ForStatement", "Continue", "Break", "Return", "Throw",
    "EmitStatement", "RevertStatement", "VariableDeclarationStatement", "ExpressionStatement", "Conditional",
    "Assignment", "TupleExpression", "UnaryOperation", "BinaryOperation", "FunctionCall", "FunctionCallOptions",
    "NewExpression", "MemberAccess", "IndexAccess", "IndexRangeAccess", "PrimaryExpression",
    "ElementaryTypeNameExpression", "Literal", "Identifier", "IdentifierPath", "UserDefinedValueTypeDefinition",
    "YulBlock", "YulAssignment", "YulVariableDeclaration", "YulExpressionStatement", "YulFunctionCall",
    "YulIdentifier", "YulLiteral", "YulIf", "YulSwitch", "YulCase", "YulForLoop", "YulBreak", "YulContinue",
    "YulLeave", "YulFunctionDefinition", "YulTypedName")
OPERATORS = ("+", "-", "*", "/", "%", "**", "<<", ">>", "&", "|", "^", "<", ">", "<=", ">=", "==", "!=", "&&", "||",
             "!", "~", "++", "--", "delete", "=", "+=", "-=", "*=", "/=", "%=", "|=", "&=", "^=", "<<=", ">>=")
MEMBERS = ("call", "send", "transfer", "delegatecall", "callcode", "staticcall", "value", "gas", "balance",
           "timestamp", "number", "origin", "sender", "coinbase", "difficulty", "prevrandao", "gaslimit", "blockhash",
           "length", "push", "pop", "selector", "encode", "encodePacked", "encodeWithSelector", "encodeWithSignature",
           "decode", "data", "sig", "codehash", "code", "basefee", "chainid", "gasprice")
BUILTINS = ("msg", "tx", "block", "now", "this", "super", "keccak256", "sha3", "sha256", "ripemd160", "ecrecover",
            "require", "assert", "revert", "selfdestruct", "suicide", "blockhash", "gasleft", "abi", "addmod",
            "mulmod", "type")
ETYPES = ("uint", "int", "address", "bool", "bytes", "bytesN", "string", "fixed", "ufixed")
CALL_KINDS = ("functionCall", "typeConversion", "structConstructorCall")
LIT_KINDS = ("number", "bool", "string", "hexString", "unicodeString")
FN_ATTRS = tuple(f"{v}/{m}" for v in ("public", "external", "internal", "private")
                 for m in ("nonpayable", "payable", "view", "pure"))
LOCATIONS = ("default", "storage", "memory", "calldata")
YUL = ("call", "callcode", "delegatecall", "staticcall", "sstore", "sload", "selfdestruct", "create", "create2",
       "timestamp", "number", "origin", "caller", "callvalue", "extcodesize", "balance", "returndatacopy",
       "returndatasize", "calldataload", "calldatacopy", "mstore", "mload", "return", "revert", "keccak256", "gas")

_ATTRS = {"BinaryOperation": OPERATORS, "UnaryOperation": OPERATORS, "Assignment": OPERATORS,
          "MemberAccess": MEMBERS, "Identifier": BUILTINS, "ElementaryTypeName": ETYPES,
          "ElementaryTypeNameExpression": ETYPES, "FunctionCall": CALL_KINDS, "Literal": LIT_KINDS,
          "FunctionDefinition": FN_ATTRS, "VariableDeclaration": LOCATIONS, "YulFunctionCall": YUL,
          "YulIdentifier": YUL}
LABELS = ("<pad>", "<unk>") + NODE_TYPES + tuple(f"{t}:{a}" for t, vals in _ATTRS.items() for a in vals)
LABEL_ID = {s: i for i, s in enumerate(LABELS)}
SKIP = {"StructuredDocumentation"}


def parse_src(src: str) -> tuple[int, int, int]:
    s, n, f = (src or "-1:-1:-1").split(":")[:3]
    return int(s), int(n), int(f)


def is_legacy(node: dict) -> bool:
    return "nodeType" not in node and "name" in node


def node_type(node: dict, legacy: bool) -> str:
    return node.get("name") if legacy else node.get("nodeType")


def children(node: dict, legacy: bool) -> list[dict]:
    if legacy:
        return [c for c in node.get("children", []) if isinstance(c, dict)]
    out = []
    for k, v in node.items():
        if k in ("typeDescriptions", "documentation"):
            continue
        if isinstance(v, dict) and "nodeType" in v:
            out.append(v)
        elif isinstance(v, list):
            out.extend(x for x in v if isinstance(x, dict) and "nodeType" in x)
    out = [c for c in out if c.get("nodeType") not in SKIP]
    return sorted(out, key=lambda c: parse_src(c.get("src"))[0])


def attrs(node: dict, legacy: bool) -> dict:
    return node.get("attributes") or {} if legacy else node


def _etype(name) -> str | None:
    if isinstance(name, dict):
        name = name.get("name")
    if not isinstance(name, str):
        return None
    name = name.split()[0]
    if re.fullmatch(r"bytes\d+|byte", name):
        return "bytesN"
    return re.sub(r"\d+(x\d+)?$", "", name)


def mutability(a: dict) -> str:
    if a.get("stateMutability"):
        return a["stateMutability"]
    return "payable" if a.get("payable") else "view" if a.get("constant") else "nonpayable"


def label(node: dict, legacy: bool) -> int:
    t, a = node_type(node, legacy), attrs(node, legacy)
    v = None
    if t in ("BinaryOperation", "UnaryOperation", "Assignment"):
        v = a.get("operator")
    elif t == "MemberAccess":
        v = a.get("memberName", a.get("member_name"))
    elif t == "Identifier":
        v = a.get("name", a.get("value"))
    elif t == "ElementaryTypeName":
        v = _etype(a.get("name"))
    elif t == "ElementaryTypeNameExpression":
        v = _etype(a.get("typeName", a.get("value")))
    elif t == "FunctionCall":
        v = a.get("kind") or ("typeConversion" if a.get("type_conversion") else "functionCall")
    elif t == "Literal":
        v = a.get("kind") or a.get("token")
    elif t == "FunctionDefinition":
        v = f"{a.get('visibility', 'public')}/{mutability(a)}"
    elif t == "VariableDeclaration":
        v = a.get("storageLocation", "default")
    elif t == "YulFunctionCall":
        fn = a.get("functionName")
        v = fn.get("name") if isinstance(fn, dict) else None
    elif t == "YulIdentifier":
        v = a.get("name")
    return LABEL_ID.get(f"{t}:{v}", LABEL_ID.get(t, 1))


def sequence(node: dict, legacy: bool, max_nodes: int = MAX_AST_NODES) -> tuple[list[list[int]], bool]:
    out, stack = [], [(node, 0, 0)]
    while stack:
        if len(out) >= max_nodes:
            return out, True
        n, depth, rank = stack.pop()
        out.append([label(n, legacy), min(depth, 63), min(rank, 15)])
        kids = children(n, legacy)
        for i in range(len(kids) - 1, -1, -1):
            stack.append((kids[i], depth + 1, i))
    return out, False


def walk(node: dict, legacy: bool):
    stack = [node]
    while stack:
        n = stack.pop()
        yield n
        stack.extend(reversed(children(n, legacy)))
