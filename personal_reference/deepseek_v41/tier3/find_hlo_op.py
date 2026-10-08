import sys
from libtorch_neuronx_lite.pyhlo.service import hlo_pb2
m = hlo_pb2.HloModuleProto()
m.ParseFromString(open(sys.argv[1], "rb").read())
want = sys.argv[2]
byid = {}
for c in m.computations:
    for ins in c.instructions:
        byid[ins.id] = (c.name, ins)
def shp(s):
    return f"{s.element_type}{list(s.dimensions)}"
dots = [(cn, i) for cn, i in byid.values() if i.opcode == "dot"]
print("dots:", len(dots), "computations:", len(m.computations))
for cn, ins in byid.values():
    if ins.name == want or ins.name.endswith(want) or str(ins.id) == want.split(".")[-1]:
        ops = [byid[o][1] for o in ins.operand_ids if o in byid]
        print(cn, ins.name, ins.id, ins.opcode, shp(ins.shape), "dnums:", ins.dot_dimension_numbers, "metadata:", ins.metadata)
        for o in ops:
            print("   operand", o.name, o.opcode, shp(o.shape), o.metadata.op_type, o.metadata.op_name, o.metadata.source_file, o.metadata.source_line)

def walk(i, depth, maxd):
    cn, ins = byid[i]
    print("  " * depth + f"{ins.name} {ins.opcode} {shp(ins.shape)}" + (f" {ins.custom_call_target}" if ins.opcode == "custom-call" else ""))
    if depth < maxd:
        for o in ins.operand_ids:
            if o in byid:
                walk(o, depth + 1, maxd)
for cn, ins in byid.values():
    if ins.name == want:
        walk(ins.id, 0, int(sys.argv[3]) if len(sys.argv) > 3 else 6)
        users = [u for _, u in byid.values() if ins.id in u.operand_ids]
        for u in users:
            print("USER", u.name, u.opcode, shp(u.shape))
            for uu in [x for _, x in byid.values() if u.id in x.operand_ids][:3]:
                print("  USER2", uu.name, uu.opcode, shp(uu.shape))
