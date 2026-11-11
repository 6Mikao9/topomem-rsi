"""Compare v12's lazy feature binding with the current v11 choice and action paths."""
from collections import Counter
import importlib.util
from itertools import product
import heapq
import os
from pathlib import Path
import random
import sys
from types import SimpleNamespace
import unittest

HERE = Path(__file__).resolve().parents[1]
REPO = Path(os.environ.get("TOPOMEM_V11_ROOT", r"D:\ws\01_研究\实验环境\topomem-rsi"))
sys.path.insert(0, str(REPO))
import integrations.topology_memory as integration
from rsi_memory.compiled_memory import CompiledProgram
from rsi_memory.memory_program import MemoryProgram
from rsi_memory.topology import Link, Route

integration.__path__.insert(0, str(HERE / "integrations/topology_memory"))


def load_runtime(name, path):
    spec = importlib.util.spec_from_file_location(f"{integration.__name__}.{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


old = load_runtime("_runtime_v11_reference", REPO / "integrations/topology_memory/torch_runtime.py")
new = load_runtime("_runtime_v12_prototype", HERE / "integrations/topology_memory/torch_runtime.py")
from integrations.topology_memory.torch_dynamic_features import DynamicFeatureView


def const(value):
    return {"op":"const", "value":value}


def feature(name):
    return {"op":"feature", "name":name}


def op(name, *args):
    return {"op":name, "args":list(args)}


def rule(event, action, when, score):
    return {"event":event, "action":action, "when":when, "score":score}


def compile_rules(*rules):
    return CompiledProgram(MemoryProgram.from_dict({"schema":1,"kind":"kv_memory_program","rules":list(rules)}))


def guarded_program():
    one, zero = const(1), const(0)
    partial = op("if_lt", feature("valid_rows"), feature("block_rows"), feature("target_is_local"), zero)
    backup_score = op("add", feature("route_transfer_ns"), op("add", feature("recency_steps"),
                  op("add", feature("logical_refs"), op("add", feature("age_ns"), feature("page_accesses")))))
    return compile_rules(
        rule("allocate","place",partial,zero),
        rule("allocate","place",op("if_lt",feature("target_free_bytes"),feature("bytes"),zero,one),backup_score),
        rule("access","replicate",one,op("add",feature("route_transfer_ns"),feature("logical_refs"))),
        rule("access","prefetch",one,feature("route_transfer_ns")),
        rule("allocate","evict",op("if_lt",feature("logical_refs"),const(2),one,zero),
             op("add",feature("recency_steps"),feature("target_is_gpu"))),
        rule("allocate","demote",op("if_lt",zero,feature("target_has_copy"),one,zero),
             op("add",feature("page_accesses"),feature("last_use_step"))),
    )


def thin_program():
    return compile_rules(rule("allocate","place",const(1),op("add",feature("route_transfer_ns"),feature("logical_refs"))))


class FakeSlot:
    def __init__(self):
        self.payload = None

    def copy_(self, source, *, non_blocking):
        if not non_blocking:
            raise AssertionError("expected the ordinary staged copy path")
        self.payload = source.payload


class FakePool:
    def __init__(self, capacity, used=()):
        self.capacity = capacity
        self.used = set(used)
        self.free = sorted(set(range(capacity))-self.used)
        self.tensor = [FakeSlot() for _ in range(capacity)]

    def take(self):
        slot = heapq.heappop(self.free)
        self.used.add(slot)
        return slot

    def release(self, slot):
        self.used.remove(slot)
        heapq.heappush(self.free,slot)


def make_route(source, target, *middle):
    nodes = (source,) if source==target else (source,*middle,target)
    links = tuple(Link(f"{a}-{b}",a,b,1_000_000_000,1,(),"pinned_host") for a,b in zip(nodes,nodes[1:]))
    return Route(nodes,links)


def runtime(version, program, free):
    """Use the actual v11/v12 methods on CPU metadata, without pretending to execute DMA."""
    obj = version._Runtime.__new__(version._Runtime)
    obj.program, obj.local, obj.arm, obj.verify = program, "gpu0", "generated", False
    obj.work_counts = None
    obj.pools = {node:FakePool(2, () if available else (0,1))
                 for node,available in zip(("gpu0","gpu1","host"),free,strict=True)}
    obj.workload = SimpleNamespace(block_bytes=64)
    obj.clock,obj.ordinal = 100,12
    obj.reads = Counter({(0,0):4,(0,1):1})
    obj.versions={}
    obj.misses,obj.link_bytes,obj.group_bytes=Counter(),Counter(),Counter()
    obj.transfer_wall_ns=0
    obj._sync=lambda nodes:None
    obj.static_routes={}
    obj.pressure_keys={}
    obj.routes={}
    obj.rule_routes={}
    obj.rule_handoffs={}
    obj.rule_features={}
    obj.rule_nodes={}
    for src,dst in product(obj.pools,repeat=2):
        middle = ("host",) if src!=dst and src.startswith("gpu") and dst.startswith("gpu") else ()
        obj.routes[src,dst] = (make_route(src,dst,*middle),)
        for group,names in program.features.items():
            route = obj.routes[src,dst][0]
            handoff = Route((src,dst),()) if group[1]=="demote" and src!=dst else None
            if handoff is not None:
                obj.rule_handoffs[group,src,dst]=handoff
            dynamic = [(name,version._FEATURE_READERS[name]) for name in sorted(names & version._DYNAMIC_FEATURES)]
            for option in (route,) if handoff is None else (route,handoff):
                static = {"route_transfer_ns":100*len(option.links), "block_rows":8, "bytes":64,
                          "target_is_local":int(dst=="gpu0"),"target_is_gpu":int(dst!="host")}
                obj.rule_features[group,option]=(static,dict(dynamic) if version is new else tuple(dynamic),128)
            obj.rule_routes[group,src,dst]=(route,)
            obj.rule_nodes[group]=set(obj.pools)
    obj.timeline=[]
    obj.pool_nodes=frozenset(obj.pools)
    obj.retired=set()
    obj.residents={node:set() for node in obj.pools}
    obj.replicas={node:set() for node in obj.pools}
    obj.local_owners=set()
    obj.blocks={}
    return obj


def block(version, name="a", index=0, valid=3, owner="gpu0", copies=None, refs=1, use=3):
    spec=SimpleNamespace(id=name,index=index,layer=0,valid_rows=valid)
    return version._Block(spec,owner,7,{"gpu0":0} if copies is None else copies,refs,2,use)


def outcome(fn):
    try:
        return ("value",fn())
    except (KeyError, ValueError) as error:
        return type(error).__name__,error.args


class DynamicFeatureLowering(unittest.TestCase):
    def test_allocate_and_access_placement_match_for_capacity_and_references(self):
        program=guarded_program()
        for free,valid,refs in product(product((False,True),repeat=3),(3,8),(1,2)):
            placements, movements = [], []
            for version in (old,new):
                engine=runtime(version,program,free)
                blocked=set()
                pick=engine._choose_place(block(version,valid=valid,refs=refs),blocked=blocked)
                placements.append((pick,blocked))
                blocked=set()
                move=engine._movement("access",block(version,owner="gpu1",copies={"gpu1":0,"host":0},refs=refs),
                                      demand=True,blocked=blocked)
                movements.append((move,blocked))
            self.assertEqual(placements[0],placements[1])
            self.assertEqual(movements[0],movements[1])

    def test_pressure_evict_and_owner_handoff_preserve_actions_and_ownership(self):
        for protected in ((),("b",)):
            observations=[]
            for version in (old,new):
                engine=runtime(version,guarded_program(),(False,True,False))
                a=block(version,copies={"gpu0":0,"host":0},refs=2)
                b=block(version,name="b",index=1,owner="host",copies={"gpu0":1,"host":1})
                engine.blocks={"a":a,"b":b}
                engine.residents.update(gpu0={"a","b"},host={"a","b"})
                engine.replicas.update(gpu0={"b"},host={"a"})
                engine.local_owners={"a"}
                demoted=set()
                self.assertTrue(engine._relieve("allocate",protected=protected,demote_nodes=("gpu0",),demoted=demoted))
                observations.append((engine.timeline,[(b.owner,b.copies) for b in (a,b)],
                                     sorted(engine.pools["gpu0"].free),demoted))
            self.assertEqual(observations[0],observations[1])
            self.assertEqual(observations[0][0][-1]["action"],"demote_owner_handoff" if protected else "evict")
            self.assertEqual(observations[0][-1],{"a"} if protected else set())

    def test_dynamic_peer_relay_preserves_ordered_place_fetch_and_selected_payload(self):
        one, zero = const(1), const(0)
        peer = op("if_lt",feature("target_is_local"),one,
                  op("if_lt",feature("valid_rows"),const(4),feature("target_is_gpu"),zero),zero)
        program = compile_rules(
            rule("allocate","place",peer,feature("route_transfer_ns")),
            rule("access","replicate",one,op("add",feature("logical_refs"),feature("route_transfer_ns"))),
        )
        traces=[]
        for version in (old,new):
            engine=runtime(version,program,(True,True,True))
            slot=engine.pools["gpu0"].take()
            state=block(version,copies={"gpu0":slot})
            engine.blocks={"a":state}
            engine.residents["gpu0"].add("a")
            engine.pools["gpu0"].tensor[slot].payload=b"immutable K/V test payload"
            engine._place_batch([state])
            self.assertEqual(state.owner,"gpu1")
            self.assertEqual(set(state.copies),{"gpu1"})
            engine._fetch([state])
            traces.append((engine.timeline,dict(engine.link_bytes),state.owner,set(state.copies),
                           engine._tensor(state,"gpu0").payload))
        self.assertEqual(traces[0],traces[1])
        self.assertEqual([action["action"] for action in traces[0][0]],["place_owner","place","producer_release","replicate"])
        self.assertEqual(traces[0][1],{"gpu0-host":64,"host-gpu1":64,"gpu1-host":64,"host-gpu0":64})
        self.assertEqual(traces[0][-1],b"immutable K/V test payload")

    def test_missing_and_arithmetic_errors_keep_rule_and_candidate_order(self):
        overflow=op("div",const(1e308),const(0))
        zero_div=op("div",const(2),const(-0.0))
        for predicate,score,fallback in (
            (const(1),op("if_lt",feature("valid_rows"),const(8),zero_div,feature("accesses")),overflow),
            (const(1),op("if_lt",feature("valid_rows"),const(8),const(1),overflow),feature("accesses")),
            (op("if_lt",feature("valid_rows"),const(8),const(0),const(1)),feature("accesses"),overflow),
            (op("if_lt",feature("valid_rows"),const(8),const(0),const(1)),const(1),overflow),
        ):
            rules=compile_rules(rule("allocate","place",predicate,score),rule("allocate","place",const(1),fallback))
            for values in ((3,), (8,), (3,8), (8,3), (3,3)):
                a=[(i,{"valid_rows":rows}) for i,rows in enumerate(values)]
                b=[(i,DynamicFeatureView({},{"valid_rows":lambda value:value},(rows,))) for i,rows in enumerate(values)]
                self.assertEqual(outcome(lambda:rules.choose("allocate","place",a)),
                                 outcome(lambda:rules.choose("allocate","place",b)))
        rules=compile_rules(rule("allocate","place",const(1),feature("valid_rows")))
        picks=[("z",DynamicFeatureView({},{"valid_rows":lambda:2},())),
               ("a",DynamicFeatureView({},{"valid_rows":lambda:2},()))]
        self.assertEqual(rules.choose("allocate","place",picks),(0,"z"))

    def test_unused_rule_features_and_lazy_branches_are_never_read(self):
        calls=[]
        def reader(name,value):
            def read():
                calls.append(name)
                return value
            return read
        program=compile_rules(rule("allocate","place",feature("logical_refs"),
                                   op("if_lt",feature("valid_rows"),const(8),const(0),feature("recency_steps"))),
                              rule("allocate","place",feature("accesses"),feature("age_ns")))
        bound=DynamicFeatureView({},{
            name:reader(name,value) for name,value in
            {"logical_refs":1,"valid_rows":3,"recency_steps":5,"accesses":3,"age_ns":9}.items()
        },())
        self.assertEqual(program.choose("allocate","place",[("chosen",bound)]),(0,"chosen"))
        self.assertEqual(calls,["logical_refs","valid_rows"])

    def test_randomized_multirule_candidate_differential(self):
        rng=random.Random(5412)
        names=("logical_refs","target_free_bytes","route_hops","accesses","age_ns")
        def expr(depth):
            if depth==0 or rng.random()<0.45:
                return feature(rng.choice(names)) if rng.random()<0.7 else const(rng.choice((-3,0,1,1e308)))
            name,count=rng.choice((("if_lt",4),("div",2),("add",2),("max",2),("mul",2)))
            return op(name,*(expr(depth-1) for _ in range(count)))
        for _ in range(200):
            policy=compile_rules(*(rule("allocate","place",expr(2),expr(2)) for _ in range(rng.randint(1,3))))
            a,b=[],[]
            for i in range(rng.randrange(6)):
                vals={name:rng.choice((-0.0,1,8,1e308,float("nan"))) for name in names if rng.random()>0.13}
                fields={"route_hops":vals["route_hops"]} if "route_hops" in vals else {}
                readers={name:lambda value=value:value for name,value in vals.items() if name not in fields}
                a.append((i,vals))
                b.append((i,DynamicFeatureView(fields,readers,())))
            self.assertEqual(outcome(lambda:policy.choose("allocate","place",a)),
                             outcome(lambda:policy.choose("allocate","place",iter(b))))


if __name__=="__main__":
    unittest.main()
