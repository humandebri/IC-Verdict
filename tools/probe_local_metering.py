"""PocketIC check: identical arithmetic, 40 additional local operations."""
import json
from pathlib import Path
import sys
import wasmtime
from pocket_ic import PocketIC
pic=PocketIC();rows=[]
for copies in (0,20):
    wat='''(module
      (import "ic0" "performance_counter" (func $count (param i32) (result i64)))
      (import "ic0" "msg_reply_data_append" (func $append (param i32 i32)))
      (import "ic0" "msg_reply" (func $reply))
      (memory 1)
      (func $probe (param i32 i32) (result i32) (local i32)
        local.get 0 COPIES local.get 1 i32.add)
      (func (export "canister_update run")
        i32.const 0 i32.const 123 i32.const 456 call $probe i32.store
        i32.const 8 i32.const 0 call $count i64.store
        i32.const 0 i32.const 16 call $append call $reply))'''.replace('COPIES','local.set 2 local.get 2 '*copies)
    cid=pic.create_canister();pic.add_cycles(cid,100_000_000_000_000);pic.install_code(cid,bytes(wasmtime.wat2wasm(wat)),[])
    data=pic.update_call(cid,'run',b'')
    rows.append({'extra_local_ops':copies*2,'result':int.from_bytes(data[:4],'little'),'instructions':int.from_bytes(data[8:16],'little')})
assert all(r['result']==579 for r in rows)
assert rows[1]['instructions']-rows[0]['instructions']==40
out={'method':'PocketIC performance_counter(0); same arithmetic, different local bookkeeping','samples':rows,'instruction_delta':40}
Path(sys.argv[1]).write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
