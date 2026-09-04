import re, glob
S="/tmp/claude-501/-Users-mohit/88a9b3f0-695b-4f6f-b89f-9254d59d5181/scratchpad"
ARCHS=["apple-m4","hsw","skx","sve512"]
WIDTH={"apple-m4":2,"hsw":4,"skx":8,"sve512":8}
WANT=["volume","derivative","localFlux","localFluxNodal","neighboringFlux","nodalFlux"]

def nums(s): return [int(x) for x in re.findall(r'\d+', s)]

def parse(path):
    txt=open(path).read()
    out={}
    for m in re.finditer(r'struct\s+([A-Za-z_][A-Za-z0-9_]*)\s*\{', txt):
        name=m.group(1); tail=txt[m.end():m.end()+3000]
        # scalar form
        nz=re.search(r'NonZeroFlops\s*=\s*(\d+)\s*;',tail)
        hw=re.search(r'HardwareFlops\s*=\s*(\d+)\s*;',tail)
        if nz and hw:
            v=[int(nz.group(1)),int(hw.group(1))]
        else:
            nza=re.search(r'NonZeroFlops\[\]\s*=\s*\{([^}]*)\}',tail)
            hwa=re.search(r'HardwareFlops\[\]\s*=\s*\{([^}]*)\}',tail)
            if not (nza and hwa): continue
            v=[sum(nums(nza.group(1))), sum(nums(hwa.group(1)))]
        cur=out.setdefault(name,[0,0]); cur[0]+=v[0]; cur[1]+=v[1]
    return out

data={a:parse(glob.glob(f"{S}/gen-{a}/equation-elastic-4-double/kernel.h")[0]) for a in ARCHS}

print("order 4, elastic, f64 — hardware/non-zero FLOP ratio (waste % of issued)\n")
hdr=f"{'kernel':20s}"+"".join(f"{a+' ('+str(WIDTH[a])+'d)':>18s}" for a in ARCHS)
print(hdr); print("-"*len(hdr))
for k in WANT:
    row=f"{k:20s}"
    for a in ARCHS:
        v=data[a].get(k)
        if v and v[0]:
            r=v[1]/v[0]; row+=f"{r:9.3f} ({(r-1)/r*100:4.1f}%)"
        else: row+=f"{'--':>18s}"
    print(row)
