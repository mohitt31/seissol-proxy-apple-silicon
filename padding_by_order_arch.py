import re, glob
import os, sys
S = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()  # directory holding the gen-* outputs
def nums(s): return [int(x) for x in re.findall(r'\d+', s)]
def parse(path):
    txt=open(path).read(); out={}
    for m in re.finditer(r'struct\s+([A-Za-z_][A-Za-z0-9_]*)\s*\{', txt):
        name=m.group(1); tail=txt[m.end():m.end()+3000]
        nz=re.search(r'NonZeroFlops\s*=\s*(\d+)\s*;',tail); hw=re.search(r'HardwareFlops\s*=\s*(\d+)\s*;',tail)
        if nz and hw: v=[int(nz.group(1)),int(hw.group(1))]
        else:
            a=re.search(r'NonZeroFlops\[\]\s*=\s*\{([^}]*)\}',tail); b=re.search(r'HardwareFlops\[\]\s*=\s*\{([^}]*)\}',tail)
            if not(a and b): continue
            v=[sum(nums(a.group(1))),sum(nums(b.group(1)))]
        c=out.setdefault(name,[0,0]); c[0]+=v[0]; c[1]+=v[1]
    return out
def path(a,o):
    p=glob.glob(f"{S}/gen-{a}-o{o}/equation-elastic-{o}-double/kernel.h")
    if not p: p=glob.glob(f"{S}/gen-{a}/equation-elastic-{o}-double/kernel.h")
    return p[0] if p else None

for kern in ["derivative","neighboringFlux"]:
    print(f"\n=== {kern} : hardware/non-zero FLOP ratio ===")
    print(f"{'order':>6}{'NEON (2 dbl)':>16}{'AVX-512 (8 dbl)':>18}")
    print("-"*40)
    for o in [2,3,4,5,6]:
        cells=[]
        for a in ["apple-m4","skx"]:
            p=path(a,o)
            if not p: cells.append("--"); continue
            d=parse(p).get(kern)
            cells.append(f"{d[1]/d[0]:.3f} ({(1-d[0]/d[1])*100:.0f}%)" if d and d[0] else "--")
        print(f"{o:>6}{cells[0]:>16}{cells[1]:>18}")
