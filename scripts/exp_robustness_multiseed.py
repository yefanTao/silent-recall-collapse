import os, json, time
import numpy as np, faiss
from scipy.stats import spearmanr

SEEDS = [int(s) for s in os.environ.get("SEEDS","0,1,2,3,4,5,6,7").split(",")]
D = int(os.environ.get("D","128")); NB = int(os.environ.get("N","20000")); K=int(os.environ.get("K","10"))
NQ = 3000
ROOT=os.path.join(os.path.dirname(__file__),"..")
OUT=os.path.join(ROOT,"outputs","validation"); os.makedirs(OUT,exist_ok=True)
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}",flush=True)
def _unit(x):
    n=np.linalg.norm(x,axis=1,keepdims=True); n[n==0]=1.0; return (x/n).astype("float32")
def rec(I,eI): return float(np.mean([len(set(a)&set(b)) for a,b in zip(I,eI)])/K)
def max_indeg(h,n):
    hn=h.hnsw; offs=faiss.vector_to_array(hn.offsets).astype(np.int64); nbr=faiss.vector_to_array(hn.neighbors).astype(np.int64)
    nb0=hn.nb_neighbors(0); deg=np.zeros(n,np.int64); idx=(offs[:n,None]+np.arange(nb0)[None,:]).ravel(); seg=nbr[idx]
    seg=seg[(seg>=0)&(seg<n)]; np.add.at(deg,seg,1); return int(deg.max())
def hnsw(x,metric=faiss.METRIC_INNER_PRODUCT):
    h=faiss.IndexHNSWFlat(x.shape[1],16,metric); h.hnsw.efConstruction=200; h.add(np.ascontiguousarray(x)); h.hnsw.efSearch=200; return h
def ivf(x,metric=faiss.METRIC_INNER_PRODUCT):
    nl=max(16,int(np.sqrt(len(x)))); iv=faiss.IndexIVFFlat(faiss.IndexFlatIP(x.shape[1]),x.shape[1],nl,metric)
    iv.train(np.ascontiguousarray(x)); iv.add(np.ascontiguousarray(x)); iv.nprobe=64; return iv

def C1_law(seed):
    rng=np.random.default_rng(seed); base=_unit(rng.standard_normal((NB,D)))
    q=np.ascontiguousarray(_unit(base+rng.normal(0,1e-3,base.shape)))
    _,ex=faiss.knn(q,np.ascontiguousarray(base),K,metric=faiss.METRIC_INNER_PRODUCT)
    sink=_unit(rng.standard_normal((1,D)))[0]; pts=[]
    for sf in [0.0,0.02,0.05,0.1,0.2,0.5]:
        S=int(sf*NB); x=base if S==0 else np.ascontiguousarray(np.concatenate([base,np.repeat(sink[None,:],S,0)]))
        h=hnsw(x); _,Ih=h.search(q,K); hr=rec(Ih,ex); mi=max_indeg(h,len(x))
        iv_=ivf(x); _,Ii=iv_.search(q,K); ir=rec(Ii,ex); pts.append((mi,hr,ir))
    a=[np.log10(max(m,1)) for m,_,_ in pts]; rho,_=spearmanr(a,[h for _,h,_ in pts])
    return {"rho_logindeg_hnsw":round(float(rho),3),"hnsw_max_recall":max(h for _,h,_ in pts),
            "hnsw_min_recall":min(h for _,h,_ in pts),"ivf_min_recall":min(i for _,_,i in pts)}

def C2_dupdim(seed):
    rng=np.random.default_rng(seed)
    # dup arm
    base=_unit(rng.standard_normal((NB,D))); q=np.ascontiguousarray(_unit(base+rng.normal(0,1e-3,base.shape)))
    _,ex=faiss.knn(q,np.ascontiguousarray(base),K,metric=faiss.METRIC_INNER_PRODUCT); sink=_unit(rng.standard_normal((1,D)))[0]
    S=int(0.5*NB); x=np.ascontiguousarray(np.concatenate([base,np.repeat(sink[None,:],S,0)])); h=hnsw(x); _,Ih=h.search(q,K)
    dup_indeg=max_indeg(h,len(x)); dup_rec=rec(Ih,ex)
    # dim arm: low-rank -> full rank, no dups
    dim_indeg=[]; dim_rec=[]
    for rank in [2,8,32,128]:
        basis=rng.standard_normal((rank,D)).astype("float32"); Z=rng.standard_normal((NB,rank)).astype("float32")
        Xr=_unit(Z@basis); qq=np.ascontiguousarray(_unit(Xr[rng.choice(NB,NQ,replace=False)]))
        _,exr=faiss.knn(qq,np.ascontiguousarray(Xr),K,metric=faiss.METRIC_INNER_PRODUCT)
        h=hnsw(Xr); _,I=h.search(qq,K); dim_indeg.append(max_indeg(h,NB)); dim_rec.append(rec(I,exr))
    return {"dup_indeg":dup_indeg,"dup_recall":round(dup_rec,4),
            "dim_max_indeg":max(dim_indeg),"dim_min_recall":round(min(dim_rec),4),
            "separation_holds":bool(dup_indeg>1000 and dup_rec<0.85 and max(dim_indeg)<200 and min(dim_rec)>0.90)}

def C3_exactnear(seed):
    rng=np.random.default_rng(seed); bg=_unit(rng.standard_normal((NB,D)))
    q=np.ascontiguousarray(_unit(bg+rng.normal(0,1e-3,bg.shape)))
    _,ex=faiss.knn(q,np.ascontiguousarray(bg),K,metric=faiss.METRIC_INNER_PRODUCT)
    hub=_unit(rng.standard_normal((1,D)))[0]; S=10000
    out={}
    for tag,eps in [("exact",0.0),("near",1e-4)]:
        if eps==0: clones=np.repeat(hub[None,:],S,0)
        else:
            g=rng.normal(0,eps,(S,D)).astype("float32"); clones=_unit(hub[None,:]+g)
        x=np.ascontiguousarray(np.concatenate([bg,clones])); h=hnsw(x); _,I=h.search(q,K)
        out[tag]={"indeg":max_indeg(h,len(x)),"recall":round(rec(I,ex),4)}
    out["dichotomy_holds"]=bool(out["exact"]["recall"]<0.85 and out["near"]["recall"]>0.92)
    return out

res={"config":{"seeds":SEEDS,"D":D,"N":NB,"K":K},"C1":[],"C2":[],"C3":[]}
for s in SEEDS:
    log(f"seed {s}: C1 law"); res["C1"].append({"seed":s,**C1_law(s)})
    log(f"seed {s}: C2 dup-vs-dim"); res["C2"].append({"seed":s,**C2_dupdim(s)})
    log(f"seed {s}: C3 exact-vs-near"); res["C3"].append({"seed":s,**C3_exactnear(s)})
    c1=res["C1"][-1]; c2=res["C2"][-1]; c3=res["C3"][-1]
    log(f"   C1 rho={c1['rho_logindeg_hnsw']} ivfmin={c1['ivf_min_recall']:.3f} | C2 sep={c2['separation_holds']} | C3 dich={c3['dichotomy_holds']}")

# aggregate
def agg(key,field):
    vals=[r[field] for r in res[key]]; return {"mean":round(float(np.mean(vals)),4),"min":round(float(np.min(vals)),4),"max":round(float(np.max(vals)),4)}
res["summary"]={
  "C1_rho": agg("C1","rho_logindeg_hnsw"), "C1_ivf_min": agg("C1","ivf_min_recall"),
  "C1_rho_always_negative": bool(all(r["rho_logindeg_hnsw"]<-0.5 for r in res["C1"])),
  "C2_separation_all_seeds": bool(all(r["separation_holds"] for r in res["C2"])),
  "C2_dup_recall": agg("C2","dup_recall"), "C2_dim_min_recall": agg("C2","dim_min_recall"),
  "C2_dim_max_indeg": agg("C2","dim_max_indeg"),
  "C3_dichotomy_all_seeds": bool(all(r["dichotomy_holds"] for r in res["C3"])),
  "C3_exact_recall_mean": round(float(np.mean([r["exact"]["recall"] for r in res["C3"]])),4),
  "C3_near_recall_mean": round(float(np.mean([r["near"]["recall"] for r in res["C3"]])),4),
}
json.dump(res,open(os.path.join(OUT,"robustness_multiseed.json"),"w"),indent=2)
log("saved outputs/validation/robustness_multiseed.json")
print(json.dumps(res["summary"],indent=2))
