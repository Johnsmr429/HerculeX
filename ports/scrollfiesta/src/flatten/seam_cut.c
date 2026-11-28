#define _USE_MATH_DEFINES
#include "seam_cut.h"
#include "slim_refine.h"
#include "mesh_topo.h"
#include "../common/ves_platform.h"

#include <assert.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define NEXT(c) (((c)/3)*3 + (((c)%3)+1)%3)
#define PREV(c) (((c)/3)*3 + (((c)%3)+2)%3)

typedef struct {
    size_t         nv, nf, nc;
    const float   *verts;
    const int32_t *faces;
    int32_t       *twin;
    int32_t       *voff;
    int32_t       *vcor;
    int            nonmanifold;
} Conn;

typedef struct { uint64_t key; int32_t corner; } EKey;
static int cmp_ekey(const void *a, const void *b)
{
    uint64_t x = ((const EKey *)a)->key, y = ((const EKey *)b)->key;
    return (x < y) ? -1 : (x > y ? 1 : 0);
}

static void conn_build(Arena_T arena, const float *verts, size_t nv,
                       const int32_t *faces, size_t nf, Conn *c)
{
    c->nv = nv; c->nf = nf; c->nc = 3*nf; c->verts = verts; c->faces = faces;
    c->nonmanifold = 0;
    c->twin = (int32_t *)ARENA_ALLOC(arena, (size_t)(c->nc * sizeof(int32_t)));
    for (size_t i = 0; i < c->nc; i++) c->twin[i] = -1;

    EKey *ek = (EKey *)ARENA_ALLOC(arena, (size_t)(c->nc * sizeof(EKey)));
    for (size_t cc = 0; cc < c->nc; cc++) {
        int32_t a = faces[cc], b = faces[NEXT(cc)];
        uint64_t lo = (uint64_t)(a < b ? a : b), hi = (uint64_t)(a < b ? b : a);
        ek[cc].key = lo * (uint64_t)nv + hi;
        ek[cc].corner = (int32_t)cc;
    }
    qsort(ek, c->nc, sizeof(EKey), cmp_ekey);
    size_t i = 0;
    while (i < c->nc) {
        size_t j = i; while (j < c->nc && ek[j].key == ek[i].key) j++;
        if (j - i == 2) {
            c->twin[ek[i].corner] = ek[i+1].corner;
            c->twin[ek[i+1].corner] = ek[i].corner;
        } else if (j - i > 2) {
            c->nonmanifold = 1;
        }
        i = j;
    }

    c->voff = (int32_t *)ARENA_CALLOC(arena, (size_t)(nv + 1), sizeof(int32_t));
    for (size_t cc = 0; cc < c->nc; cc++) c->voff[faces[cc] + 1]++;
    for (size_t v = 0; v < nv; v++) c->voff[v+1] += c->voff[v];
    c->vcor = (int32_t *)ARENA_ALLOC(arena, (size_t)(c->nc * sizeof(int32_t)));
    int32_t *cur = (int32_t *)ARENA_ALLOC(arena, (size_t)(nv * sizeof(int32_t)));
    for (size_t v = 0; v < nv; v++) cur[v] = c->voff[v];
    for (size_t cc = 0; cc < c->nc; cc++) { int32_t v = faces[cc]; c->vcor[cur[v]++] = (int32_t)cc; }
}

static void vsub(const float *p, const float *q, double o[3])
{ o[0]=(double)p[0]-q[0]; o[1]=(double)p[1]-q[1]; o[2]=(double)p[2]-q[2]; }
static double vdot(const double a[3], const double b[3])
{ return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]; }
static double vlen(const double a[3]) { return sqrt(vdot(a,a)); }
static double corner_angle(const float *p0, const float *p1, const float *p2)
{
    double a[3], b[3]; vsub(p1,p0,a); vsub(p2,p0,b);
    double la=vlen(a), lb=vlen(b);
    if (la<1e-20 || lb<1e-20) return 0.0;
    double c = vdot(a,b)/(la*lb);
    if (c>1.0)c=1.0; if(c<-1.0)c=-1.0;
    return acos(c);
}

#include "seam_cut_direction.inc"

typedef struct { uint64_t *slot; size_t cap, n; size_t nv; } SeamSet;
static void seam_init(Arena_T arena, SeamSet *s, size_t nv, size_t hint)
{
    size_t cap = 16; while (cap < hint*2) cap <<= 1;
    s->slot = (uint64_t *)ARENA_ALLOC(arena, (size_t)(cap*sizeof(uint64_t)));
    for (size_t i=0;i<cap;i++) s->slot[i] = UINT64_MAX;
    s->cap = cap; s->n = 0; s->nv = nv;
}
static uint64_t seam_key(const SeamSet *s, int32_t a, int32_t b)
{ uint64_t lo=(uint64_t)(a<b?a:b), hi=(uint64_t)(a<b?b:a); return lo*(uint64_t)s->nv+hi; }
static int seam_has(const SeamSet *s, int32_t a, int32_t b)
{
    uint64_t k=seam_key(s,a,b), m=s->cap-1, i=k&m;
    while (s->slot[i]!=UINT64_MAX){ if(s->slot[i]==k) return 1; i=(i+1)&m; }
    return 0;
}
static void seam_add(SeamSet *s, int32_t a, int32_t b)
{
    if (a==b) return;
    uint64_t k=seam_key(s,a,b), m=s->cap-1, i=k&m;
    while (s->slot[i]!=UINT64_MAX){ if(s->slot[i]==k) return; i=(i+1)&m; }
    s->slot[i]=k; s->n++;
}

static double distortion_r0(const Conn *c, int32_t n)
{
    double sum = 0.0;
    for (int32_t e = c->voff[n]; e < c->voff[n+1]; e++) {
        int32_t cc = c->vcor[e];
        const float *p0 = c->verts + 3*(size_t)n;
        const float *p1 = c->verts + 3*(size_t)c->faces[NEXT(cc)];
        const float *p2 = c->verts + 3*(size_t)c->faces[PREV(cc)];
        sum += corner_angle(p0, p1, p2);
    }
    return (2.0*M_PI - sum) / (2.0*M_PI);
}

static double longest_edge(const Conn *c, int32_t n)
{
    double best = 0.0;
    const float *pn = c->verts + 3*(size_t)n;
    for (int32_t e = c->voff[n]; e < c->voff[n+1]; e++) {
        int32_t cc = c->vcor[e];
        double d[3];
        vsub(c->verts + 3*(size_t)c->faces[NEXT(cc)], pn, d); double l=vlen(d);
        if (l>best) best=l;
        vsub(c->verts + 3*(size_t)c->faces[PREV(cc)], pn, d); l=vlen(d);
        if (l>best) best=l;
    }
    return best;
}

static double distortion_ri(Arena_T arena, const Conn *c, int32_t n, double R,
                            int32_t *facemark, int32_t mark, int32_t *queue,
                            int32_t *vstamp, int32_t *vid)
{
    const float *pn = c->verts + 3*(size_t)n;
    size_t head=0, tail=0;
    for (int32_t e = c->voff[n]; e < c->voff[n+1]; e++) {
        int32_t f = c->vcor[e] / 3;
        if (facemark[f] != mark) { facemark[f]=mark; queue[tail++]=f; }
    }
    size_t region_start = 0; (void)region_start;
    size_t qn = tail;
    while (head < qn) {
        int32_t f = queue[head++];
        for (int k=0;k<3;k++){
            int32_t cc = f*3+k; int32_t tw = c->twin[cc];
            if (tw < 0) continue;
            int32_t nf2 = tw/3;
            if (facemark[nf2]==mark) continue;
            int ok=1;
            for (int t=0;t<3;t++){ double d[3]; vsub(c->verts+3*(size_t)c->faces[nf2*3+t], pn, d);
                if (vlen(d) > R) { ok=0; break; } }
            if (ok){ facemark[nf2]=mark; queue[tail++]=nf2; qn=tail; }
        }
    }
    double sum=0.0; long nbe=0;
    Arena_Mark m2 = Arena_save(arena);
    int32_t *bp = (int32_t *)ARENA_ALLOC(arena, (size_t)(tail*3*2*sizeof(int32_t)));
    long bpn=0;
    for (size_t qi=0; qi<tail; qi++){
        int32_t f=queue[qi];
        for (int k=0;k<3;k++){
            int32_t cc=f*3+k; int32_t tw=c->twin[cc];
            int is_bnd = (tw<0) || (facemark[tw/3]!=mark);
            if (!is_bnd) continue;
            int32_t a=c->faces[cc], b=c->faces[NEXT(cc)];
            if (a==n || b==n) continue;
            sum += corner_angle(pn, c->verts+3*(size_t)a, c->verts+3*(size_t)b);
            bp[bpn++]=a; bp[bpn++]=b; nbe++;
        }
    }
    int loops_ok = 1;
    if (nbe >= 1) {
        int ncomp2 = 0;
        for (long i=0;i<bpn;i++){ int32_t v=bp[i]; if(vstamp[v]!=mark){vstamp[v]=mark; vid[v]=ncomp2++;} }
        int32_t *par = (int32_t *)ARENA_ALLOC(arena, (size_t)((size_t)ncomp2*sizeof(int32_t)));
        for (int k=0;k<ncomp2;k++) par[k]=k;
        for (long i=0;i<bpn;i+=2){
            int32_t x=vid[bp[i]], y=vid[bp[i+1]];
            while(par[x]!=x)x=par[x]; while(par[y]!=y)y=par[y];
            if(x!=y)par[x]=y;
        }
        long nloops=0; for(int k=0;k<ncomp2;k++){ int r=k; while(par[r]!=r)r=par[r]; if(r==k)nloops++; }
        if (nloops != 1) loops_ok = 0;
    } else loops_ok = 0;
    Arena_restore(arena, m2);
    if (!loops_ok) return NAN;
    return (2.0*M_PI - sum) / (2.0*M_PI);
}

static void compute_distortion(Arena_T arena, const Conn *c, int r_max,
                               double *Dr, int32_t *best_r)
{
    size_t nv=c->nv, nf=c->nf;
    int32_t *facemark = (int32_t *)ARENA_CALLOC(arena, (size_t)nf, sizeof(int32_t));
    int32_t *queue = (int32_t *)ARENA_ALLOC(arena, (size_t)(nf*sizeof(int32_t)));
    int32_t *vstamp = (int32_t *)ARENA_CALLOC(arena, (size_t)nv, sizeof(int32_t));
    int32_t *vid = (int32_t *)ARENA_ALLOC(arena, (size_t)(nv*sizeof(int32_t)));
    int32_t markctr = 0;

    for (size_t v=0; v<nv; v++) {
        double d0 = distortion_r0(c, (int32_t)v);
        double best = d0; int32_t br = 0;
        double le = longest_edge(c, (int32_t)v);
        for (int i=1; i<=r_max; i++) {
            markctr++;
            double di = distortion_ri(arena, c, (int32_t)v, (double)i*le,
                                      facemark, markctr, queue, vstamp, vid);
            if (di != di) break;
            if (di > best) { best = di; br = i; }
        }
        Dr[v] = best; best_r[v] = br;
    }

    for (size_t m=0; m<nv; m++) {
        if (Dr[m] <= 0.0) continue;
        for (int32_t e=c->voff[m]; e<c->voff[m+1]; e++) {
            int32_t cc=c->vcor[e];
            int32_t nb[2] = { c->faces[NEXT(cc)], c->faces[PREV(cc)] };
            for (int t=0;t<2;t++){
                int32_t n=nb[t];
                if (Dr[n] <= 0.0) continue;
                if (Dr[n] > Dr[m]) { Dr[m] = distortion_r0(c,(int32_t)m); best_r[m]=0; }
                else if (Dr[n] < Dr[m]) { Dr[n] = distortion_r0(c,n); best_r[n]=0; }
            }
        }
    }
}

typedef struct { double d; int32_t v; } HItem;
typedef struct { HItem *a; size_t n, cap; } Heap;
static void heap_push(Heap *h, double d, int32_t v){
    if (h->n >= h->cap) return;
    size_t i=h->n++; h->a[i].d=d; h->a[i].v=v;
    while(i>0){ size_t p=(i-1)/2; if(h->a[p].d<=h->a[i].d)break; HItem t=h->a[p];h->a[p]=h->a[i];h->a[i]=t; i=p; }
}
static HItem heap_pop(Heap *h){
    HItem top=h->a[0]; h->a[0]=h->a[--h->n]; size_t i=0;
    for(;;){ size_t l=2*i+1,r=2*i+2,s=i;
        if(l<h->n&&h->a[l].d<h->a[s].d)s=l; if(r<h->n&&h->a[r].d<h->a[s].d)s=r;
        if(s==i)break; HItem t=h->a[s];h->a[s]=h->a[i];h->a[i]=t; i=s; }
    return top;
}
static int32_t uf_find2(int32_t *p, int32_t x){ while(p[x]!=x){p[x]=p[p[x]];x=p[x];} return x; }

static long boundary_loop_terminals(Arena_T arena, const Conn *c,
                                    const char *is_boundary,
                                    char *is_terminal)
{
    Arena_Mark mark = Arena_save(arena);
    int32_t *parent = (int32_t *)ARENA_ALLOC(
        arena, (size_t)(c->nv*sizeof(int32_t)));
    int32_t *representative = (int32_t *)ARENA_ALLOC(
        arena, (size_t)(c->nv*sizeof(int32_t)));
    long nloops = 0;

    for (size_t v = 0; v < c->nv; v++) {
        parent[v] = (int32_t)v;
        representative[v] = -1;
    }
    for (size_t cc = 0; cc < c->nc; cc++) {
        int32_t a, b, ra, rb;
        if (c->twin[cc] >= 0) continue;
        a = c->faces[cc]; b = c->faces[NEXT(cc)];
        ra = uf_find2(parent, a); rb = uf_find2(parent, b);
        if (ra != rb) parent[rb] = ra;
    }
    for (size_t v = 0; v < c->nv; v++) {
        int32_t root;
        if (!is_boundary[v]) continue;
        root = uf_find2(parent, (int32_t)v);
        if (representative[root] < 0 ||
            (int32_t)v < representative[root])
            representative[root] = (int32_t)v;
    }
    for (size_t v = 0; v < c->nv; v++) {
        if (representative[v] < 0) continue;
        is_terminal[representative[v]] = 1;
        nloops++;
    }
    Arena_restore(arena, mark);
    return nloops;
}

typedef struct { double w; int32_t tu, tw, u, vv; } SCand;
static int cmp_scand(const void *a, const void *b)
{ double x=((const SCand*)a)->w, y=((const SCand*)b)->w; return (x<y)?-1:(x>y?1:0); }

static void steiner_tree(Arena_T arena, const Conn *c,
                         const char *is_terminal, SeamSet *seam)
{
    size_t nv=c->nv, nc=c->nc;
    const double EPS_BND = 1e-4;
    double  *dist = (double *)ARENA_ALLOC(arena,(size_t)(nv*sizeof(double)));
    int32_t *nrst = (int32_t *)ARENA_ALLOC(arena,(size_t)(nv*sizeof(int32_t)));
    int32_t *parent = (int32_t *)ARENA_ALLOC(arena,(size_t)(nv*sizeof(int32_t)));
    for (size_t v=0;v<nv;v++){ dist[v]=1e300; nrst[v]=-1; parent[v]=-1; }

    Heap h; h.cap=nc*4+1024; h.n=0;
    h.a=(HItem*)ARENA_ALLOC(arena,(size_t)(h.cap*sizeof(HItem)));
    for (size_t v=0;v<nv;v++) if(is_terminal[v]){ dist[v]=0; nrst[v]=(int32_t)v; heap_push(&h,0.0,(int32_t)v); }

    while (h.n>0){
        HItem it=heap_pop(&h); int32_t u=it.v;
        if (it.d>dist[u]+1e-12) continue;
        for (int32_t e=c->voff[u]; e<c->voff[u+1]; e++){
            int32_t cc=c->vcor[e];
            int32_t ws[2]={c->faces[NEXT(cc)], c->faces[PREV(cc)]};
            int32_t es[2]={cc, PREV(cc)};
            for (int t=0;t<2;t++){
                int32_t w=ws[t]; int is_bnd=(c->twin[es[t]]<0);
                double dd[3]; vsub(c->verts+3*(size_t)u, c->verts+3*(size_t)w, dd);
                double wt = is_bnd ? EPS_BND : vlen(dd);
                double nd = dist[u]+wt;
                if (nd < dist[w]-1e-12){ dist[w]=nd; nrst[w]=nrst[u]; parent[w]=u; heap_push(&h,nd,w); }
            }
        }
    }

    Arena_Mark m = Arena_save(arena);
    SCand *cand = (SCand *)ARENA_ALLOC(arena,(size_t)(nc*sizeof(SCand)));
    long ncand=0;
    for (size_t cc=0; cc<nc; cc++){
        int32_t tw=c->twin[cc];
        if (tw>=0 && (int32_t)cc>tw) continue;
        int32_t u=c->faces[cc], w=c->faces[NEXT(cc)];
        if (nrst[u]<0 || nrst[w]<0 || nrst[u]==nrst[w]) continue;
        double wt;
        if (tw<0) wt=EPS_BND;
        else { double dd[3]; vsub(c->verts+3*(size_t)u, c->verts+3*(size_t)w, dd); wt=vlen(dd); }
        cand[ncand].w=dist[u]+wt+dist[w];
        cand[ncand].tu=nrst[u]; cand[ncand].tw=nrst[w];
        cand[ncand].u=u; cand[ncand].vv=w; ncand++;
    }
    qsort(cand,(size_t)ncand,sizeof(SCand),cmp_scand);

    int32_t *uf = (int32_t *)ARENA_ALLOC(arena,(size_t)(nv*sizeof(int32_t)));
    for (size_t v=0;v<nv;v++) uf[v]=(int32_t)v;
    for (long i=0;i<ncand;i++){
        int32_t ru=uf_find2(uf,cand[i].tu), rw=uf_find2(uf,cand[i].tw);
        if (ru==rw) continue;
        uf[ru]=rw;
        int32_t x=cand[i].u;  while(parent[x]>=0){ seam_add(seam,x,parent[x]); x=parent[x]; }
        x=cand[i].vv;         while(parent[x]>=0){ seam_add(seam,x,parent[x]); x=parent[x]; }
        seam_add(seam, cand[i].u, cand[i].vv);
    }
    Arena_restore(arena, m);
}

static int cut_surgery(Arena_T arena, const Conn *c, const SeamSet *seam,
                       SeamCutResult *out)
{
    size_t nc=c->nc, nv=c->nv, nf=c->nf;
    int32_t *sp = (int32_t *)ARENA_ALLOC(arena,(size_t)(nc*sizeof(int32_t)));
    for (size_t i=0;i<nc;i++) sp[i]=(int32_t)i;
    for (size_t cc=0; cc<nc; cc++){
        int32_t tw=c->twin[cc];
        int32_t a,b,fa,fb;
        if (tw<0 || (int32_t)cc>tw) continue;
        a=c->faces[cc]; b=c->faces[NEXT(cc)];
        if (seam_has(seam,a,b)) continue;
        fa=(int32_t)(cc/3)*3;
        fb=(tw/3)*3;
        for (int endpoint=0;endpoint<2;endpoint++) {
            int32_t v=endpoint ? b : a;
            int32_t ca=(c->faces[fa]==v) ? fa :
                       (c->faces[fa+1]==v) ? fa+1 : fa+2;
            int32_t cb=(c->faces[fb]==v) ? fb :
                       (c->faces[fb+1]==v) ? fb+1 : fb+2;
            int32_t ra=ca, rb=cb;
            while(sp[ra]!=ra)ra=sp[ra];
            while(sp[rb]!=rb)rb=sp[rb];
            if(ra!=rb) sp[ra]=rb;
        }
    }
    int32_t *newid = (int32_t *)ARENA_ALLOC(arena,(size_t)(nc*sizeof(int32_t)));
    for (size_t i=0;i<nc;i++) newid[i]=-1;
    int32_t nvnew=0;
    for (size_t cc=0; cc<nc; cc++){ int32_t r=cc; while(sp[r]!=r)r=sp[r];
        if(newid[r]<0) newid[r]=nvnew++; }
    float *V = (float *)ARENA_ALLOC(arena,(size_t)((size_t)nvnew*3*sizeof(float)));
    int32_t *vmap = (int32_t *)ARENA_ALLOC(arena,(size_t)((size_t)nvnew*sizeof(int32_t)));
    int32_t *F = (int32_t *)ARENA_ALLOC(arena,(size_t)(nf*3*sizeof(int32_t)));
    for (size_t cc=0; cc<nc; cc++){
        int32_t r=cc; while(sp[r]!=r)r=sp[r];
        int32_t id=newid[r];
        int32_t v=c->faces[cc];
        V[id*3+0]=c->verts[3*(size_t)v+0]; V[id*3+1]=c->verts[3*(size_t)v+1]; V[id*3+2]=c->verts[3*(size_t)v+2];
        vmap[id]=v;
        F[cc]=id;
    }
    (void)nv;
    out->verts=V; out->faces=F; out->nv=(size_t)nvnew; out->nf=nf; out->vmap=vmap;
    return 0;
}

static int fan_ray_to_boundary(const Conn *c, SeamSet *seam, int32_t cone,
                               int32_t start, uint8_t *network, Heap *heap,
                               double *dist, int32_t *parent,
                               uint32_t *stamp, uint32_t serial,
                               const uint8_t *blocked,SeamCutDiagnostic *diagnostic)
{
    if (blocked && blocked[start]) {
        if (diagnostic) {diagnostic->reason=SEAM_CUT_RAY_BLOCKED;diagnostic->terminal=cone;diagnostic->ray=start;}
        return -1;
    }
    if (!network[start]) {
        int32_t target=-1;
        heap->n=0;stamp[start]=serial;dist[start]=0;parent[start]=-1;
        heap_push(heap,0,start);
        while (heap->n) {
            HItem it=heap_pop(heap);int32_t a=it.v;
            if (it.d>dist[a]) continue;
            if (network[a]) { target=a;break; }
            for (int32_t p=c->voff[a];p<c->voff[a+1];p++) {
                int32_t cc=c->vcor[p];
                int32_t neighbors[2]={c->faces[NEXT(cc)],c->faces[PREV(cc)]};
                for (int q=0;q<2;q++) {
                    int32_t b=neighbors[q];double edge[3];
                    if (b==cone || (blocked && blocked[b])) continue;
                    vsub(c->verts+3*(size_t)a,c->verts+3*(size_t)b,edge);
                    double next=it.d+vlen(edge);
                    if (stamp[b]!=serial || next<dist[b]) {
                        stamp[b]=serial;dist[b]=next;parent[b]=a;
                        heap_push(heap,next,b);
                    }
                }
            }
        }
        if (target<0) {
            if (diagnostic) {diagnostic->reason=SEAM_CUT_RAY_UNREACHABLE;diagnostic->terminal=cone;diagnostic->ray=start;}
            return -1;
        }
        for (int32_t a=target;parent[a]>=0;a=parent[a]) {
            seam_add(seam,a,parent[a]);network[a]=network[parent[a]]=1;
        }
    }
    seam_add(seam,cone,start);network[cone]=network[start]=1;
    return 0;
}

static int32_t overfull_fan_ray_checked(Arena_T arena, const Conn *c,
                                        const SeamSet *seam, int32_t vertex,
                                        SeamCutFanCheck fan_check, void *context)
{
    Arena_Mark mark=Arena_save(arena);
    int32_t begin=c->voff[vertex],degree=c->voff[vertex+1]-begin,result=-1;
    int32_t *neighbor=ARENA_ALLOC(arena,2*(size_t)degree*sizeof *neighbor);
    int32_t *ray=ARENA_ALLOC(arena,2*(size_t)degree*sizeof *ray);
    int32_t *queue=ARENA_ALLOC(arena,(size_t)degree*sizeof *queue);
    int32_t *corners=fan_check?ARENA_ALLOC(arena,(size_t)degree*sizeof *corners):NULL;
    int32_t *rim=fan_check?ARENA_ALLOC(arena,((size_t)degree+1)*sizeof *rim):NULL;
    double *angle=ARENA_ALLOC(arena,(size_t)degree*sizeof *angle);
    uint8_t *seen=ARENA_CALLOC(arena,(size_t)degree,1);
    for (int32_t p=0;p<degree;p++) {
        int32_t cc=c->vcor[begin+p],edges[2]={cc,PREV(cc)};
        angle[p]=corner_angle(c->verts+3*(size_t)vertex,
                             c->verts+3*(size_t)c->faces[NEXT(cc)],
                             c->verts+3*(size_t)c->faces[PREV(cc)]);
        for (int q=0;q<2;q++) {
            int32_t edge=edges[q],other=c->faces[q?PREV(cc):NEXT(cc)];
            int32_t twin=c->twin[edge];
            ray[2*p+q]=other;neighbor[2*p+q]=-1;
            if (twin<0 || seam_has(seam,vertex,other)) continue;
            for (int32_t k=0;k<degree;k++) if (c->vcor[begin+k]/3==twin/3) {
                neighbor[2*p+q]=k;break;
            }
        }
    }
    for (int32_t seed=0;seed<degree && result<0;seed++) if (!seen[seed]) {
        int32_t head=0,tail=0,start=-1;
        double total=0;
        seen[seed]=1;queue[tail++]=seed;
        while (head<tail) {
            int32_t p=queue[head++];total+=angle[p];
            if (neighbor[2*p]<0 || neighbor[2*p+1]<0) start=p;
            for (int q=0;q<2;q++) {
                int32_t next=neighbor[2*p+q];
                if (next>=0 && !seen[next]) { seen[next]=1;queue[tail++]=next; }
            }
        }
        if (!(total>2*M_PI+1e-12)) continue;
        if (start<0) {
            result=ray[2*seed];break;
        }
        if (fan_check) {
            size_t count=0;
            rim[0]=ray[2*start+(neighbor[2*start]<0?0:1)];
            for (int32_t p=start;p>=0;) {
                if (count>=(size_t)tail) { result=-2;goto done; }
                int incoming=ray[2*p]==rim[count]?0:1;
                if (ray[2*p+incoming]!=rim[count]) { result=-2;goto done; }
                corners[count]=c->vcor[begin+p];
                rim[++count]=ray[2*p+1-incoming];
                p=neighbor[2*p+1-incoming];
            }
            if (count!=(size_t)tail) { result=-2;goto done; }
            int feasible=fan_check(context,vertex,corners,rim,count);
            if (feasible<0) { result=-2;goto done; }
            if (feasible) continue;
        }
        double prefix=0,best=INFINITY;
        int32_t previous=-1;
        for (int32_t p=start;p>=0;) {
            int32_t next=-1,edge_vertex=-1;
            prefix+=angle[p];
            for (int q=0;q<2;q++) if (neighbor[2*p+q]>=0 && neighbor[2*p+q]!=previous) {
                next=neighbor[2*p+q];edge_vertex=ray[2*p+q];break;
            }
            if (next>=0 && fabs(prefix-.5*total)<best) {
                best=fabs(prefix-.5*total);result=edge_vertex;
            }
            previous=p;p=next;
        }
    }
done:
    Arena_restore(arena,mark);
    return result;
}

static int32_t overfull_fan_ray(Arena_T arena, const Conn *c,
                                const SeamSet *seam, int32_t vertex)
{ return overfull_fan_ray_checked(arena,c,seam,vertex,NULL,NULL); }

const char *SeamCutReason_name(SeamCutReason reason)
{
    switch (reason) {
    case SEAM_CUT_NOT_RUN:return "not_run";
    case SEAM_CUT_COMPLETE:return "complete";
    case SEAM_CUT_INVALID_INPUT:return "invalid_input";
    case SEAM_CUT_NONMANIFOLD:return "nonmanifold_source";
    case SEAM_CUT_RESERVED_TERMINAL:return "reserved_terminal";
    case SEAM_CUT_INELIGIBLE_TERMINAL:return "ineligible_terminal";
    case SEAM_CUT_UNREACHABLE_TERMINAL:return "unreachable_terminal";
    case SEAM_CUT_FAN_CHECK_FAILED:return "overfull_fan_check_failed";
    case SEAM_CUT_RAY_ALREADY_CUT:return "overfull_ray_already_cut";
    case SEAM_CUT_RAY_BLOCKED:return "overfull_ray_blocked";
    case SEAM_CUT_RAY_UNREACHABLE:return "overfull_ray_unreachable";
    case SEAM_CUT_OVERFULL_UNRESOLVED:return "overfull_fan_unresolved";
    case SEAM_CUT_SURGERY_FAILED:return "surgery_failed";
    default:return "unknown";
    }
}

static int seam_cut_diagnostic_fail(SeamCutDiagnostic *d,SeamCutReason reason,
                                     int32_t terminal,int32_t ray)
{
    if (d) {d->reason=reason;d->terminal=terminal;d->ray=ray;}
    return -1;
}

static int seam_cut_boundary_impl(Arena_T arena, const float *verts, size_t nv,
                                 const int32_t *faces, size_t nf,
                                 const uint8_t *terminals,
                                 SeamCutFanCheck fan_check, void *context,
                                 const uint8_t *eligible,
                                 const uint8_t *reserved,
                                 SeamCutResult *out,SeamCutDiagnostic *diagnostic)
{
    if (diagnostic) {
        memset(diagnostic,0,sizeof *diagnostic);
        diagnostic->first_reserved_terminal=diagnostic->first_unreachable_terminal=-1;
        diagnostic->terminal=diagnostic->ray=-1;
    }
    if (!arena || !verts || !faces || !terminals || !out || !nv || !nf ||
        nv>INT32_MAX || nf>(size_t)INT32_MAX/3)
        return seam_cut_diagnostic_fail(diagnostic,SEAM_CUT_INVALID_INPUT,-1,-1);
    memset(out,0,sizeof *out);
    if (diagnostic) {
        int valid=1;
        for (size_t v=0;v<nv;v++) {
            valid&=terminals[v]<=2&&(!eligible||eligible[v]<=1)&&(!reserved||reserved[v]<=1);
            diagnostic->requested_terminals+=terminals[v]!=0;
            diagnostic->overfull_terminals+=terminals[v]>=2;
            diagnostic->reserved_vertices+=reserved&&reserved[v];
            if (terminals[v]&&reserved&&reserved[v]) {
                diagnostic->reserved_terminals++;
                if (diagnostic->first_reserved_terminal<0) diagnostic->first_reserved_terminal=(int32_t)v;
            }
        }
        diagnostic->input_counts_complete=valid;
    }
    Conn c; conn_build(arena,verts,nv,faces,nf,&c);
    if (c.nonmanifold) return seam_cut_diagnostic_fail(diagnostic,SEAM_CUT_NONMANIFOLD,-1,-1);
    double *dist=ARENA_ALLOC(arena,nv*sizeof *dist);
    int32_t *parent=ARENA_ALLOC(arena,nv*sizeof *parent);
    uint8_t *boundary=ARENA_CALLOC(arena,nv,1);
    uint8_t *network=ARENA_CALLOC(arena,nv,1);
    uint8_t *blocked=(eligible||reserved)?ARENA_CALLOC(arena,nv,1):NULL;
    Heap heap={0}; heap.cap=c.nc*4+nv+1024;
    heap.a=ARENA_ALLOC(arena,heap.cap*sizeof *heap.a);
    for (size_t v=0;v<nv;v++) { dist[v]=INFINITY; parent[v]=-1; }
    for (size_t cc=0;cc<c.nc;cc++) if (c.twin[cc]<0) {
        boundary[faces[cc]]=1; boundary[faces[NEXT(cc)]]=1;
    }
    if (diagnostic) {
        int valid=diagnostic->input_counts_complete;
        for (size_t v=0;v<nv;v++) {
            valid&=!eligible||!eligible[v]||boundary[v];
            diagnostic->interior_terminals+=terminals[v]&&!boundary[v];
            if (boundary[v]) {
                int excluded=(eligible&&!eligible[v])||(reserved&&reserved[v]);
                diagnostic->boundary_vertices++;
                diagnostic->eligible_boundary_vertices+=!excluded;
                diagnostic->blocked_boundary_vertices+=excluded;
            }
        }
        diagnostic->boundary_counts_complete=valid;
    }
    for (size_t v=0;v<nv;v++) {
        if (terminals[v]>2 || (eligible && (eligible[v]>1 ||
            (eligible[v]&&!boundary[v]))) || (reserved && reserved[v]>1))
            return seam_cut_diagnostic_fail(diagnostic,SEAM_CUT_INVALID_INPUT,(int32_t)v,-1);
        if (blocked) blocked[v]=(uint8_t)((eligible&&boundary[v]&&!eligible[v]) ||
                                          (reserved&&reserved[v]));
        if (terminals[v] && blocked && blocked[v])
            return seam_cut_diagnostic_fail(diagnostic,reserved&&reserved[v]?SEAM_CUT_RESERVED_TERMINAL:SEAM_CUT_INELIGIBLE_TERMINAL,(int32_t)v,-1);
        if (boundary[v]&&(!blocked||!blocked[v])) { dist[v]=0; heap_push(&heap,0,(int32_t)v); }
    }
    while (heap.n) {
        HItem it=heap_pop(&heap); int32_t a=it.v;
        if (it.d>dist[a]) continue;
        for (int32_t k=c.voff[a];k<c.voff[a+1];k++) {
            int32_t cc=c.vcor[k], nb[2]={faces[NEXT(cc)],faces[PREV(cc)]};
            for (int q=0;q<2;q++) {
                int32_t b=nb[q]; double edge[3];
                if (blocked && blocked[b]) continue;
                vsub(verts+3*(size_t)a,verts+3*(size_t)b,edge);
                double next=it.d+vlen(edge);
                if (next<dist[b]) { dist[b]=next; parent[b]=a; heap_push(&heap,next,b); }
            }
        }
    }
    if (diagnostic) {
        for (size_t v=0;v<nv;v++) if (terminals[v]&&!boundary[v]&&!isfinite(dist[v])) {
            diagnostic->unreachable_terminals++;
            if (diagnostic->first_unreachable_terminal<0) diagnostic->first_unreachable_terminal=(int32_t)v;
        }
        diagnostic->reachability_complete=1;
    }
    SeamSet seam; seam_init(arena,&seam,nv,c.nc);
    for (size_t v=0;v<nv;v++) network[v]=(uint8_t)(boundary[v]&&(!blocked||!blocked[v]));
    for (size_t v=0;v<nv;v++) if (terminals[v] && !boundary[v]) {
        if (!isfinite(dist[v])) {
            if (diagnostic) diagnostic->seam_edges=seam.n;
            return seam_cut_diagnostic_fail(diagnostic,SEAM_CUT_UNREACHABLE_TERMINAL,(int32_t)v,-1);
        }
        out->n_terminals++;
        for (int32_t a=(int32_t)v;parent[a]>=0;a=parent[a]) {
            seam_add(&seam,a,parent[a]);network[a]=network[parent[a]]=1;
        }
    }
    uint32_t *stamp=ARENA_CALLOC(arena,nv,sizeof *stamp),serial=0;
    size_t extra_rays=0;
    for (size_t v=0;v<nv;v++) if (terminals[v]>=2) {
        size_t degree=(size_t)(c.voff[v+1]-c.voff[v]);
        int resolved=0;
        for (size_t attempt=0;attempt<degree;attempt++) {
            int32_t ray=overfull_fan_ray_checked(arena,&c,&seam,(int32_t)v,fan_check,context);
            if (diagnostic) diagnostic->seam_edges=seam.n;
            if (ray==-2) return seam_cut_diagnostic_fail(diagnostic,SEAM_CUT_FAN_CHECK_FAILED,(int32_t)v,ray);
            if (ray<0) { resolved=1;break; }
            if (seam_has(&seam,(int32_t)v,ray))
                return seam_cut_diagnostic_fail(diagnostic,SEAM_CUT_RAY_ALREADY_CUT,(int32_t)v,ray);
            if (fan_ray_to_boundary(&c,&seam,(int32_t)v,ray,network,&heap,
                                    dist,parent,stamp,++serial,blocked,diagnostic)) return -1;
            extra_rays++;
            if (diagnostic) {diagnostic->additional_rays=extra_rays;diagnostic->seam_edges=seam.n;}
        }
        if (!resolved && overfull_fan_ray_checked(arena,&c,&seam,(int32_t)v,fan_check,context)!=-1)
            return seam_cut_diagnostic_fail(diagnostic,SEAM_CUT_OVERFULL_UNRESOLVED,(int32_t)v,-1);
    }
    if (extra_rays) fprintf(stderr,"  [seamcut] %zu additional negative-cone fan branches\n",extra_rays);
    out->seam_edges=(long)seam.n;
    if (diagnostic) {diagnostic->seam_edges=seam.n;diagnostic->routes_complete=1;}
    int result=cut_surgery(arena,&c,&seam,out);
    if (diagnostic) {diagnostic->surgery_complete=result==0;diagnostic->reason=result?SEAM_CUT_SURGERY_FAILED:SEAM_CUT_COMPLETE;}
    return result;
}

int SeamCut_to_boundary_checked(Arena_T arena,const float *verts,size_t nv,
    const int32_t *faces,size_t nf,const uint8_t *terminals,
    SeamCutFanCheck fan_check,void *context,SeamCutResult *out)
{ return seam_cut_boundary_impl(arena,verts,nv,faces,nf,terminals,fan_check,context,NULL,NULL,out,NULL); }

int SeamCut_to_boundary_eligible(Arena_T arena,const float *verts,size_t nv,
    const int32_t *faces,size_t nf,const uint8_t *terminals,
    const uint8_t *eligible,const uint8_t *blocked,SeamCutResult *out)
{ return seam_cut_boundary_impl(arena,verts,nv,faces,nf,terminals,NULL,NULL,eligible,blocked,out,NULL); }

int SeamCut_to_boundary_eligible_report(Arena_T arena,const float *verts,size_t nv,
    const int32_t *faces,size_t nf,const uint8_t *terminals,
    const uint8_t *eligible,const uint8_t *blocked,SeamCutResult *out,
    SeamCutDiagnostic *diagnostic)
{ return seam_cut_boundary_impl(arena,verts,nv,faces,nf,terminals,NULL,NULL,eligible,blocked,out,diagnostic); }

int SeamCut_to_boundary(Arena_T arena, const float *verts, size_t nv,
                         const int32_t *faces, size_t nf,
                         const uint8_t *terminals, SeamCutResult *out)
{ return SeamCut_to_boundary_checked(arena,verts,nv,faces,nf,terminals,NULL,NULL,out); }

static int shortest_boundary_join(Arena_T arena, const Conn *c, SeamSet *seam,
                                  long *out_loop_count)
{
    int32_t *boundary_parent, *owner, *path_parent;
    double *dist;
    char *is_boundary, *settled, *root_seen;
    Heap heap;
    long nloops = 0;
    double best = 1e300;
    int32_t best_u = -1, best_v = -1;

    boundary_parent=(int32_t*)ARENA_ALLOC(
        arena,(size_t)(c->nv*sizeof(int32_t)));
    owner=(int32_t*)ARENA_ALLOC(arena,(size_t)(c->nv*sizeof(int32_t)));
    path_parent=(int32_t*)ARENA_ALLOC(
        arena,(size_t)(c->nv*sizeof(int32_t)));
    dist=(double*)ARENA_ALLOC(arena,(size_t)(c->nv*sizeof(double)));
    is_boundary=(char*)ARENA_CALLOC(arena,(size_t)c->nv,1L);
    settled=(char*)ARENA_CALLOC(arena,(size_t)c->nv,1L);
    root_seen=(char*)ARENA_CALLOC(arena,(size_t)c->nv,1L);
    for (size_t v=0;v<c->nv;v++) {
        boundary_parent[v]=(int32_t)v;
        owner[v]=-1; path_parent[v]=-1; dist[v]=1e300;
    }
    for (size_t cc=0;cc<c->nc;cc++) {
        int32_t a,b,ra,rb;
        if (c->twin[cc]>=0) continue;
        a=c->faces[cc]; b=c->faces[NEXT(cc)];
        is_boundary[a]=is_boundary[b]=1;
        ra=uf_find2(boundary_parent,a);
        rb=uf_find2(boundary_parent,b);
        if (ra!=rb) boundary_parent[rb]=ra;
    }
    for (size_t v=0;v<c->nv;v++) {
        int32_t root;
        if (!is_boundary[v]) continue;
        root=uf_find2(boundary_parent,(int32_t)v);
        owner[v]=root; dist[v]=0.0;
        if (!root_seen[root]) { root_seen[root]=1; nloops++; }
    }
    if (out_loop_count) *out_loop_count=nloops;
    if (nloops<2) return nloops==1 ? 1 : -1;

    heap.n=0;
    heap.cap=c->nc*4+c->nv+1024;
    heap.a=(HItem*)ARENA_ALLOC(arena,(size_t)(heap.cap*sizeof(HItem)));
    for (size_t v=0;v<c->nv;v++)
        if (is_boundary[v]) heap_push(&heap,0.0,(int32_t)v);

    while (heap.n>0) {
        HItem item=heap_pop(&heap);
        int32_t u=item.v;
        if (item.d>dist[u]+1e-12 || settled[u]) continue;
        if (item.d>=best) break;
        settled[u]=1;
        for (int32_t e=c->voff[u];e<c->voff[u+1];e++) {
            int32_t cc=c->vcor[e];
            int32_t ws[2]={c->faces[NEXT(cc)],c->faces[PREV(cc)]};
            for (int k=0;k<2;k++) {
                int32_t w=ws[k];
                double delta[3], weight, next_dist;
                vsub(c->verts+3*(size_t)u,
                     c->verts+3*(size_t)w,delta);
                weight=vlen(delta);
                if (settled[w] && owner[w]!=owner[u]) {
                    double candidate=dist[u]+weight+dist[w];
                    if (candidate<best) {
                        best=candidate; best_u=u; best_v=w;
                    }
                }
                if (settled[w]) continue;
                next_dist=dist[u]+weight;
                if (next_dist<dist[w]-1e-12) {
                    dist[w]=next_dist;
                    owner[w]=owner[u];
                    path_parent[w]=u;
                    heap_push(&heap,next_dist,w);
                }
            }
        }
    }
    if (best_u<0 || best_v<0) return -1;
    for (int32_t v=best_u;path_parent[v]>=0;v=path_parent[v])
        seam_add(seam,v,path_parent[v]);
    for (int32_t v=best_v;path_parent[v]>=0;v=path_parent[v])
        seam_add(seam,v,path_parent[v]);
    seam_add(seam,best_u,best_v);
    return seam->n>0 ? 0 : -1;
}

static int boundary_reduce_topology(Arena_T arena,
                                    const float *verts, size_t nv,
                                    const int32_t *faces, size_t nf,
                                    float **out_verts, size_t *out_nv,
                                    int32_t **out_faces, size_t *out_nf,
                                    int32_t **out_vmap,
                                    long *out_initial_loops,
                                    long *out_seam_edges)
{
    const float *cv=verts;
    const int32_t *cf=faces;
    size_t cnv=nv, cnf=nf;
    float *malloc_v=NULL;
    int32_t *malloc_f=NULL, *malloc_map=NULL;
    int32_t *accum=NULL;
    long initial_loops=-1, total_seam=0, joins=0;
    int dbg=(getenv("SEAMCUT_DEBUG")!=NULL);
    Arena_Mark iteration;

    for (;;) {
        iteration=Arena_save(arena);
        MeshTopoInfo before;
        if (MeshTopo_analyze(arena,NULL,cnv,cf,cnf,&before)!=0 ||
            before.n_nonmanifold_edges!=0 || before.n_unref_verts!=0 ||
            before.n_boundary_loops<1 || fabs(before.genus)>1e-6)
            goto fail_iteration;
        if (initial_loops<0) initial_loops=before.n_boundary_loops;
        if (before.n_boundary_loops==1) {
            float *final_v;
            int32_t *final_f, *final_map;
            Arena_restore(arena,iteration);
            final_v=(float*)ARENA_ALLOC(
                arena,(size_t)(cnv*3*sizeof(float)));
            final_f=(int32_t*)ARENA_ALLOC(
                arena,(size_t)(cnf*3*sizeof(int32_t)));
            final_map=(int32_t*)ARENA_ALLOC(
                arena,(size_t)(cnv*sizeof(int32_t)));
            memcpy(final_v,cv,cnv*3*sizeof(float));
            memcpy(final_f,cf,cnf*3*sizeof(int32_t));
            if (accum) memcpy(final_map,accum,cnv*sizeof(int32_t));
            else for (size_t v=0;v<cnv;v++) final_map[v]=(int32_t)v;
            free(malloc_map); free(malloc_f); free(malloc_v);
            *out_verts=final_v; *out_nv=cnv;
            *out_faces=final_f; *out_nf=cnf; *out_vmap=final_map;
            if (out_initial_loops) *out_initial_loops=initial_loops;
            if (out_seam_edges) *out_seam_edges=total_seam;
            return 0;
        }
        if (joins>=initial_loops) goto fail_iteration;

        {
            Conn c;
            SeamSet seam;
            SeamCutResult cut;
            MeshTopoInfo after;
            long found_loops=0;
            int32_t *next_map;
            float *next_v;
            int32_t *next_f, *next_accum;
            int join_rc;

            conn_build(arena,cv,cnv,cf,cnf,&c);
            seam_init(arena,&seam,cnv,cnv+1);
            join_rc=shortest_boundary_join(
                arena,&c,&seam,&found_loops);
            if (join_rc!=0 || found_loops!=before.n_boundary_loops) {
                if (dbg)
                    fprintf(stderr,
                            "  [seamcut] join search failed: rc=%d "
                            "loops found=%ld expected=%ld\n",
                            join_rc,found_loops,before.n_boundary_loops);
                goto fail_iteration;
            }
            memset(&cut,0,sizeof cut);
            if (cut_surgery(arena,&c,&seam,&cut)!=0) {
                if (dbg) fprintf(stderr,
                                 "  [seamcut] join surgery failed\n");
                goto fail_iteration;
            }
            join_rc=MeshTopo_analyze(
                arena,NULL,cut.nv,cut.faces,cut.nf,&after);
            if (join_rc!=0 ||
                after.n_nonmanifold_edges!=0 || after.n_unref_verts!=0 ||
                after.n_boundary_loops!=before.n_boundary_loops-1 ||
                fabs(after.genus)>1e-6) {
                if (dbg)
                    fprintf(stderr,
                            "  [seamcut] join certificate failed: rc=%d "
                            "loops %ld->%ld genus=%.3f nm=%ld unref=%ld "
                            "path=%zu V=%zu->%zu\n",
                            join_rc,before.n_boundary_loops,
                            after.n_boundary_loops,after.genus,
                            after.n_nonmanifold_edges,after.n_unref_verts,
                            seam.n,cnv,cut.nv);
                goto fail_iteration;
            }

            next_map=(int32_t*)ARENA_ALLOC(
                arena,(size_t)(cut.nv*sizeof(int32_t)));
            for (size_t v=0;v<cut.nv;v++)
                next_map[v]=accum ? accum[cut.vmap[v]] : cut.vmap[v];
            next_v=(float*)malloc(cut.nv*3*sizeof(float));
            next_f=(int32_t*)malloc(cut.nf*3*sizeof(int32_t));
            next_accum=(int32_t*)malloc(cut.nv*sizeof(int32_t));
            if (!next_v || !next_f || !next_accum) {
                free(next_accum); free(next_f); free(next_v);
                goto fail_iteration;
            }
            memcpy(next_v,cut.verts,cut.nv*3*sizeof(float));
            memcpy(next_f,cut.faces,cut.nf*3*sizeof(int32_t));
            memcpy(next_accum,next_map,cut.nv*sizeof(int32_t));
            total_seam+=(long)seam.n;
            joins++;
            if (dbg)
                fprintf(stderr,
                        "  [seamcut] boundary join %ld: loops %ld->%ld "
                        "path=%zu edge(s) V=%zu->%zu\n",
                        joins,before.n_boundary_loops,
                        after.n_boundary_loops,seam.n,cnv,cut.nv);
            Arena_restore(arena,iteration);
            free(malloc_map); free(malloc_f); free(malloc_v);
            malloc_v=next_v; malloc_f=next_f; malloc_map=next_accum;
            cv=malloc_v; cf=malloc_f; accum=malloc_map;
            cnv=cut.nv; cnf=cut.nf;
        }
    }

fail_iteration:
    Arena_restore(arena,iteration);
    free(malloc_map); free(malloc_f); free(malloc_v);
    return -1;
}

typedef struct { double d; int32_t i; } SeamDI;
static int cmp_di_desc(const void *a, const void *b)
{
    double x=((const SeamDI*)a)->d, y=((const SeamDI*)b)->d;
    return (x<y)?1:(x>y?-1:0);
}

static void tree_bfs(const Conn *c, const char *tree, int32_t src,
                     double *dist, int32_t *parent, int32_t *depth, int32_t *queue)
{
    for (size_t v=0; v<c->nv; v++){ dist[v]=1e300; parent[v]=-1; depth[v]=-1; }
    size_t h=0, t=0; dist[src]=0.0; depth[src]=0; queue[t++]=src;
    while (h<t){
        int32_t u=queue[h++];
        for (int32_t e=c->voff[u]; e<c->voff[u+1]; e++){
            int32_t cc=c->vcor[e];
            int32_t ws[2]={c->faces[NEXT(cc)], c->faces[PREV(cc)]};
            int32_t es[2]={cc, PREV(cc)};
            for (int k=0;k<2;k++){
                if (!tree[es[k]]) continue;
                int32_t w=ws[k];
                if (depth[w]!=-1) continue;
                double dd[3]; vsub(c->verts+3*(size_t)u, c->verts+3*(size_t)w, dd);
                parent[w]=u; depth[w]=depth[u]+1; dist[w]=dist[u]+vlen(dd); queue[t++]=w;
            }
        }
    }
}

static void extract_cycle(const int32_t *parent, const int32_t *depth,
                          int32_t a, int32_t b, SeamSet *eg)
{
    seam_add(eg, a, b);
    int32_t x=a, y=b;
    while (depth[x]>depth[y]){ seam_add(eg,x,parent[x]); x=parent[x]; }
    while (depth[y]>depth[x]){ seam_add(eg,y,parent[y]); y=parent[y]; }
    while (x!=y){
        seam_add(eg,x,parent[x]); x=parent[x];
        seam_add(eg,y,parent[y]); y=parent[y];
    }
}

static int is_nonseparating(Arena_T arena, const Conn *c, const SeamSet *eg)
{
    if (c->nf == 0) return 0;
    Arena_Mark m=Arena_save(arena);
    int32_t *uf=(int32_t*)ARENA_ALLOC(arena,(size_t)(c->nf*sizeof(int32_t)));
    for (size_t f=0; f<c->nf; f++) uf[f]=(int32_t)f;
    for (size_t cc=0; cc<c->nc; cc++){
        int32_t tw=c->twin[cc];
        if (tw<0 || (int32_t)cc>tw) continue;
        int32_t a=c->faces[cc], b=c->faces[NEXT(cc)];
        if (seam_has(eg,a,b)) continue;
        int32_t ra=uf_find2(uf,(int32_t)(cc/3)), rb=uf_find2(uf,tw/3);
        if (ra!=rb) uf[ra]=rb;
    }
    int32_t r0=uf_find2(uf,0); int conn=1;
    for (size_t f=1; f<c->nf; f++) if (uf_find2(uf,(int32_t)f)!=r0){ conn=0; break; }
    Arena_restore(arena,m);
    return conn;
}

typedef struct { double w; int32_t a, b; } LCand;
static int cmp_lcand(const void *p, const void *q)
{ double x=((const LCand*)p)->w, y=((const LCand*)q)->w; return (x<y)?-1:(x>y?1:0); }

static double seam_loop_length(const SeamSet *eg, size_t cnv, const float *cv)
{
    double len=0.0;
    for (size_t si=0; si<eg->cap; si++){
        uint64_t k=eg->slot[si]; if (k==UINT64_MAX) continue;
        int32_t lo=(int32_t)(k/(uint64_t)cnv), hi=(int32_t)(k%(uint64_t)cnv);
        double dd[3]; vsub(cv+3*(size_t)lo, cv+3*(size_t)hi, dd); len+=vlen(dd);
    }
    return len;
}

static void keep_largest_edge(Arena_T arena, const float *v, size_t nv,
                              const int32_t *f, size_t nf,
                              float **ov, size_t *onv, int32_t **of, size_t *onf,
                              int32_t **ovmap)
{
    size_t nc = 3*nf;
    EKey *ek = (EKey *)ARENA_ALLOC(arena, (size_t)(nc*sizeof(EKey)));
    for (size_t cc=0; cc<nc; cc++){
        int32_t a=f[cc], b=f[NEXT(cc)];
        uint64_t lo=(uint64_t)(a<b?a:b), hi=(uint64_t)(a<b?b:a);
        ek[cc].key=lo*(uint64_t)nv+hi; ek[cc].corner=(int32_t)cc;
    }
    qsort(ek, nc, sizeof(EKey), cmp_ekey);
    int32_t *uf=(int32_t*)ARENA_ALLOC(arena,(size_t)(nf*sizeof(int32_t)));
    for (size_t i=0;i<nf;i++) uf[i]=(int32_t)i;
    size_t i=0;
    while (i<nc){ size_t j=i; while(j<nc && ek[j].key==ek[i].key) j++;
        for (size_t p=i+1;p<j;p++){ int32_t ra=uf_find2(uf,ek[i].corner/3), rb=uf_find2(uf,ek[p].corner/3); if(ra!=rb) uf[ra]=rb; }
        i=j; }
    long *cnt=(long*)ARENA_CALLOC(arena,(size_t)nf,sizeof(long));
    for (size_t t=0;t<nf;t++) cnt[uf_find2(uf,(int32_t)t)]++;
    int32_t best=-1; long bestn=-1;
    for (size_t r=0;r<nf;r++) if(cnt[r]>bestn){bestn=cnt[r];best=(int32_t)r;}

    int32_t *vm=(int32_t*)ARENA_ALLOC(arena,(size_t)(nv*sizeof(int32_t)));
    int32_t *rm=(int32_t*)ARENA_ALLOC(arena,(size_t)(nv*sizeof(int32_t)));
    for (size_t k=0;k<nv;k++) vm[k]=-1;
    int32_t *F=(int32_t*)ARENA_ALLOC(arena,bestn*3*sizeof(int32_t));
    long onv2=0, fi=0;
    for (size_t t=0;t<nf;t++){
        if (uf_find2(uf,(int32_t)t)!=best) continue;
        for (int k=0;k<3;k++){ int32_t vg=f[t*3+k];
            if(vm[vg]==-1){vm[vg]=(int32_t)onv2; rm[onv2]=vg; onv2++;}
            F[fi*3+k]=vm[vg]; }
        fi++;
    }
    float *V=(float*)ARENA_ALLOC(arena,onv2*3*sizeof(float));
    int32_t *vmap=(int32_t*)ARENA_ALLOC(arena,onv2*sizeof(int32_t));
    for (long k=0;k<onv2;k++){ V[k*3+0]=v[rm[k]*3+0]; V[k*3+1]=v[rm[k]*3+1]; V[k*3+2]=v[rm[k]*3+2]; vmap[k]=rm[k]; }
    *ov=V; *onv=(size_t)onv2; *of=F; *onf=(size_t)fi; *ovmap=vmap;
}

static void genus_reduce(Arena_T arena, const float **pv, size_t *pnv,
                         const int32_t **pf, size_t *pnf,
                         int32_t **paccum, long genus0, long *ploops_cut,
                         HandleLoop *collect, size_t *ncol, double max_loop_len)
{
    const float   *cv=*pv; size_t cnv=*pnv;
    const int32_t *cf=*pf; size_t cnf=*pnf;
    int32_t *accum=NULL;
    long cut=0, cap=genus0*4+16;

    const int reclaim = (collect == NULL);
    float *mcv=NULL; int32_t *mcf=NULL; int32_t *macc=NULL;

    int dbg = (getenv("SEAMCUT_DEBUG") != NULL);
    for (;;){
        Arena_Mark it = Arena_save(arena);
        double t_it = ves_clock_sec();
        MeshTopoInfo ti; MeshTopo_analyze(arena, NULL, cnv, cf, cnf, &ti);
        long g=lround(ti.genus);
        if (g<=0){ if (reclaim) Arena_restore(arena,it); break; }
        if (cut>=cap){ fprintf(stderr,"  [seamcut] genus-reduce cap %ld hit (g=%ld left)\n",cap,g);
                       if (reclaim) Arena_restore(arena,it); break; }

        Conn c; conn_build(arena, cv, cnv, cf, cnf, &c);

        char *cotree=(char*)ARENA_CALLOC(arena,(size_t)c.nc,1L);
        size_t dual_faces=0;
        {
            char    *fseen=(char*)ARENA_CALLOC(arena,(size_t)c.nf,1L);
            int32_t *fq=(int32_t*)ARENA_ALLOC(arena,(size_t)(c.nf*sizeof(int32_t)));
            size_t qh=0,qt=0; fseen[0]=1; fq[qt++]=0;
            while (qh<qt){
                int32_t f=fq[qh++];
                for (int k=0;k<3;k++){
                    int32_t cc=f*3+k, tw=c.twin[cc];
                    if (tw<0) continue;
                    int32_t nf2=tw/3;
                    if (fseen[nf2]) continue;
                    fseen[nf2]=1; cotree[cc]=1; cotree[tw]=1; fq[qt++]=nf2;
                }
            }
            dual_faces=qt;
        }
        char *tree=(char*)ARENA_CALLOC(arena,(size_t)c.nc,1L);
        {
            int32_t *uf=(int32_t*)ARENA_ALLOC(arena,(size_t)(cnv*sizeof(int32_t)));
            for (size_t v=0;v<cnv;v++) uf[v]=(int32_t)v;
            for (size_t cc=0; cc<c.nc; cc++){
                if (c.twin[cc]>=0) continue;
                int32_t ra=uf_find2(uf,c.faces[cc]), rb=uf_find2(uf,c.faces[NEXT(cc)]);
                if (ra!=rb){ uf[ra]=rb; tree[cc]=1; }
            }
            for (size_t cc=0; cc<c.nc; cc++){
                int32_t tw=c.twin[cc];
                if (tw<0 || (int32_t)cc>tw) continue;
                if (cotree[cc]) continue;
                int32_t ra=uf_find2(uf,c.faces[cc]), rb=uf_find2(uf,c.faces[NEXT(cc)]);
                if (ra!=rb){ uf[ra]=rb; tree[cc]=1; tree[tw]=1; }
            }
            for (size_t cc=0; cc<c.nc; cc++){
                int32_t tw=c.twin[cc];
                if (tw<0 || (int32_t)cc>tw) continue;
                if (!cotree[cc]) continue;
                int32_t ra=uf_find2(uf,c.faces[cc]), rb=uf_find2(uf,c.faces[NEXT(cc)]);
                if (ra!=rb){ uf[ra]=rb; tree[cc]=1; tree[tw]=1; cotree[cc]=0; cotree[tw]=0; }
            }
        }
        double  *dist=(double *)ARENA_ALLOC(arena,(size_t)(cnv*sizeof(double)));
        int32_t *par =(int32_t*)ARENA_ALLOC(arena,(size_t)(cnv*sizeof(int32_t)));
        int32_t *dep =(int32_t*)ARENA_ALLOC(arena,(size_t)(cnv*sizeof(int32_t)));
        int32_t *bq =(int32_t*)ARENA_ALLOC(arena,(size_t)(cnv*sizeof(int32_t)));
        tree_bfs(&c, tree, 0, dist, par, dep, bq);
        LCand *cand=(LCand*)ARENA_ALLOC(arena,(size_t)(c.nc*sizeof(LCand)));
        long ncd=0;
        for (size_t cc=0; cc<c.nc; cc++){
            int32_t tw=c.twin[cc];
            if (tw<0 || (int32_t)cc>tw) continue;
            if (tree[cc] || cotree[cc]) continue;
            int32_t a=c.faces[cc], b=c.faces[NEXT(cc)];
            if (dist[a]>=1e299 || dist[b]>=1e299) continue;
            double dd[3]; vsub(cv+3*(size_t)a, cv+3*(size_t)b, dd);
            cand[ncd].w=dist[a]+vlen(dd)+dist[b]; cand[ncd].a=a; cand[ncd].b=b; ncd++;
        }
        qsort(cand,(size_t)ncd,sizeof(LCand),cmp_lcand);

        int found=0; long tested=0, skipped_long=0;
        for (long i=0;i<ncd;i++){
            Arena_Mark mk=Arena_save(arena);
            SeamSet eg; seam_init(arena,&eg,cnv,2*cnv);
            extract_cycle(par, dep, cand[i].a, cand[i].b, &eg);
            tested++;
            if (is_nonseparating(arena,&c,&eg)){
                double loop_len = seam_loop_length(&eg, cnv, cv);
                if (max_loop_len > 0.0 && loop_len > max_loop_len){
                    skipped_long++;
                    Arena_restore(arena,mk);
                    continue;
                }
                if (collect){
                    size_t ne=0;
                    for (size_t si=0; si<eg.cap; si++) if (eg.slot[si]!=UINT64_MAX) ne++;
                    int32_t *lv=(int32_t*)ARENA_ALLOC(arena,(size_t)(2*ne*sizeof(int32_t)));
                    size_t kk=0; double len=0.0;
                    for (size_t si=0; si<eg.cap; si++){
                        uint64_t key=eg.slot[si]; if (key==UINT64_MAX) continue;
                        int32_t lo=(int32_t)(key/(uint64_t)cnv), hi=(int32_t)(key%(uint64_t)cnv);
                        double dd[3]; vsub(cv+3*(size_t)lo, cv+3*(size_t)hi, dd); len+=vlen(dd);
                        lv[kk++]= accum?accum[lo]:lo;
                        lv[kk++]= accum?accum[hi]:hi;
                    }
                    collect[*ncol].verts=lv; collect[*ncol].n=kk; collect[*ncol].length=len;
                    (*ncol)++;
                }
                SeamCutResult tmp; memset(&tmp,0,sizeof tmp);
                cut_surgery(arena,&c,&eg,&tmp);
                int32_t *na=(int32_t*)ARENA_ALLOC(arena,(size_t)(tmp.nv*sizeof(int32_t)));
                for (size_t k=0;k<tmp.nv;k++) na[k]=accum?accum[tmp.vmap[k]]:tmp.vmap[k];
                cv=tmp.verts; cnv=tmp.nv; cf=tmp.faces; cnf=tmp.nf; accum=na;
                {
                    float *lv=NULL; int32_t *lf=NULL; size_t lnv=0,lnf=0; int32_t *lvm=NULL;
                    keep_largest_edge(arena, cv,cnv,cf,cnf, &lv,&lnv,&lf,&lnf,&lvm);
                    if (lnv>0 && lnv<cnv){
                        int32_t *na2=(int32_t*)ARENA_ALLOC(arena,(size_t)(lnv*sizeof(int32_t)));
                        for (size_t k=0;k<lnv;k++) na2[k]=accum[lvm[k]];
                        cv=lv; cnv=lnv; cf=lf; cnf=lnf; accum=na2;
                    }
                }
                cut++; found=1;
                break;
            }
            Arena_restore(arena,mk);
        }
        if (dbg) fprintf(stderr,"  [seamcut] g=%ld B=%ld nm=%ld V=%zu gens=%ld dualF=%zu/%zu tested=%ld -> %s (%.2fs)\n",
                         g, ti.n_boundary_loops, ti.n_nonmanifold_edges, cnv, ncd,
                         dual_faces, c.nf, tested, found?"cut":"none", ves_clock_sec()-t_it);
        if (!found){
            if (max_loop_len > 0.0 && skipped_long > 0){ if (reclaim) Arena_restore(arena,it); break; }
            if (dbg){
                long ne_int=0, nt=0, nco=0, nboth=0;
                for (size_t cc=0; cc<c.nc; cc++){
                    int32_t tw=c.twin[cc];
                    if (tw<0 || (int32_t)cc>tw) continue;
                    ne_int++; if(tree[cc])nt++; if(cotree[cc])nco++; if(tree[cc]&&cotree[cc])nboth++;
                }
                fprintf(stderr,"  [seamcut] STALL g=%ld E_int=%ld tree=%ld cotree=%ld both=%ld unref=%ld V=%zu F=%zu\n",
                        g, ne_int, nt, nco, nboth, ti.n_unref_verts, cnv, c.nf);
            }
            fprintf(stderr,"  [seamcut] no genus-reducing loop found (g=%ld)\n",g);
            if (reclaim) Arena_restore(arena,it);
            break;
        }

        if (reclaim){
            size_t vn=cnv, fn=cnf;
            float   *nv2=(float  *)malloc(vn*3*sizeof(float));
            int32_t *nf2=(int32_t*)malloc(fn*3*sizeof(int32_t));
            int32_t *na2=(int32_t*)malloc(vn*sizeof(int32_t));
            if (!nv2 || !nf2 || !na2){ free(nv2); free(nf2); free(na2);
                fprintf(stderr,"  [seamcut] genus-reduce OOM (malloc)\n");
                Arena_restore(arena,it); break; }
            memcpy(nv2, cv,    vn*3*sizeof(float));
            memcpy(nf2, cf,    fn*3*sizeof(int32_t));
            memcpy(na2, accum, vn*sizeof(int32_t));
            Arena_restore(arena, it);
            free(mcv); free(mcf); free(macc);
            mcv=nv2; mcf=nf2; macc=na2;
            cv=mcv; cf=mcf; accum=macc; cnv=vn; cnf=fn;
        }
    }

    if (mcv){
        float   *ov=(float  *)ARENA_ALLOC(arena,(size_t)(cnv*3*sizeof(float)));
        int32_t *of=(int32_t*)ARENA_ALLOC(arena,(size_t)(cnf*3*sizeof(int32_t)));
        int32_t *oa=(int32_t*)ARENA_ALLOC(arena,(size_t)(cnv*sizeof(int32_t)));
        memcpy(ov, mcv,  cnv*3*sizeof(float));
        memcpy(of, mcf,  cnf*3*sizeof(int32_t));
        memcpy(oa, macc, cnv*sizeof(int32_t));
        free(mcv); free(mcf); free(macc);
        cv=ov; cf=of; accum=oa;
    }
    *pv=cv; *pnv=cnv; *pf=cf; *pnf=cnf; *paccum=accum; *ploops_cut=cut;
}

int SeamCut_handle_loops(Arena_T arena, const float *verts, size_t nv,
                         const int32_t *faces, size_t nf,
                         HandleLoop **out_loops, size_t *out_n)
{
    assert(arena && out_loops && out_n);
    *out_loops = NULL; *out_n = 0;
    if (nv < 3 || nf < 1 || !verts || !faces) return -1;

    MeshTopoInfo ti; MeshTopo_analyze(arena, NULL, nv, faces, nf, &ti);
    long g0 = lround(ti.genus);
    if (g0 <= 0) return 0;

    HandleLoop *loops = (HandleLoop *)ARENA_CALLOC(arena, (size_t)g0,
                                                   sizeof(HandleLoop));
    size_t ncol = 0;
    const float   *cv = verts; size_t cnv = nv;
    const int32_t *cf = faces; size_t cnf = nf;
    int32_t *accum = NULL; long loops_cut = 0;
    genus_reduce(arena, &cv, &cnv, &cf, &cnf, &accum, g0, &loops_cut, loops, &ncol, 0.0);

    *out_loops = loops;
    *out_n = ncol;
    return 0;
}

int SeamCut_sever_short_handles(Arena_T arena, const float *verts, size_t nv,
                                const int32_t *faces, size_t nf, double max_loop_len,
                                float **out_verts, size_t *out_nv,
                                int32_t **out_faces, size_t *out_nf,
                                long *out_loops_cut)
{
    assert(arena && out_verts && out_nv && out_faces && out_nf);
    if (nv < 3 || nf < 1 || !verts || !faces) return -1;

    MeshTopoInfo ti; MeshTopo_analyze(arena, NULL, nv, faces, nf, &ti);
    long g0 = lround(ti.genus);

    const float   *cv = verts; size_t cnv = nv;
    const int32_t *cf = faces; size_t cnf = nf;
    int32_t *accum = NULL; long loops_cut = 0;
    if (g0 > 0)
        genus_reduce(arena, &cv, &cnv, &cf, &cnf, &accum, g0, &loops_cut, NULL, NULL,
                     max_loop_len);

    float   *ov = (float *)  ARENA_ALLOC(arena, (size_t)(cnv * 3 * sizeof(float)));
    int32_t *of = (int32_t *)ARENA_ALLOC(arena, (size_t)(cnf * 3 * sizeof(int32_t)));
    memcpy(ov, cv, cnv * 3 * sizeof(float));
    memcpy(of, cf, cnf * 3 * sizeof(int32_t));
    *out_verts = ov; *out_nv = cnv;
    *out_faces = of; *out_nf = cnf;
    if (out_loops_cut) *out_loops_cut = loops_cut;
    return 0;
}

int SeamCut_repair_manifold(Arena_T arena, const float *verts, size_t nv,
                            const int32_t *faces, size_t nf,
                            float **out_verts, size_t *out_nv,
                            int32_t **out_faces, size_t *out_nf,
                            int32_t **out_vmap)
{
    assert(arena && out_verts && out_nv && out_faces && out_nf && out_vmap);
    if (nv < 3 || nf < 1 || !verts || !faces) return -1;

    size_t nc = 3*nf;
    char *drop = (char *)ARENA_CALLOC(arena, (size_t)nf, 1L);
    long ndrop = 0, ndegen = 0;
    for (size_t t = 0; t < nf; t++) {
        int32_t a = faces[t*3+0], b = faces[t*3+1], c = faces[t*3+2];
        if (a == b || b == c || c == a) { drop[t] = 1; ndrop++; ndegen++; }
    }
    {
        Arena_Mark m = Arena_save(arena);
        EKey *ek = (EKey *)ARENA_ALLOC(arena, (size_t)(nc * sizeof(EKey)));
        size_t nek = 0;
        for (size_t cc = 0; cc < nc; cc++) {
            if (drop[cc/3]) continue;
            int32_t a = faces[cc], b = faces[NEXT(cc)];
            uint64_t lo = (uint64_t)(a < b ? a : b), hi = (uint64_t)(a < b ? b : a);
            ek[nek].key = lo*(uint64_t)nv + hi; ek[nek].corner = (int32_t)cc; nek++;
        }
        qsort(ek, nek, sizeof(EKey), cmp_ekey);
        size_t i = 0;
        while (i < nek) {
            size_t j = i; while (j < nek && ek[j].key == ek[i].key) j++;
            if (j - i > 2)
                for (size_t p = i; p < j; p++) { int32_t f = ek[p].corner/3;
                    if (!drop[f]) { drop[f] = 1; ndrop++; } }
            i = j;
        }
        Arena_restore(arena, m);
    }
    const int32_t *use_faces = faces; size_t use_nf = nf;
    if (ndrop > 0) {
        fprintf(stderr, "  [seamcut] manifold repair: dropped %ld face(s) (%ld degenerate, %ld on non-manifold edges)\n",
                ndrop, ndegen, ndrop - ndegen);
        int32_t *ff = (int32_t *)ARENA_ALLOC(arena, (size_t)((nf - (size_t)ndrop) * 3 * sizeof(int32_t)));
        size_t fnf = 0;
        for (size_t t = 0; t < nf; t++) if (!drop[t]) {
            ff[fnf*3+0]=faces[t*3+0]; ff[fnf*3+1]=faces[t*3+1]; ff[fnf*3+2]=faces[t*3+2]; fnf++;
        }
        use_faces = ff; use_nf = fnf;
        if (use_nf < 1) return -1;
    }

    Conn c; conn_build(arena, verts, nv, use_faces, use_nf, &c);
    SeamSet seam; seam_init(arena, &seam, nv, 16);
    SeamCutResult tmp; memset(&tmp, 0, sizeof tmp);
    if (cut_surgery(arena, &c, &seam, &tmp) != 0) return -1;
    *out_verts = tmp.verts; *out_nv = tmp.nv;
    *out_faces = tmp.faces; *out_nf = tmp.nf; *out_vmap = tmp.vmap;
    return 0;
}

int SeamCut_run(Arena_T arena, const float *verts, size_t nv,
                const int32_t *faces, size_t nf,
                const SeamCutOpts *opts, SeamCutResult *out)
{
    assert(arena && out);
    memset(out, 0, sizeof(*out));
    if (nv < 3 || nf < 1 || !verts || !faces) return -1;
    double accept = opts ? opts->accept_frac : 0.0;
    int r_max = (opts && opts->r_max>0) ? opts->r_max : 5;

    Arena_Mark mark = Arena_save(arena);

    MeshTopoInfo ti0; MeshTopo_analyze(arena, NULL, nv, faces, nf, &ti0);
    out->loops_before = ti0.n_boundary_loops;
    out->genus_before = lround(ti0.genus);

    int dbg = (getenv("SEAMCUT_DEBUG") != NULL);

    const float   *cv = verts; size_t cnv = nv;
    const int32_t *cf = faces; size_t cnf = nf;
    int32_t *accum = NULL; long loops_cut = 0;
    double tph = ves_clock_sec();
    if (out->genus_before > 0) {
        genus_reduce(arena, &cv, &cnv, &cf, &cnf, &accum, out->genus_before, &loops_cut,
                     NULL, NULL, 0.0);
        if (loops_cut > 0)
            fprintf(stderr, "  [seamcut] genus %ld -> 0 via %ld non-separating loop cut(s)\n",
                    out->genus_before, loops_cut);
    }
    if (dbg){ fprintf(stderr,"  [seamcut] phase genus_reduce %.2fs (nv %zu->%zu)\n", ves_clock_sec()-tph, nv, cnv); tph=ves_clock_sec(); }

    if (accept >= 1.0) {
        float *dv=NULL;
        int32_t *df=NULL, *dmap=NULL;
        size_t dnv=0, dnf=0;
        long boundary_loops=0, seam_edges=0;
        if (boundary_reduce_topology(arena,cv,cnv,cf,cnf,
                                     &dv,&dnv,&df,&dnf,&dmap,
                                     &boundary_loops,&seam_edges)!=0) {
            fprintf(stderr,
                    "  [seamcut] sequential boundary reduction failed "
                    "(V=%zu F=%zu)\n",cnv,cnf);
            Arena_restore(arena,mark);
            return -1;
        }
        out->verts=dv; out->nv=dnv;
        out->faces=df; out->nf=dnf; out->vmap=dmap;
        out->n_terminals=boundary_loops;
        out->seam_edges=seam_edges;
        if (accum)
            for (size_t v=0;v<out->nv;v++)
                out->vmap[v]=accum[out->vmap[v]];
        {
            MeshTopoInfo ti1;
            MeshTopo_analyze(arena,NULL,out->nv,out->faces,out->nf,&ti1);
            out->loops_after=ti1.n_boundary_loops;
            out->genus_after=lround(ti1.genus);
            out->is_disk_after=ti1.is_disk;
        }
        if (dbg)
            fprintf(stderr,
                    "  [seamcut] phase sequential-boundary %.2fs "
                    "(loops=%ld seam=%ld disk=%d)\n",
                    ves_clock_sec()-tph,boundary_loops,seam_edges,
                    out->is_disk_after);
        (void)mark;
        return 0;
    }

    Conn c; conn_build(arena, cv, cnv, cf, cnf, &c);

    double  *Dr = (double *)ARENA_ALLOC(arena,(size_t)(cnv*sizeof(double)));
    int32_t *br = (int32_t *)ARENA_ALLOC(arena,(size_t)(cnv*sizeof(int32_t)));
    if (accept >= 1.0) {
        memset(Dr, 0, cnv*sizeof(double));
        for (size_t v=0; v<cnv; v++) br[v] = -1;
    } else {
        compute_distortion(arena, &c, r_max, Dr, br);
    }
    if (dbg){ fprintf(stderr,"  [seamcut] phase distortion %.2fs\n", ves_clock_sec()-tph); tph=ves_clock_sec(); }

    char *is_bnd = (char *)ARENA_CALLOC(arena,(size_t)cnv,1L);
    for (size_t cc=0; cc<c.nc; cc++) if (c.twin[cc]<0){ is_bnd[c.faces[cc]]=1; is_bnd[c.faces[NEXT(cc)]]=1; }

    char *is_term=(char*)ARENA_CALLOC(arena,(size_t)cnv,1L);
    long nterm=0;
    if (accept >= 1.0) {
        nterm = boundary_loop_terminals(arena, &c, is_bnd, is_term);
    } else {
        double Dtot=0.0, s=0.0;
        double Dacc;
        int32_t *order;
        for (size_t v=0;v<cnv;v++) if(Dr[v]>0.0) Dtot+=Dr[v];
        Dacc = accept*Dtot;
        order=(int32_t*)ARENA_ALLOC(arena,(size_t)(cnv*sizeof(int32_t)));
        {
            SeamDI *di=(SeamDI*)ARENA_ALLOC(arena,(size_t)(cnv*sizeof(SeamDI)));
            for(size_t v=0;v<cnv;v++){di[v].d=Dr[v];di[v].i=(int32_t)v;}
            qsort(di, cnv, sizeof(SeamDI), cmp_di_desc);
            for(size_t v=0;v<cnv;v++) order[v]=di[v].i;
        }
        for (size_t k=0; k<cnv; k++){
            int32_t v=order[k];
            if (Dr[v]<=0.0) break;
            if (s >= Dtot - Dacc) break;
            is_term[v]=1; nterm++; s+=Dr[v];
        }
        for (size_t v=0;v<cnv;v++)
            if(is_bnd[v] && !is_term[v]){ is_term[v]=1; nterm++; }
    }
    out->n_terminals=nterm;

    SeamSet seam; seam_init(arena, &seam, cnv, c.nc);
    if (nterm >= 2) steiner_tree(arena, &c, is_term, &seam);
    out->seam_edges=(long)seam.n;
    if (dbg){
        long si=0, sb=0;
        for (size_t cc=0; cc<c.nc; cc++){
            int32_t tw=c.twin[cc];
            if (tw>=0 && (int32_t)cc>tw) continue;
            if (!seam_has(&seam, c.faces[cc], c.faces[NEXT(cc)])) continue;
            if (tw<0) sb++; else si++;
        }
        fprintf(stderr,"  [seamcut] phase steiner %.2fs (nterm=%ld seam=%ld: interior=%ld boundary=%ld)\n",
                ves_clock_sec()-tph, nterm, (long)seam.n, si, sb); tph=ves_clock_sec();
    }

    if (cut_surgery(arena, &c, &seam, out)!=0){ Arena_restore(arena,mark); return -1; }
    if (dbg){ fprintf(stderr,"  [seamcut] phase surgery %.2fs\n", ves_clock_sec()-tph); tph=ves_clock_sec(); }

    if (accum) for (size_t i=0;i<out->nv;i++) out->vmap[i]=accum[out->vmap[i]];

    MeshTopoInfo ti1; MeshTopo_analyze(arena, NULL, out->nv, out->faces, out->nf, &ti1);
    out->loops_after = ti1.n_boundary_loops;
    out->genus_after = lround(ti1.genus);
    out->is_disk_after = ti1.is_disk;
    (void)mark;
    return 0;
}

static void build_grid(Arena_T arena, int nu, int nh, int wrap, double R,
                       float **out_v, size_t *out_nv, int32_t **out_f, size_t *out_nf)
{
    size_t nvv=(size_t)nu*(size_t)nh; int uc=wrap?nu:nu-1;
    size_t nff=(size_t)uc*(size_t)(nh-1)*2;
    float *v=(float*)ARENA_ALLOC(arena,(size_t)(nvv*3*sizeof(float)));
    int32_t *f=(int32_t*)ARENA_ALLOC(arena,(size_t)(nff*3*sizeof(int32_t)));
    for(int j=0;j<nh;j++)for(int i=0;i<nu;i++){
        double th=2.0*M_PI*(double)i/(double)nu, z=2.0*(double)j;
        size_t idx=(size_t)j*(size_t)nu+(size_t)i;
        v[idx*3+0]=(float)z; v[idx*3+1]=(float)(R*sin(th)); v[idx*3+2]=(float)(R*cos(th));
    }
    size_t fi=0;
    for(int j=0;j<nh-1;j++)for(int i=0;i<uc;i++){
        int i2=wrap?(i+1)%nu:i+1;
        int32_t a=(int32_t)((size_t)j*(size_t)nu+(size_t)i), b=(int32_t)((size_t)j*(size_t)nu+(size_t)i2);
        int32_t cc=(int32_t)((size_t)(j+1)*(size_t)nu+(size_t)i), d=(int32_t)((size_t)(j+1)*(size_t)nu+(size_t)i2);
        f[fi*3+0]=a;f[fi*3+1]=b;f[fi*3+2]=cc;fi++; f[fi*3+0]=b;f[fi*3+1]=d;f[fi*3+2]=cc;fi++;
    }
    *out_v=v;*out_nv=nvv;*out_f=f;*out_nf=fi;
}

static void build_many_hole_grid(Arena_T arena,
                                 float **out_v, size_t *out_nv,
                                 int32_t **out_f, size_t *out_nf)
{
    const int nu = 20, nh = 16;
    size_t nvv = (size_t)nu*(size_t)nh;
    size_t cap = (size_t)(nu-1)*(size_t)(nh-1)*2;
    float *v=(float*)ARENA_ALLOC(arena,(size_t)(nvv*3*sizeof(float)));
    int32_t *f=(int32_t*)ARENA_ALLOC(arena,(size_t)(cap*3*sizeof(int32_t)));
    size_t fi=0;
    for (int j=0;j<nh;j++) for (int i=0;i<nu;i++) {
        size_t p=(size_t)j*(size_t)nu+(size_t)i;
        v[p*3]=(float)i; v[p*3+1]=(float)j; v[p*3+2]=0.0f;
    }
    for (int j=0;j<nh-1;j++) for (int i=0;i<nu-1;i++) {
        int hole_i=(i==3 || i==7 || i==11 || i==15);
        int hole_j=(j==3 || j==7 || j==11);
        int32_t a,b,cc,d;
        if (hole_i && hole_j) continue;
        a=(int32_t)((size_t)j*(size_t)nu+(size_t)i);
        b=(int32_t)((size_t)j*(size_t)nu+(size_t)(i+1));
        cc=(int32_t)((size_t)(j+1)*(size_t)nu+(size_t)i);
        d=(int32_t)((size_t)(j+1)*(size_t)nu+(size_t)(i+1));
        f[fi*3]=a; f[fi*3+1]=b; f[fi*3+2]=cc; fi++;
        f[fi*3]=b; f[fi*3+1]=d; f[fi*3+2]=cc; fi++;
    }
    *out_v=v; *out_nv=nvv; *out_f=f; *out_nf=fi;
}

static void build_torus(Arena_T arena, int nu, int nr, double Rb, double rt,
                        int skip_quad, float **out_v, size_t *out_nv,
                        int32_t **out_f, size_t *out_nf)
{
    size_t NV=(size_t)nu*(size_t)nr;
    float *v=(float*)ARENA_ALLOC(arena,(size_t)(NV*3*sizeof(float)));
    int32_t *f=(int32_t*)ARENA_ALLOC(arena,(size_t)((size_t)nu*(size_t)nr*2*3*sizeof(int32_t)));
    for(int j=0;j<nr;j++)for(int i=0;i<nu;i++){
        double phi=2.0*M_PI*(double)i/(double)nu, th=2.0*M_PI*(double)j/(double)nr;
        size_t idx=(size_t)j*(size_t)nu+(size_t)i;
        v[idx*3+0]=(float)((Rb+rt*cos(phi))*cos(th));
        v[idx*3+1]=(float)((Rb+rt*cos(phi))*sin(th));
        v[idx*3+2]=(float)(rt*sin(phi));
    }
    size_t fi=0;
    for(int j=0;j<nr;j++)for(int i=0;i<nu;i++){
        if(skip_quad && i==0 && j==0) continue;
        int i2=(i+1)%nu, j2=(j+1)%nr;
        int32_t a=(int32_t)((size_t)j *(size_t)nu+(size_t)i);
        int32_t b=(int32_t)((size_t)j *(size_t)nu+(size_t)i2);
        int32_t cc=(int32_t)((size_t)j2*(size_t)nu+(size_t)i);
        int32_t d=(int32_t)((size_t)j2*(size_t)nu+(size_t)i2);
        f[fi*3+0]=a;f[fi*3+1]=b;f[fi*3+2]=cc;fi++;
        f[fi*3+0]=b;f[fi*3+1]=d;f[fi*3+2]=cc;fi++;
    }
    *out_v=v;*out_nv=NV;*out_f=f;*out_nf=fi;
}

#include "seam_cut_fan_test.inc"

int SeamCut_selftest(void)
{
    int fails=0; Arena_T arena=Arena_new();
    fails+=seamcut_fan_callback_selftest(arena);
    for (int saddle=0;saddle<2;saddle++) {
        float vertices[27]={0};int32_t faces[24];uint8_t terminals[9]={2};
        for (int k=0;k<8;k++) {
            double t=2*M_PI*k/8;
            vertices[3*(k+1)]=(float)cos(t);
            vertices[3*(k+1)+1]=(float)sin(t);
            vertices[3*(k+1)+2]=saddle?(k&1?.4f:-.4f):0;
            faces[3*k]=0;faces[3*k+1]=k+1;faces[3*k+2]=(k+1)%8+1;
        }
        SeamCutResult cut={0};
        int rc=SeamCut_to_boundary(arena,vertices,9,faces,8,terminals,&cut);
        int ok=rc==0 && cut.nf==8,center_copies=0;
        double maximum=0;
        if (!rc) {
            double *angles=ARENA_CALLOC(arena,cut.nv,sizeof *angles);
            for (size_t k=0;k<3*cut.nf;k++) {
                if (cut.vmap[cut.faces[k]]!=faces[k]) ok=0;
                if (cut.vmap[cut.faces[k]]==0)
                    angles[cut.faces[k]]+=corner_angle(cut.verts+3*(size_t)cut.faces[k],
                        cut.verts+3*(size_t)cut.faces[NEXT(k)],cut.verts+3*(size_t)cut.faces[PREV(k)]);
            }
            for (size_t v=0;v<cut.nv;v++) {
                if (memcmp(cut.verts+3*v,vertices+3*(size_t)cut.vmap[v],3*sizeof(float))) ok=0;
                if (cut.vmap[v]==0) { center_copies++;maximum=fmax(maximum,angles[v]); }
            }
        }
        ok&=maximum<=2*M_PI+1e-10 && (saddle?center_copies>=2:center_copies==1);
        ok&=saddle?cut.seam_edges>=2:cut.seam_edges==1;
        fprintf(stderr,"  [seamcut %s fan] rc=%d copies=%d max-angle=%.6g seam=%ld -> %s\n",
                saddle?"negative-cone":"planar",rc,center_copies,maximum,cut.seam_edges,ok?"ok":"FAIL");
        fails+=!ok;
    }
    #include "seam_cut_direction_test.inc"
    {
        float *v; int32_t *f; size_t nvv,nff;
        build_grid(arena, 24, 10, 1, 10.0, &v,&nvv,&f,&nff);
        SeamCutOpts o={0.0,5}; SeamCutResult res;
        int rc=SeamCut_run(arena, v,nvv,f,nff,&o,&res);
        int ok=(rc==0) && res.loops_before==2 && res.loops_after==1 && res.is_disk_after;
        fprintf(stderr,"  [seamcut annulus] rc=%d loops %ld->%ld seam=%ld disk=%d nv %zu->%zu -> %s\n",
                rc,res.loops_before,res.loops_after,res.seam_edges,res.is_disk_after,nvv,res.nv, ok?"ok":"FAIL");
        fails+=ok?0:1;
    }
    {
        float *v; int32_t *f; size_t nvv,nff;
        build_many_hole_grid(arena, &v,&nvv,&f,&nff);
        SeamCutOpts o={1.0,5}; SeamCutResult res;
        int rc=SeamCut_run(arena, v,nvv,f,nff,&o,&res);
        int ok=(rc==0) && res.loops_before==13 && res.n_terminals==13
                && res.loops_after==1 && res.is_disk_after;
        fprintf(stderr,"  [seamcut many-hole topology] rc=%d loops %ld->%ld "
                "terminals=%ld seam=%ld disk=%d -> %s\n",
                rc,res.loops_before,res.loops_after,res.n_terminals,
                res.seam_edges,res.is_disk_after,ok?"ok":"FAIL");
        fails+=ok?0:1;
    }
    {
        float *v; int32_t *f; size_t nvv,nff;
        build_grid(arena, 24, 10, 1, 10.0, &v,&nvv,&f,&nff);
        for (size_t t=0;t<nff;t+=5) {
            int32_t swap=f[t*3+1]; f[t*3+1]=f[t*3+2]; f[t*3+2]=swap;
        }
        SeamCutOpts o={1.0,5}; SeamCutResult res;
        int rc=SeamCut_run(arena, v,nvv,f,nff,&o,&res);
        int ok=(rc==0) && res.loops_before==2 && res.loops_after==1
                && res.is_disk_after;
        fprintf(stderr,"  [seamcut mixed-winding annulus] rc=%d loops %ld->%ld "
                "disk=%d -> %s\n",rc,res.loops_before,res.loops_after,
                res.is_disk_after,ok?"ok":"FAIL");
        fails+=ok?0:1;
    }
    {
        float *v; int32_t *f; size_t nvv,nff;
        build_grid(arena, 16, 10, 0, 10.0, &v,&nvv,&f,&nff);
        SeamCutOpts o={0.0,5}; SeamCutResult res;
        int rc=SeamCut_run(arena, v,nvv,f,nff,&o,&res);
        int ok=(rc==0) && res.loops_after==1 && res.is_disk_after;
        fprintf(stderr,"  [seamcut disk] rc=%d loops %ld->%ld disk=%d -> %s\n",
                rc,res.loops_before,res.loops_after,res.is_disk_after, ok?"ok":"FAIL");
        fails+=ok?0:1;
    }
    {
        float *v; int32_t *f; size_t nvv,nff;
        build_torus(arena, 20, 14, 10.0, 3.0, 0, &v,&nvv,&f,&nff);
        SeamCutOpts o={0.0,5}; SeamCutResult res;
        int rc=SeamCut_run(arena, v,nvv,f,nff,&o,&res);
        int vmap_ok=(rc==0);
        if (rc==0) for (size_t k=0;k<res.nv;k++)
            if (res.vmap[k]<0 || (size_t)res.vmap[k]>=nvv){ vmap_ok=0; break; }
        int ok=(rc==0) && res.genus_before==1 && res.genus_after==0
                && res.loops_after==1 && res.is_disk_after && vmap_ok;
        fprintf(stderr,"  [seamcut torus] rc=%d genus %ld->%ld loops ->%ld disk=%d vmap=%d nv %zu->%zu -> %s\n",
                rc,res.genus_before,res.genus_after,res.loops_after,res.is_disk_after,vmap_ok,nvv,res.nv, ok?"ok":"FAIL");
        fails+=ok?0:1;
    }
    {
        float *v; int32_t *f; size_t nvv,nff;
        build_torus(arena, 20, 14, 10.0, 3.0, 1, &v,&nvv,&f,&nff);
        SeamCutOpts o={0.0,5}; SeamCutResult res;
        int rc=SeamCut_run(arena, v,nvv,f,nff,&o,&res);
        int ok=(rc==0) && res.genus_before==1 && res.genus_after==0
                && res.loops_after==1 && res.is_disk_after;
        fprintf(stderr,"  [seamcut torus+hole] rc=%d genus %ld->%ld loops %ld->%ld disk=%d -> %s\n",
                rc,res.genus_before,res.genus_after,res.loops_before,res.loops_after,res.is_disk_after, ok?"ok":"FAIL");
        fails+=ok?0:1;
    }
    {
        float *v; int32_t *f; size_t nvv,nff;
        build_torus(arena, 20, 14, 10.0, 3.0, 0, &v,&nvv,&f,&nff);
        HandleLoop *loops=NULL; size_t nl=0;
        int rc=SeamCut_handle_loops(arena, v,nvv,f,nff, &loops,&nl);
        int idx_ok = (rc==0 && nl>=1 && loops[0].n>0);
        if (idx_ok) for (size_t k=0;k<loops[0].n;k++)
            if (loops[0].verts[k]<0 || (size_t)loops[0].verts[k]>=nvv){ idx_ok=0; break; }
        int ok=(rc==0 && nl==1 && idx_ok && loops[0].length>0.0);
        fprintf(stderr,"  [seamcut handle-loops torus] rc=%d nloops=%zu len=%.2f verts=%zu -> %s\n",
                rc, nl, nl?loops[0].length:0.0, nl?loops[0].n:0, ok?"ok":"FAIL");
        fails+=ok?0:1;

        float *vd; int32_t *fd; size_t nvd,nfd;
        build_grid(arena, 16, 10, 0, 10.0, &vd,&nvd,&fd,&nfd);
        HandleLoop *ld=NULL; size_t nld=0;
        int rc2=SeamCut_handle_loops(arena, vd,nvd,fd,nfd, &ld,&nld);
        int ok2=(rc2==0 && nld==0);
        fprintf(stderr,"  [seamcut handle-loops disk] rc=%d nloops=%zu -> %s\n",
                rc2, nld, ok2?"ok":"FAIL");
        fails+=ok2?0:1;
    }
    {
        float *v; int32_t *f; size_t nvv,nff;
        build_torus(arena, 20, 14, 10.0, 3.0, 0, &v,&nvv,&f,&nff);
        float *ov; int32_t *of; size_t onv,onf; long cut=0;
        int rc=SeamCut_sever_short_handles(arena, v,nvv,f,nff, 1000.0,
                                           &ov,&onv,&of,&onf,&cut);
        MeshTopoInfo ti; ti.genus=-1.0;
        if (rc==0) MeshTopo_analyze(arena, NULL, onv, of, onf, &ti);
        int ok=(rc==0 && cut==1 && lround(ti.genus)==0);
        fprintf(stderr,"  [seamcut sever short cap=1000] rc=%d cut=%ld genus->%ld -> %s\n",
                rc, cut, rc==0?lround(ti.genus):-1, ok?"ok":"FAIL");
        fails+=ok?0:1;

        float *ov2; int32_t *of2; size_t onv2,onf2; long cut2=0;
        int rc3=SeamCut_sever_short_handles(arena, v,nvv,f,nff, 10.0,
                                            &ov2,&onv2,&of2,&onf2,&cut2);
        MeshTopoInfo ti2; ti2.genus=-1.0;
        if (rc3==0) MeshTopo_analyze(arena, NULL, onv2, of2, onf2, &ti2);
        int ok3=(rc3==0 && cut2==0 && lround(ti2.genus)==1);
        fprintf(stderr,"  [seamcut sever short cap=10] rc=%d cut=%ld genus->%ld (expect kept) -> %s\n",
                rc3, cut2, rc3==0?lround(ti2.genus):-1, ok3?"ok":"FAIL");
        fails+=ok3?0:1;
    }
    Arena_dispose(&arena);
    return fails;
}
