/*  Copyright 1993, 1995, 1997, 1998, 2002, 2005                             */
/*    copyright notices (including this entire header and the copyright      */
/*    SAME FILE REMAIN UNDER COPYRIGHT OF THE ORIGINAL AUTHOR, BOTH SOURCE   */



#ifdef SINGLE
#define REAL float
#else
#define REAL double
#endif











#define INEXACT

#include <stdint.h>
#include <setjmp.h>
#define TRIULONG uintptr_t

jmp_buf triangle_jmpbuf;
int triangle_jmpbuf_set = 0;

int triangle_malloc_fill = -1;


#define FILENAMESIZE 2048


#define INPUTLINESIZE 1024


#define TRIPERBLOCK 4092
#define SUBSEGPERBLOCK 508
#define VERTEXPERBLOCK 4092
#define VIRUSPERBLOCK 1020
#define BADSUBSEGPERBLOCK 252
#define BADTRIPERBLOCK 4092
#define FLIPSTACKERPERBLOCK 252
#define SPLAYNODEPERBLOCK 508


#define INPUTVERTEX 0
#define SEGMENTVERTEX 1
#define FREEVERTEX 2
#define DEADVERTEX -32768
#define UNDEADVERTEX -32767


#define VOID int



#define SAMPLEFACTOR 11


#define SAMPLERATE 10


#define PI 3.141592653589793238462643383279502884197169399375105820974944592308


#define SQUAREROOTTWO 1.4142135623730950488016887242096980785696718753769480732


#define ONETHIRD 0.333333333333333333333333333333333333333333333333333333333333

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#ifndef NO_TIMER
#include <sys/time.h>
#endif
#ifdef CPU86
#include <float.h>
#endif
#ifdef LINUX
#include <fpu_control.h>
#endif
#ifdef TRILIBRARY
#include "triangle.h"
#endif


#ifndef TRILIBRARY
char *readline();
char *findfield();
#endif


enum locateresult {INTRIANGLE, ONEDGE, ONVERTEX, OUTSIDE};


enum insertvertexresult {SUCCESSFULVERTEX, ENCROACHINGVERTEX, VIOLATINGVERTEX,
                         DUPLICATEVERTEX};


enum finddirectionresult {WITHIN, LEFTCOLLINEAR, RIGHTCOLLINEAR};




typedef REAL **triangle;


struct otri {
  triangle *tri;
  int orient;
};


typedef REAL **subseg;


struct osub {
  subseg *ss;
  int ssorient;
};


typedef REAL *vertex;


struct badsubseg {
  subseg encsubseg;
  vertex subsegorg, subsegdest;
};


struct badtriang {
  triangle poortri;
  REAL key;
  vertex triangorg, triangdest, triangapex;
  struct badtriang *nexttriang;
};


struct flipstacker {
  triangle flippedtri;
  struct flipstacker *prevflip;
};


struct event {
  REAL xkey, ykey;
  VOID *eventptr;
  int heapposition;
};


struct splaynode {
  struct otri keyedge;
  vertex keydest;
  struct splaynode *lchild, *rchild;
};


struct memorypool {
  VOID **firstblock, **nowblock;
  VOID *nextitem;
  VOID *deaditemstack;
  VOID **pathblock;
  VOID *pathitem;
  int alignbytes;
  int itembytes;
  int itemsperblock;
  int itemsfirstblock;
  long items, maxitems;
  int unallocateditems;
  int pathitemsleft;
};



REAL splitter;
REAL epsilon;
REAL resulterrbound;
REAL ccwerrboundA, ccwerrboundB, ccwerrboundC;
REAL iccerrboundA, iccerrboundB, iccerrboundC;
REAL o3derrboundA, o3derrboundB, o3derrboundC;


TRIULONG randomseed;



struct mesh {


  struct memorypool triangles;
  struct memorypool subsegs;
  struct memorypool vertices;
  struct memorypool viri;
  struct memorypool badsubsegs;
  struct memorypool badtriangles;
  struct memorypool flipstackers;
  struct memorypool splaynodes;


  struct badtriang *queuefront[4096];
  struct badtriang *queuetail[4096];
  int nextnonemptyq[4096];
  int firstnonemptyq;


  struct flipstacker *lastflip;


  REAL xmin, xmax, ymin, ymax;
  REAL xminextreme;
  int invertices;
  int inelements;
  int insegments;
  int holes;
  int regions;
  int undeads;
  long edges;
  int mesh_dim;
  int nextras;
  int eextras;
  long hullsize;
  int steinerleft;
  int vertexmarkindex;
  int vertex2triindex;
  int highorderindex;
  int elemattribindex;
  int areaboundindex;
  int checksegments;
  int checkquality;
  int readnodefile;
  long samples;

  long incirclecount;
  long counterclockcount;
  long orient3dcount;
  long hyperbolacount;
  long circumcentercount;
  long circletopcount;


  vertex infvertex1, infvertex2, infvertex3;


  triangle *dummytri;
  triangle *dummytribase;


  subseg *dummysub;
  subseg *dummysubbase;


  struct otri recenttri;

};



struct behavior {


  int poly, refine, quality, vararea, fixedarea, usertest;
  int regionattrib, convex, weighted, jettison;
  int firstnumber;
  int edgesout, voronoi, neighbors, geomview;
  int nobound, nopolywritten, nonodewritten, noelewritten, noiterationnum;
  int noholes, noexact, conformdel;
  int incremental, sweepline, dwyer;
  int splitseg;
  int docheck;
  int quiet, verbose;
  int usesegments;
  int order;
  int nobisect;
  int steiner;
  REAL minangle, goodangle, offconstant;
  REAL maxarea;


#ifndef TRILIBRARY
  char innodefilename[FILENAMESIZE];
  char inelefilename[FILENAMESIZE];
  char inpolyfilename[FILENAMESIZE];
  char areafilename[FILENAMESIZE];
  char outnodefilename[FILENAMESIZE];
  char outelefilename[FILENAMESIZE];
  char outpolyfilename[FILENAMESIZE];
  char edgefilename[FILENAMESIZE];
  char vnodefilename[FILENAMESIZE];
  char vedgefilename[FILENAMESIZE];
  char neighborfilename[FILENAMESIZE];
  char offfilename[FILENAMESIZE];
#endif

};





int plus1mod3[3] = {1, 2, 0};
int minus1mod3[3] = {2, 0, 1};



#define decode(ptr, otri)                                                     \
  (otri).orient = (int) ((TRIULONG) (ptr) & (TRIULONG) 3l);         \
  (otri).tri = (triangle *)                                                   \
                  ((TRIULONG) (ptr) ^ (TRIULONG) (otri).orient)


#define encode(otri)                                                          \
  (triangle) ((TRIULONG) (otri).tri | (TRIULONG) (otri).orient)



#define sym(otri1, otri2)                                                     \
  ptr = (otri1).tri[(otri1).orient];                                          \
  decode(ptr, otri2);

#define symself(otri)                                                         \
  ptr = (otri).tri[(otri).orient];                                            \
  decode(ptr, otri);


#define lnext(otri1, otri2)                                                   \
  (otri2).tri = (otri1).tri;                                                  \
  (otri2).orient = plus1mod3[(otri1).orient]

#define lnextself(otri)                                                       \
  (otri).orient = plus1mod3[(otri).orient]


#define lprev(otri1, otri2)                                                   \
  (otri2).tri = (otri1).tri;                                                  \
  (otri2).orient = minus1mod3[(otri1).orient]

#define lprevself(otri)                                                       \
  (otri).orient = minus1mod3[(otri).orient]


#define onext(otri1, otri2)                                                   \
  lprev(otri1, otri2);                                                        \
  symself(otri2);

#define onextself(otri)                                                       \
  lprevself(otri);                                                            \
  symself(otri);


#define oprev(otri1, otri2)                                                   \
  sym(otri1, otri2);                                                          \
  lnextself(otri2);

#define oprevself(otri)                                                       \
  symself(otri);                                                              \
  lnextself(otri);


#define dnext(otri1, otri2)                                                   \
  sym(otri1, otri2);                                                          \
  lprevself(otri2);

#define dnextself(otri)                                                       \
  symself(otri);                                                              \
  lprevself(otri);


#define dprev(otri1, otri2)                                                   \
  lnext(otri1, otri2);                                                        \
  symself(otri2);

#define dprevself(otri)                                                       \
  lnextself(otri);                                                            \
  symself(otri);


#define rnext(otri1, otri2)                                                   \
  sym(otri1, otri2);                                                          \
  lnextself(otri2);                                                           \
  symself(otri2);

#define rnextself(otri)                                                       \
  symself(otri);                                                              \
  lnextself(otri);                                                            \
  symself(otri);


#define rprev(otri1, otri2)                                                   \
  sym(otri1, otri2);                                                          \
  lprevself(otri2);                                                           \
  symself(otri2);

#define rprevself(otri)                                                       \
  symself(otri);                                                              \
  lprevself(otri);                                                            \
  symself(otri);


#define org(otri, vertexptr)                                                  \
  vertexptr = (vertex) (otri).tri[plus1mod3[(otri).orient] + 3]

#define dest(otri, vertexptr)                                                 \
  vertexptr = (vertex) (otri).tri[minus1mod3[(otri).orient] + 3]

#define apex(otri, vertexptr)                                                 \
  vertexptr = (vertex) (otri).tri[(otri).orient + 3]

#define setorg(otri, vertexptr)                                               \
  (otri).tri[plus1mod3[(otri).orient] + 3] = (triangle) vertexptr

#define setdest(otri, vertexptr)                                              \
  (otri).tri[minus1mod3[(otri).orient] + 3] = (triangle) vertexptr

#define setapex(otri, vertexptr)                                              \
  (otri).tri[(otri).orient + 3] = (triangle) vertexptr


#define bond(otri1, otri2)                                                    \
  (otri1).tri[(otri1).orient] = encode(otri2);                                \
  (otri2).tri[(otri2).orient] = encode(otri1)


#define dissolve(otri)                                                        \
  (otri).tri[(otri).orient] = (triangle) m->dummytri


#define otricopy(otri1, otri2)                                                \
  (otri2).tri = (otri1).tri;                                                  \
  (otri2).orient = (otri1).orient


#define otriequal(otri1, otri2)                                               \
  (((otri1).tri == (otri2).tri) &&                                            \
   ((otri1).orient == (otri2).orient))


#define infect(otri)                                                          \
  (otri).tri[6] = (triangle)                                                  \
                    ((TRIULONG) (otri).tri[6] | (TRIULONG) 2l)

#define uninfect(otri)                                                        \
  (otri).tri[6] = (triangle)                                                  \
                    ((TRIULONG) (otri).tri[6] & ~ (TRIULONG) 2l)


#define infected(otri)                                                        \
  (((TRIULONG) (otri).tri[6] & (TRIULONG) 2l) != 0l)


#define elemattribute(otri, attnum)                                           \
  ((REAL *) (otri).tri)[m->elemattribindex + (attnum)]

#define setelemattribute(otri, attnum, value)                                 \
  ((REAL *) (otri).tri)[m->elemattribindex + (attnum)] = value


#define areabound(otri)  ((REAL *) (otri).tri)[m->areaboundindex]

#define setareabound(otri, value)                                             \
  ((REAL *) (otri).tri)[m->areaboundindex] = value


#define deadtri(tria)  ((tria)[1] == (triangle) NULL)

#define killtri(tria)                                                         \
  (tria)[1] = (triangle) NULL;                                                \
  (tria)[3] = (triangle) NULL



#define sdecode(sptr, osub)                                                   \
  (osub).ssorient = (int) ((TRIULONG) (sptr) & (TRIULONG) 1l);      \
  (osub).ss = (subseg *)                                                      \
              ((TRIULONG) (sptr) & ~ (TRIULONG) 3l)


#define sencode(osub)                                                         \
  (subseg) ((TRIULONG) (osub).ss | (TRIULONG) (osub).ssorient)


#define ssym(osub1, osub2)                                                    \
  (osub2).ss = (osub1).ss;                                                    \
  (osub2).ssorient = 1 - (osub1).ssorient

#define ssymself(osub)                                                        \
  (osub).ssorient = 1 - (osub).ssorient


#define spivot(osub1, osub2)                                                  \
  sptr = (osub1).ss[(osub1).ssorient];                                        \
  sdecode(sptr, osub2)

#define spivotself(osub)                                                      \
  sptr = (osub).ss[(osub).ssorient];                                          \
  sdecode(sptr, osub)


#define snext(osub1, osub2)                                                   \
  sptr = (osub1).ss[1 - (osub1).ssorient];                                    \
  sdecode(sptr, osub2)

#define snextself(osub)                                                       \
  sptr = (osub).ss[1 - (osub).ssorient];                                      \
  sdecode(sptr, osub)


#define sorg(osub, vertexptr)                                                 \
  vertexptr = (vertex) (osub).ss[2 + (osub).ssorient]

#define sdest(osub, vertexptr)                                                \
  vertexptr = (vertex) (osub).ss[3 - (osub).ssorient]

#define setsorg(osub, vertexptr)                                              \
  (osub).ss[2 + (osub).ssorient] = (subseg) vertexptr

#define setsdest(osub, vertexptr)                                             \
  (osub).ss[3 - (osub).ssorient] = (subseg) vertexptr

#define segorg(osub, vertexptr)                                               \
  vertexptr = (vertex) (osub).ss[4 + (osub).ssorient]

#define segdest(osub, vertexptr)                                              \
  vertexptr = (vertex) (osub).ss[5 - (osub).ssorient]

#define setsegorg(osub, vertexptr)                                            \
  (osub).ss[4 + (osub).ssorient] = (subseg) vertexptr

#define setsegdest(osub, vertexptr)                                           \
  (osub).ss[5 - (osub).ssorient] = (subseg) vertexptr


#define mark(osub)  (* (int *) ((osub).ss + 8))

#define setmark(osub, value)                                                  \
  * (int *) ((osub).ss + 8) = value


#define sbond(osub1, osub2)                                                   \
  (osub1).ss[(osub1).ssorient] = sencode(osub2);                              \
  (osub2).ss[(osub2).ssorient] = sencode(osub1)


#define sdissolve(osub)                                                       \
  (osub).ss[(osub).ssorient] = (subseg) m->dummysub


#define subsegcopy(osub1, osub2)                                              \
  (osub2).ss = (osub1).ss;                                                    \
  (osub2).ssorient = (osub1).ssorient


#define subsegequal(osub1, osub2)                                             \
  (((osub1).ss == (osub2).ss) &&                                              \
   ((osub1).ssorient == (osub2).ssorient))


#define deadsubseg(sub)  ((sub)[1] == (subseg) NULL)

#define killsubseg(sub)                                                       \
  (sub)[1] = (subseg) NULL;                                                   \
  (sub)[2] = (subseg) NULL



#define tspivot(otri, osub)                                                   \
  sptr = (subseg) (otri).tri[6 + (otri).orient];                              \
  sdecode(sptr, osub)


#define stpivot(osub, otri)                                                   \
  ptr = (triangle) (osub).ss[6 + (osub).ssorient];                            \
  decode(ptr, otri)


#define tsbond(otri, osub)                                                    \
  (otri).tri[6 + (otri).orient] = (triangle) sencode(osub);                   \
  (osub).ss[6 + (osub).ssorient] = (subseg) encode(otri)


#define tsdissolve(otri)                                                      \
  (otri).tri[6 + (otri).orient] = (triangle) m->dummysub


#define stdissolve(osub)                                                      \
  (osub).ss[6 + (osub).ssorient] = (subseg) m->dummytri


#define vertexmark(vx)  ((int *) (vx))[m->vertexmarkindex]

#define setvertexmark(vx, value)                                              \
  ((int *) (vx))[m->vertexmarkindex] = value

#define vertextype(vx)  ((int *) (vx))[m->vertexmarkindex + 1]

#define setvertextype(vx, value)                                              \
  ((int *) (vx))[m->vertexmarkindex + 1] = value

#define vertex2tri(vx)  ((triangle *) (vx))[m->vertex2triindex]

#define setvertex2tri(vx, value)                                              \
  ((triangle *) (vx))[m->vertex2triindex] = value




#ifdef EXTERNAL_TEST

int triunsuitable();

#else

#ifdef ANSI_DECLARATORS
int triunsuitable(vertex triorg, vertex tridest, vertex triapex, REAL area)
#else
int triunsuitable(triorg, tridest, triapex, area)
vertex triorg;
vertex tridest;
vertex triapex;
REAL area;
#endif

{
  REAL dxoa, dxda, dxod;
  REAL dyoa, dyda, dyod;
  REAL oalen, dalen, odlen;
  REAL maxlen;

  dxoa = triorg[0] - triapex[0];
  dyoa = triorg[1] - triapex[1];
  dxda = tridest[0] - triapex[0];
  dyda = tridest[1] - triapex[1];
  dxod = triorg[0] - tridest[0];
  dyod = triorg[1] - tridest[1];
   oalen = dxoa * dxoa + dyoa * dyoa;
  dalen = dxda * dxda + dyda * dyda;
  odlen = dxod * dxod + dyod * dyod;
   maxlen = (dalen > oalen) ? dalen : oalen;
  maxlen = (odlen > maxlen) ? odlen : maxlen;

  if (maxlen > 0.05 * (triorg[0] * triorg[0] + triorg[1] * triorg[1]) + 0.02) {
    return 1;
  } else {
    return 0;
  }
}

#endif



#ifdef ANSI_DECLARATORS
void triexit(int status)
#else
void triexit(status)
int status;
#endif

{
  if (triangle_jmpbuf_set) {
    longjmp(triangle_jmpbuf, status ? status : -1);
  }
  exit(status);
}

#ifdef ANSI_DECLARATORS
VOID *trimalloc(int size)
#else
VOID *trimalloc(size)
int size;
#endif

{
  VOID *memptr;

  memptr = (VOID *) malloc((unsigned int) size);
  if (memptr == (VOID *) NULL) {
    printf("Error:  Out of memory.\n");
    triexit(1);
  }
  if (triangle_malloc_fill >= 0) {
    memset(memptr, triangle_malloc_fill, (size_t) (unsigned int) size);
  }
  return(memptr);
}

#ifdef ANSI_DECLARATORS
void trifree(VOID *memptr)
#else
void trifree(memptr)
VOID *memptr;
#endif

{
  free(memptr);
}




#ifndef TRILIBRARY

void syntax()
{
#ifdef CDT_ONLY
#ifdef REDUCED
  printf("triangle [-pAcjevngBPNEIOXzo_lQVh] input_file\n");
#else
  printf("triangle [-pAcjevngBPNEIOXzo_iFlCQVh] input_file\n");
#endif
#else
#ifdef REDUCED
  printf("triangle [-prq__a__uAcDjevngBPNEIOXzo_YS__lQVh] input_file\n");
#else
  printf("triangle [-prq__a__uAcDjevngBPNEIOXzo_YS__iFlsCQVh] input_file\n");
#endif
#endif

  printf("    -p  Triangulates a Planar Straight Line Graph (.poly file).\n");
#ifndef CDT_ONLY
  printf("    -r  Refines a previously generated mesh.\n");
  printf(
    "    -q  Quality mesh generation.  A minimum angle may be specified.\n");
  printf("    -a  Applies a maximum triangle area constraint.\n");
  printf("    -u  Applies a user-defined triangle constraint.\n");
#endif
  printf(
    "    -A  Applies attributes to identify triangles in certain regions.\n");
  printf("    -c  Encloses the convex hull with segments.\n");
#ifndef CDT_ONLY
  printf("    -D  Conforming Delaunay:  all triangles are truly Delaunay.\n");
#endif
  printf("    -j  Jettison unused vertices from output .node file.\n");
  printf("    -e  Generates an edge list.\n");
  printf("    -v  Generates a Voronoi diagram.\n");
  printf("    -n  Generates a list of triangle neighbors.\n");
  printf("    -g  Generates an .off file for Geomview.\n");
  printf("    -B  Suppresses output of boundary information.\n");
  printf("    -P  Suppresses output of .poly file.\n");
  printf("    -N  Suppresses output of .node file.\n");
  printf("    -E  Suppresses output of .ele file.\n");
  printf("    -I  Suppresses mesh iteration numbers.\n");
  printf("    -O  Ignores holes in .poly file.\n");
  printf("    -X  Suppresses use of exact arithmetic.\n");
  printf("    -z  Numbers all items starting from zero (rather than one).\n");
  printf("    -o2 Generates second-order subparametric elements.\n");
#ifndef CDT_ONLY
  printf("    -Y  Suppresses boundary segment splitting.\n");
  printf("    -S  Specifies maximum number of added Steiner points.\n");
#endif
#ifndef REDUCED
  printf("    -i  Uses incremental method, rather than divide-and-conquer.\n");
  printf("    -F  Uses Fortune's sweepline algorithm, rather than d-and-c.\n");
#endif
  printf("    -l  Uses vertical cuts only, rather than alternating cuts.\n");
#ifndef REDUCED
#ifndef CDT_ONLY
  printf(
    "    -s  Force segments into mesh by splitting (instead of using CDT).\n");
#endif
  printf("    -C  Check consistency of final mesh.\n");
#endif
  printf("    -Q  Quiet:  No terminal output except errors.\n");
  printf("    -V  Verbose:  Detailed information on what I'm doing.\n");
  printf("    -h  Help:  Detailed instructions for Triangle.\n");
  triexit(0);
}

#endif


#ifndef TRILIBRARY

void info()
{
  printf("Triangle\n");
  printf(
"A Two-Dimensional Quality Mesh Generator and Delaunay Triangulator.\n");
  printf("Version 1.6\n\n");
  printf(
"Copyright 1993, 1995, 1997, 1998, 2002, 2005 Jonathan Richard Shewchuk\n");
  printf("2360 Woolsey #H / Berkeley, California 94705-1927\n");
  printf("Bugs/comments to jrs@cs.berkeley.edu\n");
  printf(
"Created as part of the Quake project (tools for earthquake simulation).\n");
  printf(
"Supported in part by NSF Grant CMS-9318163 and an NSERC 1967 Scholarship.\n");
  printf("There is no warranty whatsoever.  Use at your own risk.\n");
#ifdef SINGLE
  printf("This executable is compiled for single precision arithmetic.\n\n\n");
#else
  printf("This executable is compiled for double precision arithmetic.\n\n\n");
#endif
  printf(
"Triangle generates exact Delaunay triangulations, constrained Delaunay\n");
  printf(
"triangulations, conforming Delaunay triangulations, Voronoi diagrams, and\n");
  printf(
"high-quality triangular meshes.  The latter can be generated with no small\n"
);
  printf(
"or large angles, and are thus suitable for finite element analysis.  If no\n"
);
  printf(
"command line switch is specified, your .node input file is read, and the\n");
  printf(
"Delaunay triangulation is returned in .node and .ele output files.  The\n");
  printf("command syntax is:\n\n");
  printf("triangle [-prq__a__uAcDjevngBPNEIOXzo_YS__iFlsCQVh] input_file\n\n");
  printf(
"Underscores indicate that numbers may optionally follow certain switches.\n");
  printf(
"Do not leave any space between a switch and its numeric parameter.\n");
  printf(
"input_file must be a file with extension .node, or extension .poly if the\n");
  printf(
"-p switch is used.  If -r is used, you must supply .node and .ele files,\n");
  printf(
"and possibly a .poly file and an .area file as well.  The formats of these\n"
);
  printf("files are described below.\n\n");
  printf("Command Line Switches:\n\n");
  printf(
"    -p  Reads a Planar Straight Line Graph (.poly file), which can specify\n"
);
  printf(
"        vertices, segments, holes, regional attributes, and regional area\n");
  printf(
"        constraints.  Generates a constrained Delaunay triangulation (CDT)\n"
);
  printf(
"        fitting the input; or, if -s, -q, -a, or -u is used, a conforming\n");
  printf(
"        constrained Delaunay triangulation (CCDT).  If you want a truly\n");
  printf(
"        Delaunay (not just constrained Delaunay) triangulation, use -D as\n");
  printf(
"        well.  When -p is not used, Triangle reads a .node file by default.\n"
);
  printf(
"    -r  Refines a previously generated mesh.  The mesh is read from a .node\n"
);
  printf(
"        file and an .ele file.  If -p is also used, a .poly file is read\n");
  printf(
"        and used to constrain segments in the mesh.  If -a is also used\n");
  printf(
"        (with no number following), an .area file is read and used to\n");
  printf(
"        impose area constraints on the mesh.  Further details on refinement\n"
);
  printf("        appear below.\n");
  printf(
"    -q  Quality mesh generation by Delaunay refinement (a hybrid of Paul\n");
  printf(
"        Chew's and Jim Ruppert's algorithms).  Adds vertices to the mesh to\n"
);
  printf(
"        ensure that all angles are between 20 and 140 degrees.  An\n");
  printf(
"        alternative bound on the minimum angle, replacing 20 degrees, may\n");
  printf(
"        be specified after the `q'.  The specified angle may include a\n");
  printf(
"        decimal point, but not exponential notation.  Note that a bound of\n"
);
  printf(
"        theta degrees on the smallest angle also implies a bound of\n");
  printf(
"        (180 - 2 theta) on the largest angle.  If the minimum angle is 28.6\n"
);
  printf(
"        degrees or smaller, Triangle is mathematically guaranteed to\n");
  printf(
"        terminate (assuming infinite precision arithmetic--Triangle may\n");
  printf(
"        fail to terminate if you run out of precision).  In practice,\n");
  printf(
"        Triangle often succeeds for minimum angles up to 34 degrees.  For\n");
  printf(
"        some meshes, however, you might need to reduce the minimum angle to\n"
);
  printf(
"        avoid problems associated with insufficient floating-point\n");
  printf("        precision.\n");
  printf(
"    -a  Imposes a maximum triangle area.  If a number follows the `a', no\n");
  printf(
"        triangle is generated whose area is larger than that number.  If no\n"
);
  printf(
"        number is specified, an .area file (if -r is used) or .poly file\n");
  printf(
"        (if -r is not used) specifies a set of maximum area constraints.\n");
  printf(
"        An .area file contains a separate area constraint for each\n");
  printf(
"        triangle, and is useful for refining a finite element mesh based on\n"
);
  printf(
"        a posteriori error estimates.  A .poly file can optionally contain\n"
);
  printf(
"        an area constraint for each segment-bounded region, thereby\n");
  printf(
"        controlling triangle densities in a first triangulation of a PSLG.\n"
);
  printf(
"        You can impose both a fixed area constraint and a varying area\n");
  printf(
"        constraint by invoking the -a switch twice, once with and once\n");
  printf(
"        without a number following.  Each area specified may include a\n");
  printf("        decimal point.\n");
  printf(
"    -u  Imposes a user-defined constraint on triangle size.  There are two\n"
);
  printf(
"        ways to use this feature.  One is to edit the triunsuitable()\n");
  printf(
"        procedure in triangle.c to encode any constraint you like, then\n");
  printf(
"        recompile Triangle.  The other is to compile triangle.c with the\n");
  printf(
"        EXTERNAL_TEST symbol set (compiler switch -DEXTERNAL_TEST), then\n");
  printf(
"        link Triangle with a separate object file that implements\n");
  printf(
"        triunsuitable().  In either case, the -u switch causes the user-\n");
  printf("        defined test to be applied to every triangle.\n");
  printf(
"    -A  Assigns an additional floating-point attribute to each triangle\n");
  printf(
"        that identifies what segment-bounded region each triangle belongs\n");
  printf(
"        to.  Attributes are assigned to regions by the .poly file.  If a\n");
  printf(
"        region is not explicitly marked by the .poly file, triangles in\n");
  printf(
"        that region are assigned an attribute of zero.  The -A switch has\n");
  printf(
"        an effect only when the -p switch is used and the -r switch is not.\n"
);
  printf(
"    -c  Creates segments on the convex hull of the triangulation.  If you\n");
  printf(
"        are triangulating a vertex set, this switch causes a .poly file to\n"
);
  printf(
"        be written, containing all edges of the convex hull.  If you are\n");
  printf(
"        triangulating a PSLG, this switch specifies that the whole convex\n");
  printf(
"        hull of the PSLG should be triangulated, regardless of what\n");
  printf(
"        segments the PSLG has.  If you do not use this switch when\n");
  printf(
"        triangulating a PSLG, Triangle assumes that you have identified the\n"
);
  printf(
"        region to be triangulated by surrounding it with segments of the\n");
  printf(
"        input PSLG.  Beware:  if you are not careful, this switch can cause\n"
);
  printf(
"        the introduction of an extremely thin angle between a PSLG segment\n"
);
  printf(
"        and a convex hull segment, which can cause overrefinement (and\n");
  printf(
"        possibly failure if Triangle runs out of precision).  If you are\n");
  printf(
"        refining a mesh, the -c switch works differently:  it causes a\n");
  printf(
"        .poly file to be written containing the boundary edges of the mesh\n"
);
  printf("        (useful if no .poly file was read).\n");
  printf(
"    -D  Conforming Delaunay triangulation:  use this switch if you want to\n"
);
  printf(
"        ensure that all the triangles in the mesh are Delaunay, and not\n");
  printf(
"        merely constrained Delaunay; or if you want to ensure that all the\n"
);
  printf(
"        Voronoi vertices lie within the triangulation.  (Some finite volume\n"
);
  printf(
"        methods have this requirement.)  This switch invokes Ruppert's\n");
  printf(
"        original algorithm, which splits every subsegment whose diametral\n");
  printf(
"        circle is encroached.  It usually increases the number of vertices\n"
);
  printf("        and triangles.\n");
  printf(
"    -j  Jettisons vertices that are not part of the final triangulation\n");
  printf(
"        from the output .node file.  By default, Triangle copies all\n");
  printf(
"        vertices in the input .node file to the output .node file, in the\n");
  printf(
"        same order, so their indices do not change.  The -j switch prevents\n"
);
  printf(
"        duplicated input vertices, or vertices `eaten' by holes, from\n");
  printf(
"        appearing in the output .node file.  Thus, if two input vertices\n");
  printf(
"        have exactly the same coordinates, only the first appears in the\n");
  printf(
"        output.  If any vertices are jettisoned, the vertex numbering in\n");
  printf(
"        the output .node file differs from that of the input .node file.\n");
  printf(
"    -e  Outputs (to an .edge file) a list of edges of the triangulation.\n");
  printf(
"    -v  Outputs the Voronoi diagram associated with the triangulation.\n");
  printf(
"        Does not attempt to detect degeneracies, so some Voronoi vertices\n");
  printf(
"        may be duplicated.  See the discussion of Voronoi diagrams below.\n");
  printf(
"    -n  Outputs (to a .neigh file) a list of triangles neighboring each\n");
  printf("        triangle.\n");
  printf(
"    -g  Outputs the mesh to an Object File Format (.off) file, suitable for\n"
);
  printf("        viewing with the Geometry Center's Geomview package.\n");
  printf(
"    -B  No boundary markers in the output .node, .poly, and .edge output\n");
  printf(
"        files.  See the detailed discussion of boundary markers below.\n");
  printf(
"    -P  No output .poly file.  Saves disk space, but you lose the ability\n");
  printf(
"        to maintain constraining segments on later refinements of the mesh.\n"
);
  printf("    -N  No output .node file.\n");
  printf("    -E  No output .ele file.\n");
  printf(
"    -I  No iteration numbers.  Suppresses the output of .node and .poly\n");
  printf(
"        files, so your input files won't be overwritten.  (If your input is\n"
);
  printf(
"        a .poly file only, a .node file is written.)  Cannot be used with\n");
  printf(
"        the -r switch, because that would overwrite your input .ele file.\n");
  printf(
"        Shouldn't be used with the -q, -a, -u, or -s switch if you are\n");
  printf(
"        using a .node file for input, because no .node file is written, so\n"
);
  printf("        there is no record of any added Steiner points.\n");
  printf("    -O  No holes.  Ignores the holes in the .poly file.\n");
  printf(
"    -X  No exact arithmetic.  Normally, Triangle uses exact floating-point\n"
);
  printf(
"        arithmetic for certain tests if it thinks the inexact tests are not\n"
);
  printf(
"        accurate enough.  Exact arithmetic ensures the robustness of the\n");
  printf(
"        triangulation algorithms, despite floating-point roundoff error.\n");
  printf(
"        Disabling exact arithmetic with the -X switch causes a small\n");
  printf(
"        improvement in speed and creates the possibility that Triangle will\n"
);
  printf("        fail to produce a valid mesh.  Not recommended.\n");
  printf(
"    -z  Numbers all items starting from zero (rather than one).  Note that\n"
);
  printf(
"        this switch is normally overridden by the value used to number the\n"
);
  printf(
"        first vertex of the input .node or .poly file.  However, this\n");
  printf(
"        switch is useful when calling Triangle from another program.\n");
  printf(
"    -o2 Generates second-order subparametric elements with six nodes each.\n"
);
  printf(
"    -Y  No new vertices on the boundary.  This switch is useful when the\n");
  printf(
"        mesh boundary must be preserved so that it conforms to some\n");
  printf(
"        adjacent mesh.  Be forewarned that you will probably sacrifice much\n"
);
  printf(
"        of the quality of the mesh; Triangle will try, but the resulting\n");
  printf(
"        mesh may contain poorly shaped triangles.  Works well if all the\n");
  printf(
"        boundary vertices are closely spaced.  Specify this switch twice\n");
  printf(
"        (`-YY') to prevent all segment splitting, including internal\n");
  printf("        boundaries.\n");
  printf(
"    -S  Specifies the maximum number of Steiner points (vertices that are\n");
  printf(
"        not in the input, but are added to meet the constraints on minimum\n"
);
  printf(
"        angle and maximum area).  The default is to allow an unlimited\n");
  printf(
"        number.  If you specify this switch with no number after it,\n");
  printf(
"        the limit is set to zero.  Triangle always adds vertices at segment\n"
);
  printf(
"        intersections, even if it needs to use more vertices than the limit\n"
);
  printf(
"        you set.  When Triangle inserts segments by splitting (-s), it\n");
  printf(
"        always adds enough vertices to ensure that all the segments of the\n"
);
  printf("        PLSG are recovered, ignoring the limit if necessary.\n");
  printf(
"    -i  Uses an incremental rather than a divide-and-conquer algorithm to\n");
  printf(
"        construct a Delaunay triangulation.  Try it if the divide-and-\n");
  printf("        conquer algorithm fails.\n");
  printf(
"    -F  Uses Steven Fortune's sweepline algorithm to construct a Delaunay\n");
  printf(
"        triangulation.  Warning:  does not use exact arithmetic for all\n");
  printf("        calculations.  An exact result is not guaranteed.\n");
  printf(
"    -l  Uses only vertical cuts in the divide-and-conquer algorithm.  By\n");
  printf(
"        default, Triangle alternates between vertical and horizontal cuts,\n"
);
  printf(
"        which usually improve the speed except with vertex sets that are\n");
  printf(
"        small or short and wide.  This switch is primarily of theoretical\n");
  printf("        interest.\n");
  printf(
"    -s  Specifies that segments should be forced into the triangulation by\n"
);
  printf(
"        recursively splitting them at their midpoints, rather than by\n");
  printf(
"        generating a constrained Delaunay triangulation.  Segment splitting\n"
);
  printf(
"        is true to Ruppert's original algorithm, but can create needlessly\n"
);
  printf(
"        small triangles.  This switch is primarily of theoretical interest.\n"
);
  printf(
"    -C  Check the consistency of the final mesh.  Uses exact arithmetic for\n"
);
  printf(
"        checking, even if the -X switch is used.  Useful if you suspect\n");
  printf("        Triangle is buggy.\n");
  printf(
"    -Q  Quiet:  Suppresses all explanation of what Triangle is doing,\n");
  printf("        unless an error occurs.\n");
  printf(
"    -V  Verbose:  Gives detailed information about what Triangle is doing.\n"
);
  printf(
"        Add more `V's for increasing amount of detail.  `-V' is most\n");
  printf(
"        useful; itgives information on algorithmic progress and much more\n");
  printf(
"        detailed statistics.  `-VV' gives vertex-by-vertex details, and\n");
  printf(
"        prints so much that Triangle runs much more slowly.  `-VVVV' gives\n"
);
  printf("        information only a debugger could love.\n");
  printf("    -h  Help:  Displays these instructions.\n");
  printf("\n");
  printf("Definitions:\n");
  printf("\n");
  printf(
"  A Delaunay triangulation of a vertex set is a triangulation whose\n");
  printf(
"  vertices are the vertex set, that covers the convex hull of the vertex\n");
  printf(
"  set.  A Delaunay triangulation has the property that no vertex lies\n");
  printf(
"  inside the circumscribing circle (circle that passes through all three\n");
  printf("  vertices) of any triangle in the triangulation.\n\n");
  printf(
"  A Voronoi diagram of a vertex set is a subdivision of the plane into\n");
  printf(
"  polygonal cells (some of which may be unbounded, meaning infinitely\n");
  printf(
"  large), where each cell is the set of points in the plane that are closer\n"
);
  printf(
"  to some input vertex than to any other input vertex.  The Voronoi diagram\n"
);
  printf("  is a geometric dual of the Delaunay triangulation.\n\n");
  printf(
"  A Planar Straight Line Graph (PSLG) is a set of vertices and segments.\n");
  printf(
"  Segments are simply edges, whose endpoints are all vertices in the PSLG.\n"
);
  printf(
"  Segments may intersect each other only at their endpoints.  The file\n");
  printf("  format for PSLGs (.poly files) is described below.\n\n");
  printf(
"  A constrained Delaunay triangulation (CDT) of a PSLG is similar to a\n");
  printf(
"  Delaunay triangulation, but each PSLG segment is present as a single edge\n"
);
  printf(
"  of the CDT.  (A constrained Delaunay triangulation is not truly a\n");
  printf(
"  Delaunay triangulation, because some of its triangles might not be\n");
  printf(
"  Delaunay.)  By definition, a CDT does not have any vertices other than\n");
  printf(
"  those specified in the input PSLG.  Depending on context, a CDT might\n");
  printf(
"  cover the convex hull of the PSLG, or it might cover only a segment-\n");
  printf("  bounded region (e.g. a polygon).\n\n");
  printf(
"  A conforming Delaunay triangulation of a PSLG is a triangulation in which\n"
);
  printf(
"  each triangle is truly Delaunay, and each PSLG segment is represented by\n"
);
  printf(
"  a linear contiguous sequence of edges of the triangulation.  New vertices\n"
);
  printf(
"  (not part of the PSLG) may appear, and each input segment may have been\n");
  printf(
"  subdivided into shorter edges (subsegments) by these additional vertices.\n"
);
  printf(
"  The new vertices are frequently necessary to maintain the Delaunay\n");
  printf("  property while ensuring that every segment is represented.\n\n");
  printf(
"  A conforming constrained Delaunay triangulation (CCDT) of a PSLG is a\n");
  printf(
"  triangulation of a PSLG whose triangles are constrained Delaunay.  New\n");
  printf("  vertices may appear, and input segments may be subdivided into\n");
  printf(
"  subsegments, but not to guarantee that segments are respected; rather, to\n"
);
  printf(
"  improve the quality of the triangles.  The high-quality meshes produced\n");
  printf(
"  by the -q switch are usually CCDTs, but can be made conforming Delaunay\n");
  printf("  with the -D switch.\n\n");
  printf("File Formats:\n\n");
  printf(
"  All files may contain comments prefixed by the character '#'.  Vertices,\n"
);
  printf(
"  triangles, edges, holes, and maximum area constraints must be numbered\n");
  printf(
"  consecutively, starting from either 1 or 0.  Whichever you choose, all\n");
  printf(
"  input files must be consistent; if the vertices are numbered from 1, so\n");
  printf(
"  must be all other objects.  Triangle automatically detects your choice\n");
  printf(
"  while reading the .node (or .poly) file.  (When calling Triangle from\n");
  printf(
"  another program, use the -z switch if you wish to number objects from\n");
  printf("  zero.)  Examples of these file formats are given below.\n\n");
  printf("  .node files:\n");
  printf(
"    First line:  <# of vertices> <dimension (must be 2)> <# of attributes>\n"
);
  printf(
"                                           <# of boundary markers (0 or 1)>\n"
);
  printf(
"    Remaining lines:  <vertex #> <x> <y> [attributes] [boundary marker]\n");
  printf("\n");
  printf(
"    The attributes, which are typically floating-point values of physical\n");
  printf(
"    quantities (such as mass or conductivity) associated with the nodes of\n"
);
  printf(
"    a finite element mesh, are copied unchanged to the output mesh.  If -q,\n"
);
  printf(
"    -a, -u, -D, or -s is selected, each new Steiner point added to the mesh\n"
);
  printf("    has attributes assigned to it by linear interpolation.\n\n");
  printf(
"    If the fourth entry of the first line is `1', the last column of the\n");
  printf(
"    remainder of the file is assumed to contain boundary markers.  Boundary\n"
);
  printf(
"    markers are used to identify boundary vertices and vertices resting on\n"
);
  printf(
"    PSLG segments; a complete description appears in a section below.  The\n"
);
  printf(
"    .node file produced by Triangle contains boundary markers in the last\n");
  printf("    column unless they are suppressed by the -B switch.\n\n");
  printf("  .ele files:\n");
  printf(
"    First line:  <# of triangles> <nodes per triangle> <# of attributes>\n");
  printf(
"    Remaining lines:  <triangle #> <node> <node> <node> ... [attributes]\n");
  printf("\n");
  printf(
"    Nodes are indices into the corresponding .node file.  The first three\n");
  printf(
"    nodes are the corner vertices, and are listed in counterclockwise order\n"
);
  printf(
"    around each triangle.  (The remaining nodes, if any, depend on the type\n"
);
  printf("    of finite element used.)\n\n");
  printf(
"    The attributes are just like those of .node files.  Because there is no\n"
);
  printf(
"    simple mapping from input to output triangles, Triangle attempts to\n");
  printf(
"    interpolate attributes, and may cause a lot of diffusion of attributes\n"
);
  printf(
"    among nearby triangles as the triangulation is refined.  Attributes do\n"
);
  printf("    not diffuse across segments, so attributes used to identify\n");
  printf("    segment-bounded regions remain intact.\n\n");
  printf(
"    In .ele files produced by Triangle, each triangular element has three\n");
  printf(
"    nodes (vertices) unless the -o2 switch is used, in which case\n");
  printf(
"    subparametric quadratic elements with six nodes each are generated.\n");
  printf(
"    The first three nodes are the corners in counterclockwise order, and\n");
  printf(
"    the fourth, fifth, and sixth nodes lie on the midpoints of the edges\n");
  printf(
"    opposite the first, second, and third vertices, respectively.\n");
  printf("\n");
  printf("  .poly files:\n");
  printf(
"    First line:  <# of vertices> <dimension (must be 2)> <# of attributes>\n"
);
  printf(
"                                           <# of boundary markers (0 or 1)>\n"
);
  printf(
"    Following lines:  <vertex #> <x> <y> [attributes] [boundary marker]\n");
  printf("    One line:  <# of segments> <# of boundary markers (0 or 1)>\n");
  printf(
"    Following lines:  <segment #> <endpoint> <endpoint> [boundary marker]\n");
  printf("    One line:  <# of holes>\n");
  printf("    Following lines:  <hole #> <x> <y>\n");
  printf(
"    Optional line:  <# of regional attributes and/or area constraints>\n");
  printf(
"    Optional following lines:  <region #> <x> <y> <attribute> <max area>\n");
  printf("\n");
  printf(
"    A .poly file represents a PSLG, as well as some additional information.\n"
);
  printf(
"    The first section lists all the vertices, and is identical to the\n");
  printf(
"    format of .node files.  <# of vertices> may be set to zero to indicate\n"
);
  printf(
"    that the vertices are listed in a separate .node file; .poly files\n");
  printf(
"    produced by Triangle always have this format.  A vertex set represented\n"
);
  printf(
"    this way has the advantage that it may easily be triangulated with or\n");
  printf(
"    without segments (depending on whether the -p switch is invoked).\n");
  printf("\n");
  printf(
"    The second section lists the segments.  Segments are edges whose\n");
  printf(
"    presence in the triangulation is enforced.  (Depending on the choice of\n"
);
  printf(
"    switches, segment might be subdivided into smaller edges).  Each\n");
  printf(
"    segment is specified by listing the indices of its two endpoints.  This\n"
);
  printf(
"    means that you must include its endpoints in the vertex list.  Each\n");
  printf("    segment, like each point, may have a boundary marker.\n\n");
  printf(
"    If -q, -a, -u, and -s are not selected, Triangle produces a constrained\n"
);
  printf(
"    Delaunay triangulation (CDT), in which each segment appears as a single\n"
);
  printf(
"    edge in the triangulation.  If -q, -a, -u, or -s is selected, Triangle\n"
);
  printf(
"    produces a conforming constrained Delaunay triangulation (CCDT), in\n");
  printf(
"    which segments may be subdivided into smaller edges.  If -D is\n");
  printf(
"    selected, Triangle produces a conforming Delaunay triangulation, so\n");
  printf(
"    that every triangle is Delaunay, and not just constrained Delaunay.\n");
  printf("\n");
  printf(
"    The third section lists holes (and concavities, if -c is selected) in\n");
  printf(
"    the triangulation.  Holes are specified by identifying a point inside\n");
  printf(
"    each hole.  After the triangulation is formed, Triangle creates holes\n");
  printf(
"    by eating triangles, spreading out from each hole point until its\n");
  printf(
"    progress is blocked by segments in the PSLG.  You must be careful to\n");
  printf(
"    enclose each hole in segments, or your whole triangulation might be\n");
  printf(
"    eaten away.  If the two triangles abutting a segment are eaten, the\n");
  printf(
"    segment itself is also eaten.  Do not place a hole directly on a\n");
  printf("    segment; if you do, Triangle chooses one side of the segment\n");
  printf("    arbitrarily.\n\n");
  printf(
"    The optional fourth section lists regional attributes (to be assigned\n");
  printf(
"    to all triangles in a region) and regional constraints on the maximum\n");
  printf(
"    triangle area.  Triangle reads this section only if the -A switch is\n");
  printf(
"    used or the -a switch is used without a number following it, and the -r\n"
);
  printf(
"    switch is not used.  Regional attributes and area constraints are\n");
  printf(
"    propagated in the same manner as holes:  you specify a point for each\n");
  printf(
"    attribute and/or constraint, and the attribute and/or constraint\n");
  printf(
"    affects the whole region (bounded by segments) containing the point.\n");
  printf(
"    If two values are written on a line after the x and y coordinate, the\n");
  printf(
"    first such value is assumed to be a regional attribute (but is only\n");
  printf(
"    applied if the -A switch is selected), and the second value is assumed\n"
);
  printf(
"    to be a regional area constraint (but is only applied if the -a switch\n"
);
  printf(
"    is selected).  You may specify just one value after the coordinates,\n");
  printf(
"    which can serve as both an attribute and an area constraint, depending\n"
);
  printf(
"    on the choice of switches.  If you are using the -A and -a switches\n");
  printf(
"    simultaneously and wish to assign an attribute to some region without\n");
  printf("    imposing an area constraint, use a negative maximum area.\n\n");
  printf(
"    When a triangulation is created from a .poly file, you must either\n");
  printf(
"    enclose the entire region to be triangulated in PSLG segments, or\n");
  printf(
"    use the -c switch, which automatically creates extra segments that\n");
  printf(
"    enclose the convex hull of the PSLG.  If you do not use the -c switch,\n"
);
  printf(
"    Triangle eats all triangles that are not enclosed by segments; if you\n");
  printf(
"    are not careful, your whole triangulation may be eaten away.  If you do\n"
);
  printf(
"    use the -c switch, you can still produce concavities by the appropriate\n"
);
  printf(
"    placement of holes just inside the boundary of the convex hull.\n");
  printf("\n");
  printf(
"    An ideal PSLG has no intersecting segments, nor any vertices that lie\n");
  printf(
"    upon segments (except, of course, the endpoints of each segment).  You\n"
);
  printf(
"    aren't required to make your .poly files ideal, but you should be aware\n"
);
  printf(
"    of what can go wrong.  Segment intersections are relatively safe--\n");
  printf(
"    Triangle calculates the intersection points for you and adds them to\n");
  printf(
"    the triangulation--as long as your machine's floating-point precision\n");
  printf(
"    doesn't become a problem.  You are tempting the fates if you have three\n"
);
  printf(
"    segments that cross at the same location, and expect Triangle to figure\n"
);
  printf(
"    out where the intersection point is.  Thanks to floating-point roundoff\n"
);
  printf(
"    error, Triangle will probably decide that the three segments intersect\n"
);
  printf(
"    at three different points, and you will find a minuscule triangle in\n");
  printf(
"    your output--unless Triangle tries to refine the tiny triangle, uses\n");
  printf(
"    up the last bit of machine precision, and fails to terminate at all.\n");
  printf(
"    You're better off putting the intersection point in the input files,\n");
  printf(
"    and manually breaking up each segment into two.  Similarly, if you\n");
  printf(
"    place a vertex at the middle of a segment, and hope that Triangle will\n"
);
  printf(
"    break up the segment at that vertex, you might get lucky.  On the other\n"
);
  printf(
"    hand, Triangle might decide that the vertex doesn't lie precisely on\n");
  printf(
"    the segment, and you'll have a needle-sharp triangle in your output--or\n"
);
  printf("    a lot of tiny triangles if you're generating a quality mesh.\n");
  printf("\n");
  printf(
"    When Triangle reads a .poly file, it also writes a .poly file, which\n");
  printf(
"    includes all the subsegments--the edges that are parts of input\n");
  printf(
"    segments.  If the -c switch is used, the output .poly file also\n");
  printf(
"    includes all of the edges on the convex hull.  Hence, the output .poly\n"
);
  printf(
"    file is useful for finding edges associated with input segments and for\n"
);
  printf(
"    setting boundary conditions in finite element simulations.  Moreover,\n");
  printf(
"    you will need the output .poly file if you plan to refine the output\n");
  printf(
"    mesh, and don't want segments to be missing in later triangulations.\n");
  printf("\n");
  printf("  .area files:\n");
  printf("    First line:  <# of triangles>\n");
  printf("    Following lines:  <triangle #> <maximum area>\n");
  printf("\n");
  printf(
"    An .area file associates with each triangle a maximum area that is used\n"
);
  printf(
"    for mesh refinement.  As with other file formats, every triangle must\n");
  printf(
"    be represented, and the triangles must be numbered consecutively.  A\n");
  printf(
"    triangle may be left unconstrained by assigning it a negative maximum\n");
  printf("    area.\n\n");
  printf("  .edge files:\n");
  printf("    First line:  <# of edges> <# of boundary markers (0 or 1)>\n");
  printf(
"    Following lines:  <edge #> <endpoint> <endpoint> [boundary marker]\n");
  printf("\n");
  printf(
"    Endpoints are indices into the corresponding .node file.  Triangle can\n"
);
  printf(
"    produce .edge files (use the -e switch), but cannot read them.  The\n");
  printf(
"    optional column of boundary markers is suppressed by the -B switch.\n");
  printf("\n");
  printf(
"    In Voronoi diagrams, one also finds a special kind of edge that is an\n");
  printf(
"    infinite ray with only one endpoint.  For these edges, a different\n");
  printf("    format is used:\n\n");
  printf("        <edge #> <endpoint> -1 <direction x> <direction y>\n\n");
  printf(
"    The `direction' is a floating-point vector that indicates the direction\n"
);
  printf("    of the infinite ray.\n\n");
  printf("  .neigh files:\n");
  printf(
"    First line:  <# of triangles> <# of neighbors per triangle (always 3)>\n"
);
  printf(
"    Following lines:  <triangle #> <neighbor> <neighbor> <neighbor>\n");
  printf("\n");
  printf(
"    Neighbors are indices into the corresponding .ele file.  An index of -1\n"
);
  printf(
"    indicates no neighbor (because the triangle is on an exterior\n");
  printf(
"    boundary).  The first neighbor of triangle i is opposite the first\n");
  printf("    corner of triangle i, and so on.\n\n");
  printf(
"    Triangle can produce .neigh files (use the -n switch), but cannot read\n"
);
  printf("    them.\n\n");
  printf("Boundary Markers:\n\n");
  printf(
"  Boundary markers are tags used mainly to identify which output vertices\n");
  printf(
"  and edges are associated with which PSLG segment, and to identify which\n");
  printf(
"  vertices and edges occur on a boundary of the triangulation.  A common\n");
  printf(
"  use is to determine where boundary conditions should be applied to a\n");
  printf(
"  finite element mesh.  You can prevent boundary markers from being written\n"
);
  printf("  into files produced by Triangle by using the -B switch.\n\n");
  printf(
"  The boundary marker associated with each segment in an output .poly file\n"
);
  printf("  and each edge in an output .edge file is chosen as follows:\n");
  printf(
"    - If an output edge is part or all of a PSLG segment with a nonzero\n");
  printf(
"      boundary marker, then the edge is assigned the same marker.\n");
  printf(
"    - Otherwise, if the edge lies on a boundary of the triangulation\n");
  printf(
"      (even the boundary of a hole), then the edge is assigned the marker\n");
  printf("      one (1).\n");
  printf("    - Otherwise, the edge is assigned the marker zero (0).\n");
  printf(
"  The boundary marker associated with each vertex in an output .node file\n");
  printf("  is chosen as follows:\n");
  printf(
"    - If a vertex is assigned a nonzero boundary marker in the input file,\n"
);
  printf(
"      then it is assigned the same marker in the output .node file.\n");
  printf(
"    - Otherwise, if the vertex lies on a PSLG segment (even if it is an\n");
  printf(
"      endpoint of the segment) with a nonzero boundary marker, then the\n");
  printf(
"      vertex is assigned the same marker.  If the vertex lies on several\n");
  printf("      such segments, one of the markers is chosen arbitrarily.\n");
  printf(
"    - Otherwise, if the vertex occurs on a boundary of the triangulation,\n");
  printf("      then the vertex is assigned the marker one (1).\n");
  printf("    - Otherwise, the vertex is assigned the marker zero (0).\n");
  printf("\n");
  printf(
"  If you want Triangle to determine for you which vertices and edges are on\n"
);
  printf(
"  the boundary, assign them the boundary marker zero (or use no markers at\n"
);
  printf(
"  all) in your input files.  In the output files, all boundary vertices,\n");
  printf("  edges, and segments will be assigned the value one.\n\n");
  printf("Triangulation Iteration Numbers:\n\n");
  printf(
"  Because Triangle can read and refine its own triangulations, input\n");
  printf(
"  and output files have iteration numbers.  For instance, Triangle might\n");
  printf(
"  read the files mesh.3.node, mesh.3.ele, and mesh.3.poly, refine the\n");
  printf(
"  triangulation, and output the files mesh.4.node, mesh.4.ele, and\n");
  printf("  mesh.4.poly.  Files with no iteration number are treated as if\n");
  printf(
"  their iteration number is zero; hence, Triangle might read the file\n");
  printf(
"  points.node, triangulate it, and produce the files points.1.node and\n");
  printf("  points.1.ele.\n\n");
  printf(
"  Iteration numbers allow you to create a sequence of successively finer\n");
  printf(
"  meshes suitable for multigrid methods.  They also allow you to produce a\n"
);
  printf(
"  sequence of meshes using error estimate-driven mesh refinement.\n");
  printf("\n");
  printf(
"  If you're not using refinement or quality meshing, and you don't like\n");
  printf(
"  iteration numbers, use the -I switch to disable them.  This switch also\n");
  printf(
"  disables output of .node and .poly files to prevent your input files from\n"
);
  printf(
"  being overwritten.  (If the input is a .poly file that contains its own\n");
  printf(
"  points, a .node file is written.  This can be quite convenient for\n");
  printf("  computing CDTs or quality meshes.)\n\n");
  printf("Examples of How to Use Triangle:\n\n");
  printf(
"  `triangle dots' reads vertices from dots.node, and writes their Delaunay\n"
);
  printf(
"  triangulation to dots.1.node and dots.1.ele.  (dots.1.node is identical\n");
  printf(
"  to dots.node.)  `triangle -I dots' writes the triangulation to dots.ele\n");
  printf(
"  instead.  (No additional .node file is needed, so none is written.)\n");
  printf("\n");
  printf(
"  `triangle -pe object.1' reads a PSLG from object.1.poly (and possibly\n");
  printf(
"  object.1.node, if the vertices are omitted from object.1.poly) and writes\n"
);
  printf(
"  its constrained Delaunay triangulation to object.2.node and object.2.ele.\n"
);
  printf(
"  The segments are copied to object.2.poly, and all edges are written to\n");
  printf("  object.2.edge.\n\n");
  printf(
"  `triangle -pq31.5a.1 object' reads a PSLG from object.poly (and possibly\n"
);
  printf(
"  object.node), generates a mesh whose angles are all between 31.5 and 117\n"
);
  printf(
"  degrees and whose triangles all have areas of 0.1 or less, and writes the\n"
);
  printf(
"  mesh to object.1.node and object.1.ele.  Each segment may be broken up\n");
  printf("  into multiple subsegments; these are written to object.1.poly.\n");
  printf("\n");
  printf(
"  Here is a sample file `box.poly' describing a square with a square hole:\n"
);
  printf("\n");
  printf(
"    # A box with eight vertices in 2D, no attributes, one boundary marker.\n"
);
  printf("    8 2 0 1\n");
  printf("     # Outer box has these vertices:\n");
  printf("     1   0 0   0\n");
  printf("     2   0 3   0\n");
  printf("     3   3 0   0\n");
  printf("     4   3 3   33     # A special marker for this vertex.\n");
  printf("     # Inner square has these vertices:\n");
  printf("     5   1 1   0\n");
  printf("     6   1 2   0\n");
  printf("     7   2 1   0\n");
  printf("     8   2 2   0\n");
  printf("    # Five segments with boundary markers.\n");
  printf("    5 1\n");
  printf("     1   1 2   5      # Left side of outer box.\n");
  printf("     # Square hole has these segments:\n");
  printf("     2   5 7   0\n");
  printf("     3   7 8   0\n");
  printf("     4   8 6   10\n");
  printf("     5   6 5   0\n");
  printf("    # One hole in the middle of the inner square.\n");
  printf("    1\n");
  printf("     1   1.5 1.5\n");
  printf("\n");
  printf(
"  Note that some segments are missing from the outer square, so you must\n");
  printf(
"  use the `-c' switch.  After `triangle -pqc box.poly', here is the output\n"
);
  printf(
"  file `box.1.node', with twelve vertices.  The last four vertices were\n");
  printf(
"  added to meet the angle constraint.  Vertices 1, 2, and 9 have markers\n");
  printf(
"  from segment 1.  Vertices 6 and 8 have markers from segment 4.  All the\n");
  printf(
"  other vertices but 4 have been marked to indicate that they lie on a\n");
  printf("  boundary.\n\n");
  printf("    12  2  0  1\n");
  printf("       1    0   0      5\n");
  printf("       2    0   3      5\n");
  printf("       3    3   0      1\n");
  printf("       4    3   3     33\n");
  printf("       5    1   1      1\n");
  printf("       6    1   2     10\n");
  printf("       7    2   1      1\n");
  printf("       8    2   2     10\n");
  printf("       9    0   1.5    5\n");
  printf("      10    1.5   0    1\n");
  printf("      11    3   1.5    1\n");
  printf("      12    1.5   3    1\n");
  printf("    # Generated by triangle -pqc box.poly\n");
  printf("\n");
  printf("  Here is the output file `box.1.ele', with twelve triangles.\n");
  printf("\n");
  printf("    12  3  0\n");
  printf("       1     5   6   9\n");
  printf("       2    10   3   7\n");
  printf("       3     6   8  12\n");
  printf("       4     9   1   5\n");
  printf("       5     6   2   9\n");
  printf("       6     7   3  11\n");
  printf("       7    11   4   8\n");
  printf("       8     7   5  10\n");
  printf("       9    12   2   6\n");
  printf("      10     8   7  11\n");
  printf("      11     5   1  10\n");
  printf("      12     8   4  12\n");
  printf("    # Generated by triangle -pqc box.poly\n\n");
  printf(
"  Here is the output file `box.1.poly'.  Note that segments have been added\n"
);
  printf(
"  to represent the convex hull, and some segments have been subdivided by\n");
  printf(
"  newly added vertices.  Note also that <# of vertices> is set to zero to\n");
  printf("  indicate that the vertices should be read from the .node file.\n");
  printf("\n");
  printf("    0  2  0  1\n");
  printf("    12  1\n");
  printf("       1     1   9     5\n");
  printf("       2     5   7     1\n");
  printf("       3     8   7     1\n");
  printf("       4     6   8    10\n");
  printf("       5     5   6     1\n");
  printf("       6     3  10     1\n");
  printf("       7     4  11     1\n");
  printf("       8     2  12     1\n");
  printf("       9     9   2     5\n");
  printf("      10    10   1     1\n");
  printf("      11    11   3     1\n");
  printf("      12    12   4     1\n");
  printf("    1\n");
  printf("       1   1.5 1.5\n");
  printf("    # Generated by triangle -pqc box.poly\n");
  printf("\n");
  printf("Refinement and Area Constraints:\n");
  printf("\n");
  printf(
"  The -r switch causes a mesh (.node and .ele files) to be read and\n");
  printf(
"  refined.  If the -p switch is also used, a .poly file is read and used to\n"
);
  printf(
"  specify edges that are constrained and cannot be eliminated (although\n");
  printf(
"  they can be subdivided into smaller edges) by the refinement process.\n");
  printf("\n");
  printf(
"  When you refine a mesh, you generally want to impose tighter constraints.\n"
);
  printf(
"  One way to accomplish this is to use -q with a larger angle, or -a\n");
  printf(
"  followed by a smaller area than you used to generate the mesh you are\n");
  printf(
"  refining.  Another way to do this is to create an .area file, which\n");
  printf(
"  specifies a maximum area for each triangle, and use the -a switch\n");
  printf(
"  (without a number following).  Each triangle's area constraint is applied\n"
);
  printf(
"  to that triangle.  Area constraints tend to diffuse as the mesh is\n");
  printf(
"  refined, so if there are large variations in area constraint between\n");
  printf(
"  adjacent triangles, you may not get the results you want.  In that case,\n"
);
  printf(
"  consider instead using the -u switch and writing a C procedure that\n");
  printf("  determines which triangles are too large.\n\n");
  printf(
"  If you are refining a mesh composed of linear (three-node) elements, the\n"
);
  printf(
"  output mesh contains all the nodes present in the input mesh, in the same\n"
);
  printf(
"  order, with new nodes added at the end of the .node file.  However, the\n");
  printf(
"  refinement is not hierarchical: there is no guarantee that each output\n");
  printf(
"  element is contained in a single input element.  Often, an output element\n"
);
  printf(
"  can overlap two or three input elements, and some input edges are not\n");
  printf(
"  present in the output mesh.  Hence, a sequence of refined meshes forms a\n"
);
  printf(
"  hierarchy of nodes, but not a hierarchy of elements.  If you refine a\n");
  printf(
"  mesh of higher-order elements, the hierarchical property applies only to\n"
);
  printf(
"  the nodes at the corners of an element; the midpoint nodes on each edge\n");
  printf("  are discarded before the mesh is refined.\n\n");
  printf(
"  Maximum area constraints in .poly files operate differently from those in\n"
);
  printf(
"  .area files.  A maximum area in a .poly file applies to the whole\n");
  printf(
"  (segment-bounded) region in which a point falls, whereas a maximum area\n");
  printf(
"  in an .area file applies to only one triangle.  Area constraints in .poly\n"
);
  printf(
"  files are used only when a mesh is first generated, whereas area\n");
  printf(
"  constraints in .area files are used only to refine an existing mesh, and\n"
);
  printf(
"  are typically based on a posteriori error estimates resulting from a\n");
  printf("  finite element simulation on that mesh.\n\n");
  printf(
"  `triangle -rq25 object.1' reads object.1.node and object.1.ele, then\n");
  printf(
"  refines the triangulation to enforce a 25 degree minimum angle, and then\n"
);
  printf(
"  writes the refined triangulation to object.2.node and object.2.ele.\n");
  printf("\n");
  printf(
"  `triangle -rpaa6.2 z.3' reads z.3.node, z.3.ele, z.3.poly, and z.3.area.\n"
);
  printf(
"  After reconstructing the mesh and its subsegments, Triangle refines the\n");
  printf(
"  mesh so that no triangle has area greater than 6.2, and furthermore the\n");
  printf(
"  triangles satisfy the maximum area constraints in z.3.area.  No angle\n");
  printf(
"  bound is imposed at all.  The output is written to z.4.node, z.4.ele, and\n"
);
  printf("  z.4.poly.\n\n");
  printf(
"  The sequence `triangle -qa1 x', `triangle -rqa.3 x.1', `triangle -rqa.1\n");
  printf(
"  x.2' creates a sequence of successively finer meshes x.1, x.2, and x.3,\n");
  printf("  suitable for multigrid.\n\n");
  printf("Convex Hulls and Mesh Boundaries:\n\n");
  printf(
"  If the input is a vertex set (not a PSLG), Triangle produces its convex\n");
  printf(
"  hull as a by-product in the output .poly file if you use the -c switch.\n");
  printf(
"  There are faster algorithms for finding a two-dimensional convex hull\n");
  printf("  than triangulation, of course, but this one comes for free.\n\n");
  printf(
"  If the input is an unconstrained mesh (you are using the -r switch but\n");
  printf(
"  not the -p switch), Triangle produces a list of its boundary edges\n");
  printf(
"  (including hole boundaries) as a by-product when you use the -c switch.\n");
  printf(
"  If you also use the -p switch, the output .poly file contains all the\n");
  printf("  segments from the input .poly file as well.\n\n");
  printf("Voronoi Diagrams:\n\n");
  printf(
"  The -v switch produces a Voronoi diagram, in files suffixed .v.node and\n");
  printf(
"  .v.edge.  For example, `triangle -v points' reads points.node, produces\n");
  printf(
"  its Delaunay triangulation in points.1.node and points.1.ele, and\n");
  printf(
"  produces its Voronoi diagram in points.1.v.node and points.1.v.edge.  The\n"
);
  printf(
"  .v.node file contains a list of all Voronoi vertices, and the .v.edge\n");
  printf(
"  file contains a list of all Voronoi edges, some of which may be infinite\n"
);
  printf(
"  rays.  (The choice of filenames makes it easy to run the set of Voronoi\n");
  printf("  vertices through Triangle, if so desired.)\n\n");
  printf(
"  This implementation does not use exact arithmetic to compute the Voronoi\n"
);
  printf(
"  vertices, and does not check whether neighboring vertices are identical.\n"
);
  printf(
"  Be forewarned that if the Delaunay triangulation is degenerate or\n");
  printf(
"  near-degenerate, the Voronoi diagram may have duplicate vertices or\n");
  printf("  crossing edges.\n\n");
  printf(
"  The result is a valid Voronoi diagram only if Triangle's output is a true\n"
);
  printf(
"  Delaunay triangulation.  The Voronoi output is usually meaningless (and\n");
  printf(
"  may contain crossing edges and other pathology) if the output is a CDT or\n"
);
  printf(
"  CCDT, or if it has holes or concavities.  If the triangulated domain is\n");
  printf(
"  convex and has no holes, you can use -D switch to force Triangle to\n");
  printf(
"  construct a conforming Delaunay triangulation instead of a CCDT, so the\n");
  printf("  Voronoi diagram will be valid.\n\n");
  printf("Mesh Topology:\n\n");
  printf(
"  You may wish to know which triangles are adjacent to a certain Delaunay\n");
  printf(
"  edge in an .edge file, which Voronoi cells are adjacent to a certain\n");
  printf(
"  Voronoi edge in a .v.edge file, or which Voronoi cells are adjacent to\n");
  printf(
"  each other.  All of this information can be found by cross-referencing\n");
  printf(
"  output files with the recollection that the Delaunay triangulation and\n");
  printf("  the Voronoi diagram are planar duals.\n\n");
  printf(
"  Specifically, edge i of an .edge file is the dual of Voronoi edge i of\n");
  printf(
"  the corresponding .v.edge file, and is rotated 90 degrees counterclock-\n");
  printf(
"  wise from the Voronoi edge.  Triangle j of an .ele file is the dual of\n");
  printf(
"  vertex j of the corresponding .v.node file.  Voronoi cell k is the dual\n");
  printf("  of vertex k of the corresponding .node file.\n\n");
  printf(
"  Hence, to find the triangles adjacent to a Delaunay edge, look at the\n");
  printf(
"  vertices of the corresponding Voronoi edge.  If the endpoints of a\n");
  printf(
"  Voronoi edge are Voronoi vertices 2 and 6 respectively, then triangles 2\n"
);
  printf(
"  and 6 adjoin the left and right sides of the corresponding Delaunay edge,\n"
);
  printf(
"  respectively.  To find the Voronoi cells adjacent to a Voronoi edge, look\n"
);
  printf(
"  at the endpoints of the corresponding Delaunay edge.  If the endpoints of\n"
);
  printf(
"  a Delaunay edge are input vertices 7 and 12, then Voronoi cells 7 and 12\n"
);
  printf(
"  adjoin the right and left sides of the corresponding Voronoi edge,\n");
  printf(
"  respectively.  To find which Voronoi cells are adjacent to each other,\n");
  printf("  just read the list of Delaunay edges.\n\n");
  printf(
"  Triangle does not write a list of the edges adjoining each Voronoi cell,\n"
);
  printf(
"  but you can reconstructed it straightforwardly.  For instance, to find\n");
  printf(
"  all the edges of Voronoi cell 1, search the output .edge file for every\n");
  printf(
"  edge that has input vertex 1 as an endpoint.  The corresponding dual\n");
  printf(
"  edges in the output .v.edge file form the boundary of Voronoi cell 1.\n");
  printf("\n");
  printf(
"  For each Voronoi vertex, the .neigh file gives a list of the three\n");
  printf(
"  Voronoi vertices attached to it.  You might find this more convenient\n");
  printf("  than the .v.edge file.\n\n");
  printf("Quadratic Elements:\n\n");
  printf(
"  Triangle generates meshes with subparametric quadratic elements if the\n");
  printf(
"  -o2 switch is specified.  Quadratic elements have six nodes per element,\n"
);
  printf(
"  rather than three.  `Subparametric' means that the edges of the triangles\n"
);
  printf(
"  are always straight, so that subparametric quadratic elements are\n");
  printf(
"  geometrically identical to linear elements, even though they can be used\n"
);
  printf(
"  with quadratic interpolating functions.  The three extra nodes of an\n");
  printf(
"  element fall at the midpoints of the three edges, with the fourth, fifth,\n"
);
  printf(
"  and sixth nodes appearing opposite the first, second, and third corners\n");
  printf("  respectively.\n\n");
  printf("Domains with Small Angles:\n\n");
  printf(
"  If two input segments adjoin each other at a small angle, clearly the -q\n"
);
  printf(
"  switch cannot remove the small angle.  Moreover, Triangle may have no\n");
  printf(
"  choice but to generate additional triangles whose smallest angles are\n");
  printf(
"  smaller than the specified bound.  However, these triangles only appear\n");
  printf(
"  between input segments separated by small angles.  Moreover, if you\n");
  printf(
"  request a minimum angle of theta degrees, Triangle will generally produce\n"
);
  printf(
"  no angle larger than 180 - 2 theta, even if it is forced to compromise on\n"
);
  printf("  the minimum angle.\n\n");
  printf("Statistics:\n\n");
  printf(
"  After generating a mesh, Triangle prints a count of entities in the\n");
  printf(
"  output mesh, including the number of vertices, triangles, edges, exterior\n"
);
  printf(
"  boundary edges (i.e. subsegments on the boundary of the triangulation,\n");
  printf(
"  including hole boundaries), interior boundary edges (i.e. subsegments of\n"
);
  printf(
"  input segments not on the boundary), and total subsegments.  If you've\n");
  printf(
"  forgotten the statistics for an existing mesh, run Triangle on that mesh\n"
);
  printf(
"  with the -rNEP switches to read the mesh and print the statistics without\n"
);
  printf(
"  writing any files.  Use -rpNEP if you've got a .poly file for the mesh.\n");
  printf("\n");
  printf(
"  The -V switch produces extended statistics, including a rough estimate\n");
  printf(
"  of memory use, the number of calls to geometric predicates, and\n");
  printf(
"  histograms of the angles and the aspect ratios of the triangles in the\n");
  printf("  mesh.\n\n");
  printf("Exact Arithmetic:\n\n");
  printf(
"  Triangle uses adaptive exact arithmetic to perform what computational\n");
  printf(
"  geometers call the `orientation' and `incircle' tests.  If the floating-\n"
);
  printf(
"  point arithmetic of your machine conforms to the IEEE 754 standard (as\n");
  printf(
"  most workstations do), and does not use extended precision internal\n");
  printf(
"  floating-point registers, then your output is guaranteed to be an\n");
  printf(
"  absolutely true Delaunay or constrained Delaunay triangulation, roundoff\n"
);
  printf(
"  error notwithstanding.  The word `adaptive' implies that these arithmetic\n"
);
  printf(
"  routines compute the result only to the precision necessary to guarantee\n"
);
  printf(
"  correctness, so they are usually nearly as fast as their approximate\n");
  printf("  counterparts.\n\n");
  printf(
"  May CPUs, including Intel x86 processors, have extended precision\n");
  printf(
"  floating-point registers.  These must be reconfigured so their precision\n"
);
  printf(
"  is reduced to memory precision.  Triangle does this if it is compiled\n");
  printf("  correctly.  See the makefile for details.\n\n");
  printf(
"  The exact tests can be disabled with the -X switch.  On most inputs, this\n"
);
  printf(
"  switch reduces the computation time by about eight percent--it's not\n");
  printf(
"  worth the risk.  There are rare difficult inputs (having many collinear\n");
  printf(
"  and cocircular vertices), however, for which the difference in speed\n");
  printf(
"  could be a factor of two.  Be forewarned that these are precisely the\n");
  printf(
"  inputs most likely to cause errors if you use the -X switch.  Hence, the\n"
);
  printf("  -X switch is not recommended.\n\n");
  printf(
"  Unfortunately, the exact tests don't solve every numerical problem.\n");
  printf(
"  Exact arithmetic is not used to compute the positions of new vertices,\n");
  printf(
"  because the bit complexity of vertex coordinates would grow without\n");
  printf(
"  bound.  Hence, segment intersections aren't computed exactly; in very\n");
  printf(
"  unusual cases, roundoff error in computing an intersection point might\n");
  printf(
"  actually lead to an inverted triangle and an invalid triangulation.\n");
  printf(
"  (This is one reason to specify your own intersection points in your .poly\n"
);
  printf(
"  files.)  Similarly, exact arithmetic is not used to compute the vertices\n"
);
  printf("  of the Voronoi diagram.\n\n");
  printf(
"  Another pair of problems not solved by the exact arithmetic routines is\n");
  printf(
"  underflow and overflow.  If Triangle is compiled for double precision\n");
  printf(
"  arithmetic, I believe that Triangle's geometric predicates work correctly\n"
);
  printf(
"  if the exponent of every input coordinate falls in the range [-148, 201].\n"
);
  printf(
"  Underflow can silently prevent the orientation and incircle tests from\n");
  printf(
"  being performed exactly, while overflow typically causes a floating\n");
  printf("  exception.\n\n");
  printf("Calling Triangle from Another Program:\n\n");
  printf("  Read the file triangle.h for details.\n\n");
  printf("Troubleshooting:\n\n");
  printf("  Please read this section before mailing me bugs.\n\n");
  printf("  `My output mesh has no triangles!'\n\n");
  printf(
"    If you're using a PSLG, you've probably failed to specify a proper set\n"
);
  printf(
"    of bounding segments, or forgotten to use the -c switch.  Or you may\n");
  printf(
"    have placed a hole badly, thereby eating all your triangles.  To test\n");
  printf("    these possibilities, try again with the -c and -O switches.\n");
  printf(
"    Alternatively, all your input vertices may be collinear, in which case\n"
);
  printf("    you can hardly expect to triangulate them.\n\n");
  printf("  `Triangle doesn't terminate, or just crashes.'\n\n");
  printf(
"    Bad things can happen when triangles get so small that the distance\n");
  printf(
"    between their vertices isn't much larger than the precision of your\n");
  printf(
"    machine's arithmetic.  If you've compiled Triangle for single-precision\n"
);
  printf(
"    arithmetic, you might do better by recompiling it for double-precision.\n"
);
  printf(
"    Then again, you might just have to settle for more lenient constraints\n"
);
  printf(
"    on the minimum angle and the maximum area than you had planned.\n");
  printf("\n");
  printf(
"    You can minimize precision problems by ensuring that the origin lies\n");
  printf(
"    inside your vertex set, or even inside the densest part of your\n");
  printf(
"    mesh.  If you're triangulating an object whose x-coordinates all fall\n");
  printf(
"    between 6247133 and 6247134, you're not leaving much floating-point\n");
  printf("    precision for Triangle to work with.\n\n");
  printf(
"    Precision problems can occur covertly if the input PSLG contains two\n");
  printf(
"    segments that meet (or intersect) at an extremely small angle, or if\n");
  printf(
"    such an angle is introduced by the -c switch.  If you don't realize\n");
  printf(
"    that a tiny angle is being formed, you might never discover why\n");
  printf(
"    Triangle is crashing.  To check for this possibility, use the -S switch\n"
);
  printf(
"    (with an appropriate limit on the number of Steiner points, found by\n");
  printf(
"    trial-and-error) to stop Triangle early, and view the output .poly file\n"
);
  printf(
"    with Show Me (described below).  Look carefully for regions where dense\n"
);
  printf(
"    clusters of vertices are forming and for small angles between segments.\n"
);
  printf(
"    Zoom in closely, as such segments might look like a single segment from\n"
);
  printf("    a distance.\n\n");
  printf(
"    If some of the input values are too large, Triangle may suffer a\n");
  printf(
"    floating exception due to overflow when attempting to perform an\n");
  printf(
"    orientation or incircle test.  (Read the section on exact arithmetic\n");
  printf(
"    above.)  Again, I recommend compiling Triangle for double (rather\n");
  printf("    than single) precision arithmetic.\n\n");
  printf(
"    Unexpected problems can arise if you use quality meshing (-q, -a, or\n");
  printf(
"    -u) with an input that is not segment-bounded--that is, if your input\n");
  printf(
"    is a vertex set, or you're using the -c switch.  If the convex hull of\n"
);
  printf(
"    your input vertices has collinear vertices on its boundary, an input\n");
  printf(
"    vertex that you think lies on the convex hull might actually lie just\n");
  printf(
"    inside the convex hull.  If so, the vertex and the nearby convex hull\n");
  printf(
"    edge form an extremely thin triangle.  When Triangle tries to refine\n");
  printf(
"    the mesh to enforce angle and area constraints, Triangle might generate\n"
);
  printf(
"    extremely tiny triangles, or it might fail because of insufficient\n");
  printf("    floating-point precision.\n\n");
  printf(
"  `The numbering of the output vertices doesn't match the input vertices.'\n"
);
  printf("\n");
  printf(
"    You may have had duplicate input vertices, or you may have eaten some\n");
  printf(
"    of your input vertices with a hole, or by placing them outside the area\n"
);
  printf(
"    enclosed by segments.  In any case, you can solve the problem by not\n");
  printf("    using the -j switch.\n\n");
  printf(
"  `Triangle executes without incident, but when I look at the resulting\n");
  printf(
"  mesh, it has overlapping triangles or other geometric inconsistencies.'\n");
  printf("\n");
  printf(
"    If you select the -X switch, Triangle occasionally makes mistakes due\n");
  printf(
"    to floating-point roundoff error.  Although these errors are rare,\n");
  printf(
"    don't use the -X switch.  If you still have problems, please report the\n"
);
  printf("    bug.\n\n");
  printf(
"  `Triangle executes without incident, but when I look at the resulting\n");
  printf("  Voronoi diagram, it has overlapping edges or other geometric\n");
  printf("  inconsistencies.'\n");
  printf("\n");
  printf(
"    If your input is a PSLG (-p), you can only expect a meaningful Voronoi\n"
);
  printf(
"    diagram if the domain you are triangulating is convex and free of\n");
  printf(
"    holes, and you use the -D switch to construct a conforming Delaunay\n");
  printf("    triangulation (instead of a CDT or CCDT).\n\n");
  printf(
"  Strange things can happen if you've taken liberties with your PSLG.  Do\n");
  printf(
"  you have a vertex lying in the middle of a segment?  Triangle sometimes\n");
  printf(
"  copes poorly with that sort of thing.  Do you want to lay out a collinear\n"
);
  printf(
"  row of evenly spaced, segment-connected vertices?  Have you simply\n");
  printf(
"  defined one long segment connecting the leftmost vertex to the rightmost\n"
);
  printf(
"  vertex, and a bunch of vertices lying along it?  This method occasionally\n"
);
  printf(
"  works, especially with horizontal and vertical lines, but often it\n");
  printf(
"  doesn't, and you'll have to connect each adjacent pair of vertices with a\n"
);
  printf("  separate segment.  If you don't like it, tough.\n\n");
  printf(
"  Furthermore, if you have segments that intersect other than at their\n");
  printf(
"  endpoints, try not to let the intersections fall extremely close to PSLG\n"
);
  printf("  vertices or each other.\n\n");
  printf(
"  If you have problems refining a triangulation not produced by Triangle:\n");
  printf(
"  Are you sure the triangulation is geometrically valid?  Is it formatted\n");
  printf(
"  correctly for Triangle?  Are the triangles all listed so the first three\n"
);
  printf(
"  vertices are their corners in counterclockwise order?  Are all of the\n");
  printf(
"  triangles constrained Delaunay?  Triangle's Delaunay refinement algorithm\n"
);
  printf("  assumes that it starts with a CDT.\n\n");
  printf("Show Me:\n\n");
  printf(
"  Triangle comes with a separate program named `Show Me', whose primary\n");
  printf(
"  purpose is to draw meshes on your screen or in PostScript.  Its secondary\n"
);
  printf(
"  purpose is to check the validity of your input files, and do so more\n");
  printf(
"  thoroughly than Triangle does.  Unlike Triangle, Show Me requires that\n");
  printf(
"  you have the X Windows system.  Sorry, Microsoft Windows users.\n");
  printf("\n");
  printf("Triangle on the Web:\n");
  printf("\n");
  printf("  To see an illustrated version of these instructions, check out\n");
  printf("\n");
  printf("    http://www.cs.cmu.edu/~quake/triangle.html\n");
  printf("\n");
  printf("A Brief Plea:\n");
  printf("\n");
  printf(
"  If you use Triangle, and especially if you use it to accomplish real\n");
  printf(
"  work, I would like very much to hear from you.  A short letter or email\n");
  printf(
"  (to jrs@cs.berkeley.edu) describing how you use Triangle will mean a lot\n"
);
  printf(
"  to me.  The more people I know are using this program, the more easily I\n"
);
  printf(
"  can justify spending time on improvements, which in turn will benefit\n");
  printf(
"  you.  Also, I can put you on a list to receive email whenever a new\n");
  printf("  version of Triangle is available.\n\n");
  printf(
"  If you use a mesh generated by Triangle in a publication, please include\n"
);
  printf(
"  an acknowledgment as well.  And please spell Triangle with a capital `T'!\n"
);
  printf(
"  If you want to include a citation, use `Jonathan Richard Shewchuk,\n");
  printf(
"  ``Triangle: Engineering a 2D Quality Mesh Generator and Delaunay\n");
  printf(
"  Triangulator,'' in Applied Computational Geometry:  Towards Geometric\n");
  printf(
"  Engineering (Ming C. Lin and Dinesh Manocha, editors), volume 1148 of\n");
  printf(
"  Lecture Notes in Computer Science, pages 203-222, Springer-Verlag,\n");
  printf(
"  Berlin, May 1996.  (From the First ACM Workshop on Applied Computational\n"
);
  printf("  Geometry.)'\n\n");
  printf("Research credit:\n\n");
  printf(
"  Of course, I can take credit for only a fraction of the ideas that made\n");
  printf(
"  this mesh generator possible.  Triangle owes its existence to the efforts\n"
);
  printf(
"  of many fine computational geometers and other researchers, including\n");
  printf(
"  Marshall Bern, L. Paul Chew, Kenneth L. Clarkson, Boris Delaunay, Rex A.\n"
);
  printf(
"  Dwyer, David Eppstein, Steven Fortune, Leonidas J. Guibas, Donald E.\n");
  printf(
"  Knuth, Charles L. Lawson, Der-Tsai Lee, Gary L. Miller, Ernst P. Mucke,\n");
  printf(
"  Steven E. Pav, Douglas M. Priest, Jim Ruppert, Isaac Saias, Bruce J.\n");
  printf(
"  Schachter, Micha Sharir, Peter W. Shor, Daniel D. Sleator, Jorge Stolfi,\n"
);
  printf("  Robert E. Tarjan, Alper Ungor, Christopher J. Van Wyk, Noel J.\n");
  printf(
"  Walkington, and Binhai Zhu.  See the comments at the beginning of the\n");
  printf("  source code for references.\n\n");
  triexit(0);
}

#endif


void internalerror()
{
  printf("  Please report this bug to jrs@cs.berkeley.edu\n");
  printf("  Include the message above, your input data set, and the exact\n");
  printf("    command line you used to run Triangle.\n");
  triexit(1);
}


#ifdef ANSI_DECLARATORS
void parsecommandline(int argc, char **argv, struct behavior *b)
#else
void parsecommandline(argc, argv, b)
int argc;
char **argv;
struct behavior *b;
#endif

{
#ifdef TRILIBRARY
#define STARTINDEX 0
#else
#define STARTINDEX 1
  int increment;
  int meshnumber;
#endif
  int i, j, k;
  char workstring[FILENAMESIZE];

  b->poly = b->refine = b->quality = 0;
  b->vararea = b->fixedarea = b->usertest = 0;
  b->regionattrib = b->convex = b->weighted = b->jettison = 0;
  b->firstnumber = 1;
  b->edgesout = b->voronoi = b->neighbors = b->geomview = 0;
  b->nobound = b->nopolywritten = b->nonodewritten = b->noelewritten = 0;
  b->noiterationnum = 0;
  b->noholes = b->noexact = 0;
  b->incremental = b->sweepline = 0;
  b->dwyer = 1;
  b->splitseg = 0;
  b->docheck = 0;
  b->nobisect = 0;
  b->conformdel = 0;
  b->steiner = -1;
  b->order = 1;
  b->minangle = 0.0;
  b->maxarea = -1.0;
  b->quiet = b->verbose = 0;
#ifndef TRILIBRARY
  b->innodefilename[0] = '\0';
#endif

  for (i = STARTINDEX; i < argc; i++) {
#ifndef TRILIBRARY
    if (argv[i][0] == '-') {
#endif
      for (j = STARTINDEX; argv[i][j] != '\0'; j++) {
        if (argv[i][j] == 'p') {
          b->poly = 1;
	}
#ifndef CDT_ONLY
        if (argv[i][j] == 'r') {
          b->refine = 1;
	}
        if (argv[i][j] == 'q') {
          b->quality = 1;
          if (((argv[i][j + 1] >= '0') && (argv[i][j + 1] <= '9')) ||
              (argv[i][j + 1] == '.')) {
            k = 0;
            while (((argv[i][j + 1] >= '0') && (argv[i][j + 1] <= '9')) ||
                   (argv[i][j + 1] == '.')) {
              j++;
              workstring[k] = argv[i][j];
              k++;
            }
            workstring[k] = '\0';
            b->minangle = (REAL) strtod(workstring, (char **) NULL);
	  } else {
            b->minangle = 20.0;
	  }
	}
        if (argv[i][j] == 'a') {
          b->quality = 1;
          if (((argv[i][j + 1] >= '0') && (argv[i][j + 1] <= '9')) ||
              (argv[i][j + 1] == '.')) {
            b->fixedarea = 1;
            k = 0;
            while (((argv[i][j + 1] >= '0') && (argv[i][j + 1] <= '9')) ||
                   (argv[i][j + 1] == '.')) {
              j++;
              workstring[k] = argv[i][j];
              k++;
            }
            workstring[k] = '\0';
            b->maxarea = (REAL) strtod(workstring, (char **) NULL);
            if (b->maxarea <= 0.0) {
              printf("Error:  Maximum area must be greater than zero.\n");
              triexit(1);
	    }
	  } else {
            b->vararea = 1;
	  }
	}
        if (argv[i][j] == 'u') {
          b->quality = 1;
          b->usertest = 1;
        }
#endif
        if (argv[i][j] == 'A') {
          b->regionattrib = 1;
        }
        if (argv[i][j] == 'c') {
          b->convex = 1;
        }
        if (argv[i][j] == 'w') {
          b->weighted = 1;
        }
        if (argv[i][j] == 'W') {
          b->weighted = 2;
        }
        if (argv[i][j] == 'j') {
          b->jettison = 1;
        }
        if (argv[i][j] == 'z') {
          b->firstnumber = 0;
        }
        if (argv[i][j] == 'e') {
          b->edgesout = 1;
	}
        if (argv[i][j] == 'v') {
          b->voronoi = 1;
	}
        if (argv[i][j] == 'n') {
          b->neighbors = 1;
	}
        if (argv[i][j] == 'g') {
          b->geomview = 1;
	}
        if (argv[i][j] == 'B') {
          b->nobound = 1;
	}
        if (argv[i][j] == 'P') {
          b->nopolywritten = 1;
	}
        if (argv[i][j] == 'N') {
          b->nonodewritten = 1;
	}
        if (argv[i][j] == 'E') {
          b->noelewritten = 1;
	}
#ifndef TRILIBRARY
        if (argv[i][j] == 'I') {
          b->noiterationnum = 1;
	}
#endif
        if (argv[i][j] == 'O') {
          b->noholes = 1;
	}
        if (argv[i][j] == 'X') {
          b->noexact = 1;
	}
        if (argv[i][j] == 'o') {
          if (argv[i][j + 1] == '2') {
            j++;
            b->order = 2;
          }
	}
#ifndef CDT_ONLY
        if (argv[i][j] == 'Y') {
          b->nobisect++;
	}
        if (argv[i][j] == 'S') {
          b->steiner = 0;
          while ((argv[i][j + 1] >= '0') && (argv[i][j + 1] <= '9')) {
            j++;
            b->steiner = b->steiner * 10 + (int) (argv[i][j] - '0');
          }
        }
#endif
#ifndef REDUCED
        if (argv[i][j] == 'i') {
          b->incremental = 1;
        }
        if (argv[i][j] == 'F') {
          b->sweepline = 1;
        }
#endif
        if (argv[i][j] == 'l') {
          b->dwyer = 0;
        }
#ifndef REDUCED
#ifndef CDT_ONLY
        if (argv[i][j] == 's') {
          b->splitseg = 1;
        }
        if ((argv[i][j] == 'D') || (argv[i][j] == 'L')) {
          b->quality = 1;
          b->conformdel = 1;
        }
#endif
        if (argv[i][j] == 'C') {
          b->docheck = 1;
        }
#endif
        if (argv[i][j] == 'Q') {
          b->quiet = 1;
        }
        if (argv[i][j] == 'V') {
          b->verbose++;
        }
#ifndef TRILIBRARY
        if ((argv[i][j] == 'h') || (argv[i][j] == 'H') ||
            (argv[i][j] == '?')) {
          info();
	}
#endif
      }
#ifndef TRILIBRARY
    } else {
      strncpy(b->innodefilename, argv[i], FILENAMESIZE - 1);
      b->innodefilename[FILENAMESIZE - 1] = '\0';
    }
#endif
  }
#ifndef TRILIBRARY
  if (b->innodefilename[0] == '\0') {
    syntax();
  }
  if (!strcmp(&b->innodefilename[strlen(b->innodefilename) - 5], ".node")) {
    b->innodefilename[strlen(b->innodefilename) - 5] = '\0';
  }
  if (!strcmp(&b->innodefilename[strlen(b->innodefilename) - 5], ".poly")) {
    b->innodefilename[strlen(b->innodefilename) - 5] = '\0';
    b->poly = 1;
  }
#ifndef CDT_ONLY
  if (!strcmp(&b->innodefilename[strlen(b->innodefilename) - 4], ".ele")) {
    b->innodefilename[strlen(b->innodefilename) - 4] = '\0';
    b->refine = 1;
  }
  if (!strcmp(&b->innodefilename[strlen(b->innodefilename) - 5], ".area")) {
    b->innodefilename[strlen(b->innodefilename) - 5] = '\0';
    b->refine = 1;
    b->quality = 1;
    b->vararea = 1;
  }
#endif
#endif
  b->usesegments = b->poly || b->refine || b->quality || b->convex;
  b->goodangle = cos(b->minangle * PI / 180.0);
  if (b->goodangle == 1.0) {
    b->offconstant = 0.0;
  } else {
    b->offconstant = 0.475 * sqrt((1.0 + b->goodangle) / (1.0 - b->goodangle));
  }
  b->goodangle *= b->goodangle;
  if (b->refine && b->noiterationnum) {
    printf(
      "Error:  You cannot use the -I switch when refining a triangulation.\n");
    triexit(1);
  }
    if (!b->refine && !b->poly) {
    b->vararea = 0;
  }
    if (b->refine || !b->poly) {
    b->regionattrib = 0;
  }
    if (b->weighted && (b->poly || b->quality)) {
    b->weighted = 0;
    if (!b->quiet) {
      printf("Warning:  weighted triangulations (-w, -W) are incompatible\n");
      printf("  with PSLGs (-p) and meshing (-q, -a, -u).  Weights ignored.\n"
             );
    }
  }
  if (b->jettison && b->nonodewritten && !b->quiet) {
    printf("Warning:  -j and -N switches are somewhat incompatible.\n");
    printf("  If any vertices are jettisoned, you will need the output\n");
    printf("  .node file to reconstruct the new node indices.");
  }

#ifndef TRILIBRARY
  strcpy(b->inpolyfilename, b->innodefilename);
  strcpy(b->inelefilename, b->innodefilename);
  strcpy(b->areafilename, b->innodefilename);
  increment = 0;
  strcpy(workstring, b->innodefilename);
  j = 1;
  while (workstring[j] != '\0') {
    if ((workstring[j] == '.') && (workstring[j + 1] != '\0')) {
      increment = j + 1;
    }
    j++;
  }
  meshnumber = 0;
  if (increment > 0) {
    j = increment;
    do {
      if ((workstring[j] >= '0') && (workstring[j] <= '9')) {
        meshnumber = meshnumber * 10 + (int) (workstring[j] - '0');
      } else {
        increment = 0;
      }
      j++;
    } while (workstring[j] != '\0');
  }
  if (b->noiterationnum) {
    strcpy(b->outnodefilename, b->innodefilename);
    strcpy(b->outelefilename, b->innodefilename);
    strcpy(b->edgefilename, b->innodefilename);
    strcpy(b->vnodefilename, b->innodefilename);
    strcpy(b->vedgefilename, b->innodefilename);
    strcpy(b->neighborfilename, b->innodefilename);
    strcpy(b->offfilename, b->innodefilename);
    strcat(b->outnodefilename, ".node");
    strcat(b->outelefilename, ".ele");
    strcat(b->edgefilename, ".edge");
    strcat(b->vnodefilename, ".v.node");
    strcat(b->vedgefilename, ".v.edge");
    strcat(b->neighborfilename, ".neigh");
    strcat(b->offfilename, ".off");
  } else if (increment == 0) {
    strcpy(b->outnodefilename, b->innodefilename);
    strcpy(b->outpolyfilename, b->innodefilename);
    strcpy(b->outelefilename, b->innodefilename);
    strcpy(b->edgefilename, b->innodefilename);
    strcpy(b->vnodefilename, b->innodefilename);
    strcpy(b->vedgefilename, b->innodefilename);
    strcpy(b->neighborfilename, b->innodefilename);
    strcpy(b->offfilename, b->innodefilename);
    strcat(b->outnodefilename, ".1.node");
    strcat(b->outpolyfilename, ".1.poly");
    strcat(b->outelefilename, ".1.ele");
    strcat(b->edgefilename, ".1.edge");
    strcat(b->vnodefilename, ".1.v.node");
    strcat(b->vedgefilename, ".1.v.edge");
    strcat(b->neighborfilename, ".1.neigh");
    strcat(b->offfilename, ".1.off");
  } else {
    workstring[increment] = '%';
    workstring[increment + 1] = 'd';
    workstring[increment + 2] = '\0';
    sprintf(b->outnodefilename, workstring, meshnumber + 1);
    strcpy(b->outpolyfilename, b->outnodefilename);
    strcpy(b->outelefilename, b->outnodefilename);
    strcpy(b->edgefilename, b->outnodefilename);
    strcpy(b->vnodefilename, b->outnodefilename);
    strcpy(b->vedgefilename, b->outnodefilename);
    strcpy(b->neighborfilename, b->outnodefilename);
    strcpy(b->offfilename, b->outnodefilename);
    strcat(b->outnodefilename, ".node");
    strcat(b->outpolyfilename, ".poly");
    strcat(b->outelefilename, ".ele");
    strcat(b->edgefilename, ".edge");
    strcat(b->vnodefilename, ".v.node");
    strcat(b->vedgefilename, ".v.edge");
    strcat(b->neighborfilename, ".neigh");
    strcat(b->offfilename, ".off");
  }
  strcat(b->innodefilename, ".node");
  strcat(b->inpolyfilename, ".poly");
  strcat(b->inelefilename, ".ele");
  strcat(b->areafilename, ".area");
#endif
}




#ifdef ANSI_DECLARATORS
void printtriangle(struct mesh *m, struct behavior *b, struct otri *t)
#else
void printtriangle(m, b, t)
struct mesh *m;
struct behavior *b;
struct otri *t;
#endif

{
  struct otri printtri;
  struct osub printsh;
  vertex printvertex;

  printf("triangle x%lx with orientation %d:\n", (TRIULONG) t->tri,
         t->orient);
  decode(t->tri[0], printtri);
  if (printtri.tri == m->dummytri) {
    printf("    [0] = Outer space\n");
  } else {
    printf("    [0] = x%lx  %d\n", (TRIULONG) printtri.tri,
           printtri.orient);
  }
  decode(t->tri[1], printtri);
  if (printtri.tri == m->dummytri) {
    printf("    [1] = Outer space\n");
  } else {
    printf("    [1] = x%lx  %d\n", (TRIULONG) printtri.tri,
           printtri.orient);
  }
  decode(t->tri[2], printtri);
  if (printtri.tri == m->dummytri) {
    printf("    [2] = Outer space\n");
  } else {
    printf("    [2] = x%lx  %d\n", (TRIULONG) printtri.tri,
           printtri.orient);
  }

  org(*t, printvertex);
  if (printvertex == (vertex) NULL)
    printf("    Origin[%d] = NULL\n", (t->orient + 1) % 3 + 3);
  else
    printf("    Origin[%d] = x%lx  (%.12g, %.12g)\n",
           (t->orient + 1) % 3 + 3, (TRIULONG) printvertex,
           printvertex[0], printvertex[1]);
  dest(*t, printvertex);
  if (printvertex == (vertex) NULL)
    printf("    Dest  [%d] = NULL\n", (t->orient + 2) % 3 + 3);
  else
    printf("    Dest  [%d] = x%lx  (%.12g, %.12g)\n",
           (t->orient + 2) % 3 + 3, (TRIULONG) printvertex,
           printvertex[0], printvertex[1]);
  apex(*t, printvertex);
  if (printvertex == (vertex) NULL)
    printf("    Apex  [%d] = NULL\n", t->orient + 3);
  else
    printf("    Apex  [%d] = x%lx  (%.12g, %.12g)\n",
           t->orient + 3, (TRIULONG) printvertex,
           printvertex[0], printvertex[1]);

  if (b->usesegments) {
    sdecode(t->tri[6], printsh);
    if (printsh.ss != m->dummysub) {
      printf("    [6] = x%lx  %d\n", (TRIULONG) printsh.ss,
             printsh.ssorient);
    }
    sdecode(t->tri[7], printsh);
    if (printsh.ss != m->dummysub) {
      printf("    [7] = x%lx  %d\n", (TRIULONG) printsh.ss,
             printsh.ssorient);
    }
    sdecode(t->tri[8], printsh);
    if (printsh.ss != m->dummysub) {
      printf("    [8] = x%lx  %d\n", (TRIULONG) printsh.ss,
             printsh.ssorient);
    }
  }

  if (b->vararea) {
    printf("    Area constraint:  %.4g\n", areabound(*t));
  }
}


#ifdef ANSI_DECLARATORS
void printsubseg(struct mesh *m, struct behavior *b, struct osub *s)
#else
void printsubseg(m, b, s)
struct mesh *m;
struct behavior *b;
struct osub *s;
#endif

{
  struct osub printsh;
  struct otri printtri;
  vertex printvertex;

  printf("subsegment x%lx with orientation %d and mark %d:\n",
         (TRIULONG) s->ss, s->ssorient, mark(*s));
  sdecode(s->ss[0], printsh);
  if (printsh.ss == m->dummysub) {
    printf("    [0] = No subsegment\n");
  } else {
    printf("    [0] = x%lx  %d\n", (TRIULONG) printsh.ss,
           printsh.ssorient);
  }
  sdecode(s->ss[1], printsh);
  if (printsh.ss == m->dummysub) {
    printf("    [1] = No subsegment\n");
  } else {
    printf("    [1] = x%lx  %d\n", (TRIULONG) printsh.ss,
           printsh.ssorient);
  }

  sorg(*s, printvertex);
  if (printvertex == (vertex) NULL)
    printf("    Origin[%d] = NULL\n", 2 + s->ssorient);
  else
    printf("    Origin[%d] = x%lx  (%.12g, %.12g)\n",
           2 + s->ssorient, (TRIULONG) printvertex,
           printvertex[0], printvertex[1]);
  sdest(*s, printvertex);
  if (printvertex == (vertex) NULL)
    printf("    Dest  [%d] = NULL\n", 3 - s->ssorient);
  else
    printf("    Dest  [%d] = x%lx  (%.12g, %.12g)\n",
           3 - s->ssorient, (TRIULONG) printvertex,
           printvertex[0], printvertex[1]);

  decode(s->ss[6], printtri);
  if (printtri.tri == m->dummytri) {
    printf("    [6] = Outer space\n");
  } else {
    printf("    [6] = x%lx  %d\n", (TRIULONG) printtri.tri,
           printtri.orient);
  }
  decode(s->ss[7], printtri);
  if (printtri.tri == m->dummytri) {
    printf("    [7] = Outer space\n");
  } else {
    printf("    [7] = x%lx  %d\n", (TRIULONG) printtri.tri,
           printtri.orient);
  }

  segorg(*s, printvertex);
  if (printvertex == (vertex) NULL)
    printf("    Segment origin[%d] = NULL\n", 4 + s->ssorient);
  else
    printf("    Segment origin[%d] = x%lx  (%.12g, %.12g)\n",
           4 + s->ssorient, (TRIULONG) printvertex,
           printvertex[0], printvertex[1]);
  segdest(*s, printvertex);
  if (printvertex == (vertex) NULL)
    printf("    Segment dest  [%d] = NULL\n", 5 - s->ssorient);
  else
    printf("    Segment dest  [%d] = x%lx  (%.12g, %.12g)\n",
           5 - s->ssorient, (TRIULONG) printvertex,
           printvertex[0], printvertex[1]);
}




#ifdef ANSI_DECLARATORS
void poolzero(struct memorypool *pool)
#else
void poolzero(pool)
struct memorypool *pool;
#endif

{
  pool->firstblock = (VOID **) NULL;
  pool->nowblock = (VOID **) NULL;
  pool->nextitem = (VOID *) NULL;
  pool->deaditemstack = (VOID *) NULL;
  pool->pathblock = (VOID **) NULL;
  pool->pathitem = (VOID *) NULL;
  pool->alignbytes = 0;
  pool->itembytes = 0;
  pool->itemsperblock = 0;
  pool->itemsfirstblock = 0;
  pool->items = 0;
  pool->maxitems = 0;
  pool->unallocateditems = 0;
  pool->pathitemsleft = 0;
}


#ifdef ANSI_DECLARATORS
void poolrestart(struct memorypool *pool)
#else
void poolrestart(pool)
struct memorypool *pool;
#endif

{
  TRIULONG alignptr;

  pool->items = 0;
  pool->maxitems = 0;

   pool->nowblock = pool->firstblock;
   alignptr = (TRIULONG) (pool->nowblock + 1);
   pool->nextitem = (VOID *)
    (alignptr + (TRIULONG) pool->alignbytes -
     (alignptr % (TRIULONG) pool->alignbytes));
   pool->unallocateditems = pool->itemsfirstblock;
   pool->deaditemstack = (VOID *) NULL;
}


#ifdef ANSI_DECLARATORS
void poolinit(struct memorypool *pool, int bytecount, int itemcount,
              int firstitemcount, int alignment)
#else
void poolinit(pool, bytecount, itemcount, firstitemcount, alignment)
struct memorypool *pool;
int bytecount;
int itemcount;
int firstitemcount;
int alignment;
#endif

{
      if (alignment > sizeof(VOID *)) {
    pool->alignbytes = alignment;
  } else {
    pool->alignbytes = sizeof(VOID *);
  }
  pool->itembytes = ((bytecount - 1) / pool->alignbytes + 1) *
                    pool->alignbytes;
  pool->itemsperblock = itemcount;
  if (firstitemcount == 0) {
    pool->itemsfirstblock = itemcount;
  } else {
    pool->itemsfirstblock = firstitemcount;
  }

     pool->firstblock = (VOID **)
    trimalloc(pool->itemsfirstblock * pool->itembytes + (int) sizeof(VOID *) +
              pool->alignbytes);
   *(pool->firstblock) = (VOID *) NULL;
  poolrestart(pool);
}


#ifdef ANSI_DECLARATORS
void pooldeinit(struct memorypool *pool)
#else
void pooldeinit(pool)
struct memorypool *pool;
#endif

{
  while (pool->firstblock != (VOID **) NULL) {
    pool->nowblock = (VOID **) *(pool->firstblock);
    trifree((VOID *) pool->firstblock);
    pool->firstblock = pool->nowblock;
  }
}


#ifdef ANSI_DECLARATORS
VOID *poolalloc(struct memorypool *pool)
#else
VOID *poolalloc(pool)
struct memorypool *pool;
#endif

{
  VOID *newitem;
  VOID **newblock;
  TRIULONG alignptr;

    if (pool->deaditemstack != (VOID *) NULL) {
    newitem = pool->deaditemstack;
    pool->deaditemstack = * (VOID **) pool->deaditemstack;
  } else {
       if (pool->unallocateditems == 0) {
           if (*(pool->nowblock) == (VOID *) NULL) {
               newblock = (VOID **) trimalloc(pool->itemsperblock * pool->itembytes +
                                       (int) sizeof(VOID *) +
                                       pool->alignbytes);
        *(pool->nowblock) = (VOID *) newblock;
               *newblock = (VOID *) NULL;
      }

           pool->nowblock = (VOID **) *(pool->nowblock);
                alignptr = (TRIULONG) (pool->nowblock + 1);
           pool->nextitem = (VOID *)
        (alignptr + (TRIULONG) pool->alignbytes -
         (alignptr % (TRIULONG) pool->alignbytes));
           pool->unallocateditems = pool->itemsperblock;
    }

       newitem = pool->nextitem;
       pool->nextitem = (VOID *) ((char *) pool->nextitem + pool->itembytes);
    pool->unallocateditems--;
    pool->maxitems++;
  }
  pool->items++;
  return newitem;
}


#ifdef ANSI_DECLARATORS
void pooldealloc(struct memorypool *pool, VOID *dyingitem)
#else
void pooldealloc(pool, dyingitem)
struct memorypool *pool;
VOID *dyingitem;
#endif

{
   *((VOID **) dyingitem) = pool->deaditemstack;
  pool->deaditemstack = dyingitem;
  pool->items--;
}


#ifdef ANSI_DECLARATORS
void traversalinit(struct memorypool *pool)
#else
void traversalinit(pool)
struct memorypool *pool;
#endif

{
  TRIULONG alignptr;

   pool->pathblock = pool->firstblock;
   alignptr = (TRIULONG) (pool->pathblock + 1);
   pool->pathitem = (VOID *)
    (alignptr + (TRIULONG) pool->alignbytes -
     (alignptr % (TRIULONG) pool->alignbytes));
   pool->pathitemsleft = pool->itemsfirstblock;
}


#ifdef ANSI_DECLARATORS
VOID *traverse(struct memorypool *pool)
#else
VOID *traverse(pool)
struct memorypool *pool;
#endif

{
  VOID *newitem;
  TRIULONG alignptr;

   if (pool->pathitem == pool->nextitem) {
    return (VOID *) NULL;
  }

   if (pool->pathitemsleft == 0) {
       pool->pathblock = (VOID **) *(pool->pathblock);
       alignptr = (TRIULONG) (pool->pathblock + 1);
       pool->pathitem = (VOID *)
      (alignptr + (TRIULONG) pool->alignbytes -
       (alignptr % (TRIULONG) pool->alignbytes));
       pool->pathitemsleft = pool->itemsperblock;
  }

  newitem = pool->pathitem;
   pool->pathitem = (VOID *) ((char *) pool->pathitem + pool->itembytes);
  pool->pathitemsleft--;
  return newitem;
}


#ifdef ANSI_DECLARATORS
void dummyinit(struct mesh *m, struct behavior *b, int trianglebytes,
               int subsegbytes)
#else
void dummyinit(m, b, trianglebytes, subsegbytes)
struct mesh *m;
struct behavior *b;
int trianglebytes;
int subsegbytes;
#endif

{
  TRIULONG alignptr;

   m->dummytribase = (triangle *) trimalloc(trianglebytes +
                                           m->triangles.alignbytes);
   alignptr = (TRIULONG) m->dummytribase;
  m->dummytri = (triangle *)
    (alignptr + (TRIULONG) m->triangles.alignbytes -
     (alignptr % (TRIULONG) m->triangles.alignbytes));
      m->dummytri[0] = (triangle) m->dummytri;
  m->dummytri[1] = (triangle) m->dummytri;
  m->dummytri[2] = (triangle) m->dummytri;
   m->dummytri[3] = (triangle) NULL;
  m->dummytri[4] = (triangle) NULL;
  m->dummytri[5] = (triangle) NULL;

  if (b->usesegments) {
             m->dummysubbase = (subseg *) trimalloc(subsegbytes +
                                           m->subsegs.alignbytes);
       alignptr = (TRIULONG) m->dummysubbase;
    m->dummysub = (subseg *)
      (alignptr + (TRIULONG) m->subsegs.alignbytes -
       (alignptr % (TRIULONG) m->subsegs.alignbytes));
                m->dummysub[0] = (subseg) m->dummysub;
    m->dummysub[1] = (subseg) m->dummysub;
       m->dummysub[2] = (subseg) NULL;
    m->dummysub[3] = (subseg) NULL;
    m->dummysub[4] = (subseg) NULL;
    m->dummysub[5] = (subseg) NULL;
       m->dummysub[6] = (subseg) m->dummytri;
    m->dummysub[7] = (subseg) m->dummytri;
       * (int *) (m->dummysub + 8) = 0;

          m->dummytri[6] = (triangle) m->dummysub;
    m->dummytri[7] = (triangle) m->dummysub;
    m->dummytri[8] = (triangle) m->dummysub;
  }
}


#ifdef ANSI_DECLARATORS
void initializevertexpool(struct mesh *m, struct behavior *b)
#else
void initializevertexpool(m, b)
struct mesh *m;
struct behavior *b;
#endif

{
  int vertexsize;

     m->vertexmarkindex = ((m->mesh_dim + m->nextras) * sizeof(REAL) +
                        sizeof(int) - 1) /
                       sizeof(int);
  vertexsize = (m->vertexmarkindex + 2) * sizeof(int);
  if (b->poly) {
          m->vertex2triindex = (vertexsize + sizeof(triangle) - 1) /
                         sizeof(triangle);
    vertexsize = (m->vertex2triindex + 1) * sizeof(triangle);
  }

   poolinit(&m->vertices, vertexsize, VERTEXPERBLOCK,
           m->invertices > VERTEXPERBLOCK ? m->invertices : VERTEXPERBLOCK,
           sizeof(REAL));
}


#ifdef ANSI_DECLARATORS
void initializetrisubpools(struct mesh *m, struct behavior *b)
#else
void initializetrisubpools(m, b)
struct mesh *m;
struct behavior *b;
#endif

{
  int trisize;

      m->highorderindex = 6 + (b->usesegments * 3);
   trisize = ((b->order + 1) * (b->order + 2) / 2 + (m->highorderindex - 3)) *
            sizeof(triangle);
    m->elemattribindex = (trisize + sizeof(REAL) - 1) / sizeof(REAL);
     m->areaboundindex = m->elemattribindex + m->eextras + b->regionattrib;
    if (b->vararea) {
    trisize = (m->areaboundindex + 1) * sizeof(REAL);
  } else if (m->eextras + b->regionattrib > 0) {
    trisize = m->areaboundindex * sizeof(REAL);
  }
      if ((b->voronoi || b->neighbors) &&
      (trisize < 6 * sizeof(triangle) + sizeof(int))) {
    trisize = 6 * sizeof(triangle) + sizeof(int);
  }

   poolinit(&m->triangles, trisize, TRIPERBLOCK,
           (2 * m->invertices - 2) > TRIPERBLOCK ? (2 * m->invertices - 2) :
           TRIPERBLOCK, 4);

  if (b->usesegments) {
          poolinit(&m->subsegs, 8 * sizeof(triangle) + sizeof(int),
             SUBSEGPERBLOCK, SUBSEGPERBLOCK, 4);

       dummyinit(m, b, m->triangles.itembytes, m->subsegs.itembytes);
  } else {
       dummyinit(m, b, m->triangles.itembytes, 0);
  }
}


#ifdef ANSI_DECLARATORS
void triangledealloc(struct mesh *m, triangle *dyingtriangle)
#else
void triangledealloc(m, dyingtriangle)
struct mesh *m;
triangle *dyingtriangle;
#endif

{
    killtri(dyingtriangle);
  pooldealloc(&m->triangles, (VOID *) dyingtriangle);
}


#ifdef ANSI_DECLARATORS
triangle *triangletraverse(struct mesh *m)
#else
triangle *triangletraverse(m)
struct mesh *m;
#endif

{
  triangle *newtriangle;

  do {
    newtriangle = (triangle *) traverse(&m->triangles);
    if (newtriangle == (triangle *) NULL) {
      return (triangle *) NULL;
    }
  } while (deadtri(newtriangle));
  return newtriangle;
}


#ifdef ANSI_DECLARATORS
void subsegdealloc(struct mesh *m, subseg *dyingsubseg)
#else
void subsegdealloc(m, dyingsubseg)
struct mesh *m;
subseg *dyingsubseg;
#endif

{
    killsubseg(dyingsubseg);
  pooldealloc(&m->subsegs, (VOID *) dyingsubseg);
}


#ifdef ANSI_DECLARATORS
subseg *subsegtraverse(struct mesh *m)
#else
subseg *subsegtraverse(m)
struct mesh *m;
#endif

{
  subseg *newsubseg;

  do {
    newsubseg = (subseg *) traverse(&m->subsegs);
    if (newsubseg == (subseg *) NULL) {
      return (subseg *) NULL;
    }
  } while (deadsubseg(newsubseg));
  return newsubseg;
}


#ifdef ANSI_DECLARATORS
void vertexdealloc(struct mesh *m, vertex dyingvertex)
#else
void vertexdealloc(m, dyingvertex)
struct mesh *m;
vertex dyingvertex;
#endif

{
    setvertextype(dyingvertex, DEADVERTEX);
  pooldealloc(&m->vertices, (VOID *) dyingvertex);
}


#ifdef ANSI_DECLARATORS
vertex vertextraverse(struct mesh *m)
#else
vertex vertextraverse(m)
struct mesh *m;
#endif

{
  vertex newvertex;

  do {
    newvertex = (vertex) traverse(&m->vertices);
    if (newvertex == (vertex) NULL) {
      return (vertex) NULL;
    }
  } while (vertextype(newvertex) == DEADVERTEX);
  return newvertex;
}


#ifndef CDT_ONLY

#ifdef ANSI_DECLARATORS
void badsubsegdealloc(struct mesh *m, struct badsubseg *dyingseg)
#else
void badsubsegdealloc(m, dyingseg)
struct mesh *m;
struct badsubseg *dyingseg;
#endif

{
    dyingseg->subsegorg = (vertex) NULL;
  pooldealloc(&m->badsubsegs, (VOID *) dyingseg);
}

#endif


#ifndef CDT_ONLY

#ifdef ANSI_DECLARATORS
struct badsubseg *badsubsegtraverse(struct mesh *m)
#else
struct badsubseg *badsubsegtraverse(m)
struct mesh *m;
#endif

{
  struct badsubseg *newseg;

  do {
    newseg = (struct badsubseg *) traverse(&m->badsubsegs);
    if (newseg == (struct badsubseg *) NULL) {
      return (struct badsubseg *) NULL;
    }
  } while (newseg->subsegorg == (vertex) NULL);
  return newseg;
}

#endif


#ifdef ANSI_DECLARATORS
vertex getvertex(struct mesh *m, struct behavior *b, int number)
#else
vertex getvertex(m, b, number)
struct mesh *m;
struct behavior *b;
int number;
#endif

{
  VOID **getblock;
  char *foundvertex;
  TRIULONG alignptr;
  int current;

  getblock = m->vertices.firstblock;
  current = b->firstnumber;

   if (current + m->vertices.itemsfirstblock <= number) {
    getblock = (VOID **) *getblock;
    current += m->vertices.itemsfirstblock;
    while (current + m->vertices.itemsperblock <= number) {
      getblock = (VOID **) *getblock;
      current += m->vertices.itemsperblock;
    }
  }

   alignptr = (TRIULONG) (getblock + 1);
  foundvertex = (char *) (alignptr + (TRIULONG) m->vertices.alignbytes -
                          (alignptr % (TRIULONG) m->vertices.alignbytes));
  return (vertex) (foundvertex + m->vertices.itembytes * (number - current));
}


#ifdef ANSI_DECLARATORS
void triangledeinit(struct mesh *m, struct behavior *b)
#else
void triangledeinit(m, b)
struct mesh *m;
struct behavior *b;
#endif

{
  pooldeinit(&m->triangles);
  trifree((VOID *) m->dummytribase);
  if (b->usesegments) {
    pooldeinit(&m->subsegs);
    trifree((VOID *) m->dummysubbase);
  }
  pooldeinit(&m->vertices);
#ifndef CDT_ONLY
  if (b->quality) {
    pooldeinit(&m->badsubsegs);
    if ((b->minangle > 0.0) || b->vararea || b->fixedarea || b->usertest) {
      pooldeinit(&m->badtriangles);
      pooldeinit(&m->flipstackers);
    }
  }
#endif
}




#ifdef ANSI_DECLARATORS
void maketriangle(struct mesh *m, struct behavior *b, struct otri *newotri)
#else
void maketriangle(m, b, newotri)
struct mesh *m;
struct behavior *b;
struct otri *newotri;
#endif

{
  int i;

  newotri->tri = (triangle *) poolalloc(&m->triangles);
   newotri->tri[0] = (triangle) m->dummytri;
  newotri->tri[1] = (triangle) m->dummytri;
  newotri->tri[2] = (triangle) m->dummytri;
   newotri->tri[3] = (triangle) NULL;
  newotri->tri[4] = (triangle) NULL;
  newotri->tri[5] = (triangle) NULL;
  if (b->usesegments) {
          newotri->tri[6] = (triangle) m->dummysub;
    newotri->tri[7] = (triangle) m->dummysub;
    newotri->tri[8] = (triangle) m->dummysub;
  }
  for (i = 0; i < m->eextras; i++) {
    setelemattribute(*newotri, i, 0.0);
  }
  if (b->vararea) {
    setareabound(*newotri, -1.0);
  }

  newotri->orient = 0;
}


#ifdef ANSI_DECLARATORS
void makesubseg(struct mesh *m, struct osub *newsubseg)
#else
void makesubseg(m, newsubseg)
struct mesh *m;
struct osub *newsubseg;
#endif

{
  newsubseg->ss = (subseg *) poolalloc(&m->subsegs);
    newsubseg->ss[0] = (subseg) m->dummysub;
  newsubseg->ss[1] = (subseg) m->dummysub;
   newsubseg->ss[2] = (subseg) NULL;
  newsubseg->ss[3] = (subseg) NULL;
  newsubseg->ss[4] = (subseg) NULL;
  newsubseg->ss[5] = (subseg) NULL;
   newsubseg->ss[6] = (subseg) m->dummytri;
  newsubseg->ss[7] = (subseg) m->dummytri;
   setmark(*newsubseg, 0);

  newsubseg->ssorient = 0;
}





#define Absolute(a)  ((a) >= 0.0 ? (a) : -(a))


#define Fast_Two_Sum_Tail(a, b, x, y) \
  bvirt = x - a; \
  y = b - bvirt

#define Fast_Two_Sum(a, b, x, y) \
  x = (REAL) (a + b); \
  Fast_Two_Sum_Tail(a, b, x, y)

#define Two_Sum_Tail(a, b, x, y) \
  bvirt = (REAL) (x - a); \
  avirt = x - bvirt; \
  bround = b - bvirt; \
  around = a - avirt; \
  y = around + bround

#define Two_Sum(a, b, x, y) \
  x = (REAL) (a + b); \
  Two_Sum_Tail(a, b, x, y)

#define Two_Diff_Tail(a, b, x, y) \
  bvirt = (REAL) (a - x); \
  avirt = x + bvirt; \
  bround = bvirt - b; \
  around = a - avirt; \
  y = around + bround

#define Two_Diff(a, b, x, y) \
  x = (REAL) (a - b); \
  Two_Diff_Tail(a, b, x, y)

#define Split(a, ahi, alo) \
  c = (REAL) (splitter * a); \
  abig = (REAL) (c - a); \
  ahi = c - abig; \
  alo = a - ahi

#define Two_Product_Tail(a, b, x, y) \
  Split(a, ahi, alo); \
  Split(b, bhi, blo); \
  err1 = x - (ahi * bhi); \
  err2 = err1 - (alo * bhi); \
  err3 = err2 - (ahi * blo); \
  y = (alo * blo) - err3

#define Two_Product(a, b, x, y) \
  x = (REAL) (a * b); \
  Two_Product_Tail(a, b, x, y)


#define Two_Product_Presplit(a, b, bhi, blo, x, y) \
  x = (REAL) (a * b); \
  Split(a, ahi, alo); \
  err1 = x - (ahi * bhi); \
  err2 = err1 - (alo * bhi); \
  err3 = err2 - (ahi * blo); \
  y = (alo * blo) - err3


#define Square_Tail(a, x, y) \
  Split(a, ahi, alo); \
  err1 = x - (ahi * ahi); \
  err3 = err1 - ((ahi + ahi) * alo); \
  y = (alo * alo) - err3

#define Square(a, x, y) \
  x = (REAL) (a * a); \
  Square_Tail(a, x, y)


#define Two_One_Sum(a1, a0, b, x2, x1, x0) \
  Two_Sum(a0, b , _i, x0); \
  Two_Sum(a1, _i, x2, x1)

#define Two_One_Diff(a1, a0, b, x2, x1, x0) \
  Two_Diff(a0, b , _i, x0); \
  Two_Sum( a1, _i, x2, x1)

#define Two_Two_Sum(a1, a0, b1, b0, x3, x2, x1, x0) \
  Two_One_Sum(a1, a0, b0, _j, _0, x0); \
  Two_One_Sum(_j, _0, b1, x3, x2, x1)

#define Two_Two_Diff(a1, a0, b1, b0, x3, x2, x1, x0) \
  Two_One_Diff(a1, a0, b0, _j, _0, x0); \
  Two_One_Diff(_j, _0, b1, x3, x2, x1)


#define Two_One_Product(a1, a0, b, x3, x2, x1, x0) \
  Split(b, bhi, blo); \
  Two_Product_Presplit(a0, b, bhi, blo, _i, x0); \
  Two_Product_Presplit(a1, b, bhi, blo, _j, _0); \
  Two_Sum(_i, _0, _k, x1); \
  Fast_Two_Sum(_j, _k, x3, x2)


void exactinit()
{
  REAL half;
  REAL check, lastcheck;
  int every_other;
#ifdef LINUX
  int cword;
#endif

#ifdef CPU86
#ifdef SINGLE
  _control87(_PC_24, _MCW_PC);
#else
  _control87(_PC_53, _MCW_PC);
#endif
#endif
#ifdef LINUX
#ifdef SINGLE
   cword = 4210;
#else
   cword = 4722;
#endif
  _FPU_SETCW(cword);
#endif

  every_other = 1;
  half = 0.5;
  epsilon = 1.0;
  splitter = 1.0;
  check = 1.0;
      do {
    lastcheck = check;
    epsilon *= half;
    if (every_other) {
      splitter *= 2.0;
    }
    every_other = !every_other;
    check = 1.0 + epsilon;
  } while ((check != 1.0) && (check != lastcheck));
  splitter += 1.0;
   resulterrbound = (3.0 + 8.0 * epsilon) * epsilon;
  ccwerrboundA = (3.0 + 16.0 * epsilon) * epsilon;
  ccwerrboundB = (2.0 + 12.0 * epsilon) * epsilon;
  ccwerrboundC = (9.0 + 64.0 * epsilon) * epsilon * epsilon;
  iccerrboundA = (10.0 + 96.0 * epsilon) * epsilon;
  iccerrboundB = (4.0 + 48.0 * epsilon) * epsilon;
  iccerrboundC = (44.0 + 576.0 * epsilon) * epsilon * epsilon;
  o3derrboundA = (7.0 + 56.0 * epsilon) * epsilon;
  o3derrboundB = (3.0 + 28.0 * epsilon) * epsilon;
  o3derrboundC = (26.0 + 288.0 * epsilon) * epsilon * epsilon;
}


#ifdef ANSI_DECLARATORS
int fast_expansion_sum_zeroelim(int elen, REAL *e, int flen, REAL *f, REAL *h)
#else
int fast_expansion_sum_zeroelim(elen, e, flen, f, h)
int elen;
REAL *e;
int flen;
REAL *f;
REAL *h;
#endif

{
  REAL Q;
  INEXACT REAL Qnew;
  INEXACT REAL hh;
  INEXACT REAL bvirt;
  REAL avirt, bround, around;
  int eindex, findex, hindex;
  REAL enow, fnow;

  enow = e[0];
  fnow = f[0];
  eindex = findex = 0;
  if ((fnow > enow) == (fnow > -enow)) {
    Q = enow;
    if (++eindex < elen) enow = e[eindex];
  } else {
    Q = fnow;
    if (++findex < flen) fnow = f[findex];
  }
  hindex = 0;
  if ((eindex < elen) && (findex < flen)) {
    if ((fnow > enow) == (fnow > -enow)) {
      Fast_Two_Sum(enow, Q, Qnew, hh);
      if (++eindex < elen) enow = e[eindex];
    } else {
      Fast_Two_Sum(fnow, Q, Qnew, hh);
      if (++findex < flen) fnow = f[findex];
    }
    Q = Qnew;
    if (hh != 0.0) {
      h[hindex++] = hh;
    }
    while ((eindex < elen) && (findex < flen)) {
      if ((fnow > enow) == (fnow > -enow)) {
        Two_Sum(Q, enow, Qnew, hh);
        if (++eindex < elen) enow = e[eindex];
      } else {
        Two_Sum(Q, fnow, Qnew, hh);
        if (++findex < flen) fnow = f[findex];
      }
      Q = Qnew;
      if (hh != 0.0) {
        h[hindex++] = hh;
      }
    }
  }
  while (eindex < elen) {
    Two_Sum(Q, enow, Qnew, hh);
    if (++eindex < elen) enow = e[eindex];
    Q = Qnew;
    if (hh != 0.0) {
      h[hindex++] = hh;
    }
  }
  while (findex < flen) {
    Two_Sum(Q, fnow, Qnew, hh);
    if (++findex < flen) fnow = f[findex];
    Q = Qnew;
    if (hh != 0.0) {
      h[hindex++] = hh;
    }
  }
  if ((Q != 0.0) || (hindex == 0)) {
    h[hindex++] = Q;
  }
  return hindex;
}


#ifdef ANSI_DECLARATORS
int scale_expansion_zeroelim(int elen, REAL *e, REAL b, REAL *h)
#else
int scale_expansion_zeroelim(elen, e, b, h)
int elen;
REAL *e;
REAL b;
REAL *h;
#endif

{
  INEXACT REAL Q, sum;
  REAL hh;
  INEXACT REAL product1;
  REAL product0;
  int eindex, hindex;
  REAL enow;
  INEXACT REAL bvirt;
  REAL avirt, bround, around;
  INEXACT REAL c;
  INEXACT REAL abig;
  REAL ahi, alo, bhi, blo;
  REAL err1, err2, err3;

  Split(b, bhi, blo);
  Two_Product_Presplit(e[0], b, bhi, blo, Q, hh);
  hindex = 0;
  if (hh != 0) {
    h[hindex++] = hh;
  }
  for (eindex = 1; eindex < elen; eindex++) {
    enow = e[eindex];
    Two_Product_Presplit(enow, b, bhi, blo, product1, product0);
    Two_Sum(Q, product0, sum, hh);
    if (hh != 0) {
      h[hindex++] = hh;
    }
    Fast_Two_Sum(product1, sum, Q, hh);
    if (hh != 0) {
      h[hindex++] = hh;
    }
  }
  if ((Q != 0.0) || (hindex == 0)) {
    h[hindex++] = Q;
  }
  return hindex;
}


#ifdef ANSI_DECLARATORS
REAL estimate(int elen, REAL *e)
#else
REAL estimate(elen, e)
int elen;
REAL *e;
#endif

{
  REAL Q;
  int eindex;

  Q = e[0];
  for (eindex = 1; eindex < elen; eindex++) {
    Q += e[eindex];
  }
  return Q;
}


#ifdef ANSI_DECLARATORS
REAL counterclockwiseadapt(vertex pa, vertex pb, vertex pc, REAL detsum)
#else
REAL counterclockwiseadapt(pa, pb, pc, detsum)
vertex pa;
vertex pb;
vertex pc;
REAL detsum;
#endif

{
  INEXACT REAL acx, acy, bcx, bcy;
  REAL acxtail, acytail, bcxtail, bcytail;
  INEXACT REAL detleft, detright;
  REAL detlefttail, detrighttail;
  REAL det, errbound;
  REAL B[4], C1[8], C2[12], D[16];
  INEXACT REAL B3;
  int C1length, C2length, Dlength;
  REAL u[4];
  INEXACT REAL u3;
  INEXACT REAL s1, t1;
  REAL s0, t0;

  INEXACT REAL bvirt;
  REAL avirt, bround, around;
  INEXACT REAL c;
  INEXACT REAL abig;
  REAL ahi, alo, bhi, blo;
  REAL err1, err2, err3;
  INEXACT REAL _i, _j;
  REAL _0;

  acx = (REAL) (pa[0] - pc[0]);
  bcx = (REAL) (pb[0] - pc[0]);
  acy = (REAL) (pa[1] - pc[1]);
  bcy = (REAL) (pb[1] - pc[1]);

  Two_Product(acx, bcy, detleft, detlefttail);
  Two_Product(acy, bcx, detright, detrighttail);

  Two_Two_Diff(detleft, detlefttail, detright, detrighttail,
               B3, B[2], B[1], B[0]);
  B[3] = B3;

  det = estimate(4, B);
  errbound = ccwerrboundB * detsum;
  if ((det >= errbound) || (-det >= errbound)) {
    return det;
  }

  Two_Diff_Tail(pa[0], pc[0], acx, acxtail);
  Two_Diff_Tail(pb[0], pc[0], bcx, bcxtail);
  Two_Diff_Tail(pa[1], pc[1], acy, acytail);
  Two_Diff_Tail(pb[1], pc[1], bcy, bcytail);

  if ((acxtail == 0.0) && (acytail == 0.0)
      && (bcxtail == 0.0) && (bcytail == 0.0)) {
    return det;
  }

  errbound = ccwerrboundC * detsum + resulterrbound * Absolute(det);
  det += (acx * bcytail + bcy * acxtail)
       - (acy * bcxtail + bcx * acytail);
  if ((det >= errbound) || (-det >= errbound)) {
    return det;
  }

  Two_Product(acxtail, bcy, s1, s0);
  Two_Product(acytail, bcx, t1, t0);
  Two_Two_Diff(s1, s0, t1, t0, u3, u[2], u[1], u[0]);
  u[3] = u3;
  C1length = fast_expansion_sum_zeroelim(4, B, 4, u, C1);

  Two_Product(acx, bcytail, s1, s0);
  Two_Product(acy, bcxtail, t1, t0);
  Two_Two_Diff(s1, s0, t1, t0, u3, u[2], u[1], u[0]);
  u[3] = u3;
  C2length = fast_expansion_sum_zeroelim(C1length, C1, 4, u, C2);

  Two_Product(acxtail, bcytail, s1, s0);
  Two_Product(acytail, bcxtail, t1, t0);
  Two_Two_Diff(s1, s0, t1, t0, u3, u[2], u[1], u[0]);
  u[3] = u3;
  Dlength = fast_expansion_sum_zeroelim(C2length, C2, 4, u, D);

  return(D[Dlength - 1]);
}

#ifdef ANSI_DECLARATORS
REAL counterclockwise(struct mesh *m, struct behavior *b,
                      vertex pa, vertex pb, vertex pc)
#else
REAL counterclockwise(m, b, pa, pb, pc)
struct mesh *m;
struct behavior *b;
vertex pa;
vertex pb;
vertex pc;
#endif

{
  REAL detleft, detright, det;
  REAL detsum, errbound;

  m->counterclockcount++;

  detleft = (pa[0] - pc[0]) * (pb[1] - pc[1]);
  detright = (pa[1] - pc[1]) * (pb[0] - pc[0]);
  det = detleft - detright;

  if (b->noexact) {
    return det;
  }

  if (detleft > 0.0) {
    if (detright <= 0.0) {
      return det;
    } else {
      detsum = detleft + detright;
    }
  } else if (detleft < 0.0) {
    if (detright >= 0.0) {
      return det;
    } else {
      detsum = -detleft - detright;
    }
  } else {
    return det;
  }

  errbound = ccwerrboundA * detsum;
  if ((det >= errbound) || (-det >= errbound)) {
    return det;
  }

  return counterclockwiseadapt(pa, pb, pc, detsum);
}


#ifdef ANSI_DECLARATORS
REAL incircleadapt(vertex pa, vertex pb, vertex pc, vertex pd, REAL permanent)
#else
REAL incircleadapt(pa, pb, pc, pd, permanent)
vertex pa;
vertex pb;
vertex pc;
vertex pd;
REAL permanent;
#endif

{
  INEXACT REAL adx, bdx, cdx, ady, bdy, cdy;
  REAL det, errbound;

  INEXACT REAL bdxcdy1, cdxbdy1, cdxady1, adxcdy1, adxbdy1, bdxady1;
  REAL bdxcdy0, cdxbdy0, cdxady0, adxcdy0, adxbdy0, bdxady0;
  REAL bc[4], ca[4], ab[4];
  INEXACT REAL bc3, ca3, ab3;
  REAL axbc[8], axxbc[16], aybc[8], ayybc[16], adet[32];
  int axbclen, axxbclen, aybclen, ayybclen, alen;
  REAL bxca[8], bxxca[16], byca[8], byyca[16], bdet[32];
  int bxcalen, bxxcalen, bycalen, byycalen, blen;
  REAL cxab[8], cxxab[16], cyab[8], cyyab[16], cdet[32];
  int cxablen, cxxablen, cyablen, cyyablen, clen;
  REAL abdet[64];
  int ablen;
  REAL fin1[1152], fin2[1152];
  REAL *finnow, *finother, *finswap;
  int finlength;

  REAL adxtail, bdxtail, cdxtail, adytail, bdytail, cdytail;
  INEXACT REAL adxadx1, adyady1, bdxbdx1, bdybdy1, cdxcdx1, cdycdy1;
  REAL adxadx0, adyady0, bdxbdx0, bdybdy0, cdxcdx0, cdycdy0;
  REAL aa[4], bb[4], cc[4];
  INEXACT REAL aa3, bb3, cc3;
  INEXACT REAL ti1, tj1;
  REAL ti0, tj0;
  REAL u[4], v[4];
  INEXACT REAL u3, v3;
  REAL temp8[8], temp16a[16], temp16b[16], temp16c[16];
  REAL temp32a[32], temp32b[32], temp48[48], temp64[64];
  int temp8len, temp16alen, temp16blen, temp16clen;
  int temp32alen, temp32blen, temp48len, temp64len;
  REAL axtbb[8], axtcc[8], aytbb[8], aytcc[8];
  int axtbblen, axtcclen, aytbblen, aytcclen;
  REAL bxtaa[8], bxtcc[8], bytaa[8], bytcc[8];
  int bxtaalen, bxtcclen, bytaalen, bytcclen;
  REAL cxtaa[8], cxtbb[8], cytaa[8], cytbb[8];
  int cxtaalen, cxtbblen, cytaalen, cytbblen;
  REAL axtbc[8], aytbc[8], bxtca[8], bytca[8], cxtab[8], cytab[8];
  int axtbclen, aytbclen, bxtcalen, bytcalen, cxtablen, cytablen;
  REAL axtbct[16], aytbct[16], bxtcat[16], bytcat[16], cxtabt[16], cytabt[16];
  int axtbctlen, aytbctlen, bxtcatlen, bytcatlen, cxtabtlen, cytabtlen;
  REAL axtbctt[8], aytbctt[8], bxtcatt[8];
  REAL bytcatt[8], cxtabtt[8], cytabtt[8];
  int axtbcttlen, aytbcttlen, bxtcattlen, bytcattlen, cxtabttlen, cytabttlen;
  REAL abt[8], bct[8], cat[8];
  int abtlen, bctlen, catlen;
  REAL abtt[4], bctt[4], catt[4];
  int abttlen, bcttlen, cattlen;
  INEXACT REAL abtt3, bctt3, catt3;
  REAL negate;

  INEXACT REAL bvirt;
  REAL avirt, bround, around;
  INEXACT REAL c;
  INEXACT REAL abig;
  REAL ahi, alo, bhi, blo;
  REAL err1, err2, err3;
  INEXACT REAL _i, _j;
  REAL _0;

  adx = (REAL) (pa[0] - pd[0]);
  bdx = (REAL) (pb[0] - pd[0]);
  cdx = (REAL) (pc[0] - pd[0]);
  ady = (REAL) (pa[1] - pd[1]);
  bdy = (REAL) (pb[1] - pd[1]);
  cdy = (REAL) (pc[1] - pd[1]);

  Two_Product(bdx, cdy, bdxcdy1, bdxcdy0);
  Two_Product(cdx, bdy, cdxbdy1, cdxbdy0);
  Two_Two_Diff(bdxcdy1, bdxcdy0, cdxbdy1, cdxbdy0, bc3, bc[2], bc[1], bc[0]);
  bc[3] = bc3;
  axbclen = scale_expansion_zeroelim(4, bc, adx, axbc);
  axxbclen = scale_expansion_zeroelim(axbclen, axbc, adx, axxbc);
  aybclen = scale_expansion_zeroelim(4, bc, ady, aybc);
  ayybclen = scale_expansion_zeroelim(aybclen, aybc, ady, ayybc);
  alen = fast_expansion_sum_zeroelim(axxbclen, axxbc, ayybclen, ayybc, adet);

  Two_Product(cdx, ady, cdxady1, cdxady0);
  Two_Product(adx, cdy, adxcdy1, adxcdy0);
  Two_Two_Diff(cdxady1, cdxady0, adxcdy1, adxcdy0, ca3, ca[2], ca[1], ca[0]);
  ca[3] = ca3;
  bxcalen = scale_expansion_zeroelim(4, ca, bdx, bxca);
  bxxcalen = scale_expansion_zeroelim(bxcalen, bxca, bdx, bxxca);
  bycalen = scale_expansion_zeroelim(4, ca, bdy, byca);
  byycalen = scale_expansion_zeroelim(bycalen, byca, bdy, byyca);
  blen = fast_expansion_sum_zeroelim(bxxcalen, bxxca, byycalen, byyca, bdet);

  Two_Product(adx, bdy, adxbdy1, adxbdy0);
  Two_Product(bdx, ady, bdxady1, bdxady0);
  Two_Two_Diff(adxbdy1, adxbdy0, bdxady1, bdxady0, ab3, ab[2], ab[1], ab[0]);
  ab[3] = ab3;
  cxablen = scale_expansion_zeroelim(4, ab, cdx, cxab);
  cxxablen = scale_expansion_zeroelim(cxablen, cxab, cdx, cxxab);
  cyablen = scale_expansion_zeroelim(4, ab, cdy, cyab);
  cyyablen = scale_expansion_zeroelim(cyablen, cyab, cdy, cyyab);
  clen = fast_expansion_sum_zeroelim(cxxablen, cxxab, cyyablen, cyyab, cdet);

  ablen = fast_expansion_sum_zeroelim(alen, adet, blen, bdet, abdet);
  finlength = fast_expansion_sum_zeroelim(ablen, abdet, clen, cdet, fin1);

  det = estimate(finlength, fin1);
  errbound = iccerrboundB * permanent;
  if ((det >= errbound) || (-det >= errbound)) {
    return det;
  }

  Two_Diff_Tail(pa[0], pd[0], adx, adxtail);
  Two_Diff_Tail(pa[1], pd[1], ady, adytail);
  Two_Diff_Tail(pb[0], pd[0], bdx, bdxtail);
  Two_Diff_Tail(pb[1], pd[1], bdy, bdytail);
  Two_Diff_Tail(pc[0], pd[0], cdx, cdxtail);
  Two_Diff_Tail(pc[1], pd[1], cdy, cdytail);
  if ((adxtail == 0.0) && (bdxtail == 0.0) && (cdxtail == 0.0)
      && (adytail == 0.0) && (bdytail == 0.0) && (cdytail == 0.0)) {
    return det;
  }

  errbound = iccerrboundC * permanent + resulterrbound * Absolute(det);
  det += ((adx * adx + ady * ady) * ((bdx * cdytail + cdy * bdxtail)
                                     - (bdy * cdxtail + cdx * bdytail))
          + 2.0 * (adx * adxtail + ady * adytail) * (bdx * cdy - bdy * cdx))
       + ((bdx * bdx + bdy * bdy) * ((cdx * adytail + ady * cdxtail)
                                     - (cdy * adxtail + adx * cdytail))
          + 2.0 * (bdx * bdxtail + bdy * bdytail) * (cdx * ady - cdy * adx))
       + ((cdx * cdx + cdy * cdy) * ((adx * bdytail + bdy * adxtail)
                                     - (ady * bdxtail + bdx * adytail))
          + 2.0 * (cdx * cdxtail + cdy * cdytail) * (adx * bdy - ady * bdx));
  if ((det >= errbound) || (-det >= errbound)) {
    return det;
  }

  finnow = fin1;
  finother = fin2;

  if ((bdxtail != 0.0) || (bdytail != 0.0)
      || (cdxtail != 0.0) || (cdytail != 0.0)) {
    Square(adx, adxadx1, adxadx0);
    Square(ady, adyady1, adyady0);
    Two_Two_Sum(adxadx1, adxadx0, adyady1, adyady0, aa3, aa[2], aa[1], aa[0]);
    aa[3] = aa3;
  }
  if ((cdxtail != 0.0) || (cdytail != 0.0)
      || (adxtail != 0.0) || (adytail != 0.0)) {
    Square(bdx, bdxbdx1, bdxbdx0);
    Square(bdy, bdybdy1, bdybdy0);
    Two_Two_Sum(bdxbdx1, bdxbdx0, bdybdy1, bdybdy0, bb3, bb[2], bb[1], bb[0]);
    bb[3] = bb3;
  }
  if ((adxtail != 0.0) || (adytail != 0.0)
      || (bdxtail != 0.0) || (bdytail != 0.0)) {
    Square(cdx, cdxcdx1, cdxcdx0);
    Square(cdy, cdycdy1, cdycdy0);
    Two_Two_Sum(cdxcdx1, cdxcdx0, cdycdy1, cdycdy0, cc3, cc[2], cc[1], cc[0]);
    cc[3] = cc3;
  }

  if (adxtail != 0.0) {
    axtbclen = scale_expansion_zeroelim(4, bc, adxtail, axtbc);
    temp16alen = scale_expansion_zeroelim(axtbclen, axtbc, 2.0 * adx,
                                          temp16a);

    axtcclen = scale_expansion_zeroelim(4, cc, adxtail, axtcc);
    temp16blen = scale_expansion_zeroelim(axtcclen, axtcc, bdy, temp16b);

    axtbblen = scale_expansion_zeroelim(4, bb, adxtail, axtbb);
    temp16clen = scale_expansion_zeroelim(axtbblen, axtbb, -cdy, temp16c);

    temp32alen = fast_expansion_sum_zeroelim(temp16alen, temp16a,
                                            temp16blen, temp16b, temp32a);
    temp48len = fast_expansion_sum_zeroelim(temp16clen, temp16c,
                                            temp32alen, temp32a, temp48);
    finlength = fast_expansion_sum_zeroelim(finlength, finnow, temp48len,
                                            temp48, finother);
    finswap = finnow; finnow = finother; finother = finswap;
  }
  if (adytail != 0.0) {
    aytbclen = scale_expansion_zeroelim(4, bc, adytail, aytbc);
    temp16alen = scale_expansion_zeroelim(aytbclen, aytbc, 2.0 * ady,
                                          temp16a);

    aytbblen = scale_expansion_zeroelim(4, bb, adytail, aytbb);
    temp16blen = scale_expansion_zeroelim(aytbblen, aytbb, cdx, temp16b);

    aytcclen = scale_expansion_zeroelim(4, cc, adytail, aytcc);
    temp16clen = scale_expansion_zeroelim(aytcclen, aytcc, -bdx, temp16c);

    temp32alen = fast_expansion_sum_zeroelim(temp16alen, temp16a,
                                            temp16blen, temp16b, temp32a);
    temp48len = fast_expansion_sum_zeroelim(temp16clen, temp16c,
                                            temp32alen, temp32a, temp48);
    finlength = fast_expansion_sum_zeroelim(finlength, finnow, temp48len,
                                            temp48, finother);
    finswap = finnow; finnow = finother; finother = finswap;
  }
  if (bdxtail != 0.0) {
    bxtcalen = scale_expansion_zeroelim(4, ca, bdxtail, bxtca);
    temp16alen = scale_expansion_zeroelim(bxtcalen, bxtca, 2.0 * bdx,
                                          temp16a);

    bxtaalen = scale_expansion_zeroelim(4, aa, bdxtail, bxtaa);
    temp16blen = scale_expansion_zeroelim(bxtaalen, bxtaa, cdy, temp16b);

    bxtcclen = scale_expansion_zeroelim(4, cc, bdxtail, bxtcc);
    temp16clen = scale_expansion_zeroelim(bxtcclen, bxtcc, -ady, temp16c);

    temp32alen = fast_expansion_sum_zeroelim(temp16alen, temp16a,
                                            temp16blen, temp16b, temp32a);
    temp48len = fast_expansion_sum_zeroelim(temp16clen, temp16c,
                                            temp32alen, temp32a, temp48);
    finlength = fast_expansion_sum_zeroelim(finlength, finnow, temp48len,
                                            temp48, finother);
    finswap = finnow; finnow = finother; finother = finswap;
  }
  if (bdytail != 0.0) {
    bytcalen = scale_expansion_zeroelim(4, ca, bdytail, bytca);
    temp16alen = scale_expansion_zeroelim(bytcalen, bytca, 2.0 * bdy,
                                          temp16a);

    bytcclen = scale_expansion_zeroelim(4, cc, bdytail, bytcc);
    temp16blen = scale_expansion_zeroelim(bytcclen, bytcc, adx, temp16b);

    bytaalen = scale_expansion_zeroelim(4, aa, bdytail, bytaa);
    temp16clen = scale_expansion_zeroelim(bytaalen, bytaa, -cdx, temp16c);

    temp32alen = fast_expansion_sum_zeroelim(temp16alen, temp16a,
                                            temp16blen, temp16b, temp32a);
    temp48len = fast_expansion_sum_zeroelim(temp16clen, temp16c,
                                            temp32alen, temp32a, temp48);
    finlength = fast_expansion_sum_zeroelim(finlength, finnow, temp48len,
                                            temp48, finother);
    finswap = finnow; finnow = finother; finother = finswap;
  }
  if (cdxtail != 0.0) {
    cxtablen = scale_expansion_zeroelim(4, ab, cdxtail, cxtab);
    temp16alen = scale_expansion_zeroelim(cxtablen, cxtab, 2.0 * cdx,
                                          temp16a);

    cxtbblen = scale_expansion_zeroelim(4, bb, cdxtail, cxtbb);
    temp16blen = scale_expansion_zeroelim(cxtbblen, cxtbb, ady, temp16b);

    cxtaalen = scale_expansion_zeroelim(4, aa, cdxtail, cxtaa);
    temp16clen = scale_expansion_zeroelim(cxtaalen, cxtaa, -bdy, temp16c);

    temp32alen = fast_expansion_sum_zeroelim(temp16alen, temp16a,
                                            temp16blen, temp16b, temp32a);
    temp48len = fast_expansion_sum_zeroelim(temp16clen, temp16c,
                                            temp32alen, temp32a, temp48);
    finlength = fast_expansion_sum_zeroelim(finlength, finnow, temp48len,
                                            temp48, finother);
    finswap = finnow; finnow = finother; finother = finswap;
  }
  if (cdytail != 0.0) {
    cytablen = scale_expansion_zeroelim(4, ab, cdytail, cytab);
    temp16alen = scale_expansion_zeroelim(cytablen, cytab, 2.0 * cdy,
                                          temp16a);

    cytaalen = scale_expansion_zeroelim(4, aa, cdytail, cytaa);
    temp16blen = scale_expansion_zeroelim(cytaalen, cytaa, bdx, temp16b);

    cytbblen = scale_expansion_zeroelim(4, bb, cdytail, cytbb);
    temp16clen = scale_expansion_zeroelim(cytbblen, cytbb, -adx, temp16c);

    temp32alen = fast_expansion_sum_zeroelim(temp16alen, temp16a,
                                            temp16blen, temp16b, temp32a);
    temp48len = fast_expansion_sum_zeroelim(temp16clen, temp16c,
                                            temp32alen, temp32a, temp48);
    finlength = fast_expansion_sum_zeroelim(finlength, finnow, temp48len,
                                            temp48, finother);
    finswap = finnow; finnow = finother; finother = finswap;
  }

  if ((adxtail != 0.0) || (adytail != 0.0)) {
    if ((bdxtail != 0.0) || (bdytail != 0.0)
        || (cdxtail != 0.0) || (cdytail != 0.0)) {
      Two_Product(bdxtail, cdy, ti1, ti0);
      Two_Product(bdx, cdytail, tj1, tj0);
      Two_Two_Sum(ti1, ti0, tj1, tj0, u3, u[2], u[1], u[0]);
      u[3] = u3;
      negate = -bdy;
      Two_Product(cdxtail, negate, ti1, ti0);
      negate = -bdytail;
      Two_Product(cdx, negate, tj1, tj0);
      Two_Two_Sum(ti1, ti0, tj1, tj0, v3, v[2], v[1], v[0]);
      v[3] = v3;
      bctlen = fast_expansion_sum_zeroelim(4, u, 4, v, bct);

      Two_Product(bdxtail, cdytail, ti1, ti0);
      Two_Product(cdxtail, bdytail, tj1, tj0);
      Two_Two_Diff(ti1, ti0, tj1, tj0, bctt3, bctt[2], bctt[1], bctt[0]);
      bctt[3] = bctt3;
      bcttlen = 4;
    } else {
      bct[0] = 0.0;
      bctlen = 1;
      bctt[0] = 0.0;
      bcttlen = 1;
    }

    if (adxtail != 0.0) {
      temp16alen = scale_expansion_zeroelim(axtbclen, axtbc, adxtail, temp16a);
      axtbctlen = scale_expansion_zeroelim(bctlen, bct, adxtail, axtbct);
      temp32alen = scale_expansion_zeroelim(axtbctlen, axtbct, 2.0 * adx,
                                            temp32a);
      temp48len = fast_expansion_sum_zeroelim(temp16alen, temp16a,
                                              temp32alen, temp32a, temp48);
      finlength = fast_expansion_sum_zeroelim(finlength, finnow, temp48len,
                                              temp48, finother);
      finswap = finnow; finnow = finother; finother = finswap;
      if (bdytail != 0.0) {
        temp8len = scale_expansion_zeroelim(4, cc, adxtail, temp8);
        temp16alen = scale_expansion_zeroelim(temp8len, temp8, bdytail,
                                              temp16a);
        finlength = fast_expansion_sum_zeroelim(finlength, finnow, temp16alen,
                                                temp16a, finother);
        finswap = finnow; finnow = finother; finother = finswap;
      }
      if (cdytail != 0.0) {
        temp8len = scale_expansion_zeroelim(4, bb, -adxtail, temp8);
        temp16alen = scale_expansion_zeroelim(temp8len, temp8, cdytail,
                                              temp16a);
        finlength = fast_expansion_sum_zeroelim(finlength, finnow, temp16alen,
                                                temp16a, finother);
        finswap = finnow; finnow = finother; finother = finswap;
      }

      temp32alen = scale_expansion_zeroelim(axtbctlen, axtbct, adxtail,
                                            temp32a);
      axtbcttlen = scale_expansion_zeroelim(bcttlen, bctt, adxtail, axtbctt);
      temp16alen = scale_expansion_zeroelim(axtbcttlen, axtbctt, 2.0 * adx,
                                            temp16a);
      temp16blen = scale_expansion_zeroelim(axtbcttlen, axtbctt, adxtail,
                                            temp16b);
      temp32blen = fast_expansion_sum_zeroelim(temp16alen, temp16a,
                                              temp16blen, temp16b, temp32b);
      temp64len = fast_expansion_sum_zeroelim(temp32alen, temp32a,
                                              temp32blen, temp32b, temp64);
      finlength = fast_expansion_sum_zeroelim(finlength, finnow, temp64len,
                                              temp64, finother);
      finswap = finnow; finnow = finother; finother = finswap;
    }
    if (adytail != 0.0) {
      temp16alen = scale_expansion_zeroelim(aytbclen, aytbc, adytail, temp16a);
      aytbctlen = scale_expansion_zeroelim(bctlen, bct, adytail, aytbct);
      temp32alen = scale_expansion_zeroelim(aytbctlen, aytbct, 2.0 * ady,
                                            temp32a);
      temp48len = fast_expansion_sum_zeroelim(temp16alen, temp16a,
                                              temp32alen, temp32a, temp48);
      finlength = fast_expansion_sum_zeroelim(finlength, finnow, temp48len,
                                              temp48, finother);
      finswap = finnow; finnow = finother; finother = finswap;


      temp32alen = scale_expansion_zeroelim(aytbctlen, aytbct, adytail,
                                            temp32a);
      aytbcttlen = scale_expansion_zeroelim(bcttlen, bctt, adytail, aytbctt);
      temp16alen = scale_expansion_zeroelim(aytbcttlen, aytbctt, 2.0 * ady,
                                            temp16a);
      temp16blen = scale_expansion_zeroelim(aytbcttlen, aytbctt, adytail,
                                            temp16b);
      temp32blen = fast_expansion_sum_zeroelim(temp16alen, temp16a,
                                              temp16blen, temp16b, temp32b);
      temp64len = fast_expansion_sum_zeroelim(temp32alen, temp32a,
                                              temp32blen, temp32b, temp64);
      finlength = fast_expansion_sum_zeroelim(finlength, finnow, temp64len,
                                              temp64, finother);
      finswap = finnow; finnow = finother; finother = finswap;
    }
  }
  if ((bdxtail != 0.0) || (bdytail != 0.0)) {
    if ((cdxtail != 0.0) || (cdytail != 0.0)
        || (adxtail != 0.0) || (adytail != 0.0)) {
      Two_Product(cdxtail, ady, ti1, ti0);
      Two_Product(cdx, adytail, tj1, tj0);
      Two_Two_Sum(ti1, ti0, tj1, tj0, u3, u[2], u[1], u[0]);
      u[3] = u3;
      negate = -cdy;
      Two_Product(adxtail, negate, ti1, ti0);
      negate = -cdytail;
      Two_Product(adx, negate, tj1, tj0);
      Two_Two_Sum(ti1, ti0, tj1, tj0, v3, v[2], v[1], v[0]);
      v[3] = v3;
      catlen = fast_expansion_sum_zeroelim(4, u, 4, v, cat);

      Two_Product(cdxtail, adytail, ti1, ti0);
      Two_Product(adxtail, cdytail, tj1, tj0);
      Two_Two_Diff(ti1, ti0, tj1, tj0, catt3, catt[2], catt[1], catt[0]);
      catt[3] = catt3;
      cattlen = 4;
    } else {
      cat[0] = 0.0;
      catlen = 1;
      catt[0] = 0.0;
      cattlen = 1;
    }

    if (bdxtail != 0.0) {
      temp16alen = scale_expansion_zeroelim(bxtcalen, bxtca, bdxtail, temp16a);
      bxtcatlen = scale_expansion_zeroelim(catlen, cat, bdxtail, bxtcat);
      temp32alen = scale_expansion_zeroelim(bxtcatlen, bxtcat, 2.0 * bdx,
                                            temp32a);
      temp48len = fast_expansion_sum_zeroelim(temp16alen, temp16a,
                                              temp32alen, temp32a, temp48);
      finlength = fast_expansion_sum_zeroelim(finlength, finnow, temp48len,
                                              temp48, finother);
      finswap = finnow; finnow = finother; finother = finswap;
      if (cdytail != 0.0) {
        temp8len = scale_expansion_zeroelim(4, aa, bdxtail, temp8);
        temp16alen = scale_expansion_zeroelim(temp8len, temp8, cdytail,
                                              temp16a);
        finlength = fast_expansion_sum_zeroelim(finlength, finnow, temp16alen,
                                                temp16a, finother);
        finswap = finnow; finnow = finother; finother = finswap;
      }
      if (adytail != 0.0) {
        temp8len = scale_expansion_zeroelim(4, cc, -bdxtail, temp8);
        temp16alen = scale_expansion_zeroelim(temp8len, temp8, adytail,
                                              temp16a);
        finlength = fast_expansion_sum_zeroelim(finlength, finnow, temp16alen,
                                                temp16a, finother);
        finswap = finnow; finnow = finother; finother = finswap;
      }

      temp32alen = scale_expansion_zeroelim(bxtcatlen, bxtcat, bdxtail,
                                            temp32a);
      bxtcattlen = scale_expansion_zeroelim(cattlen, catt, bdxtail, bxtcatt);
      temp16alen = scale_expansion_zeroelim(bxtcattlen, bxtcatt, 2.0 * bdx,
                                            temp16a);
      temp16blen = scale_expansion_zeroelim(bxtcattlen, bxtcatt, bdxtail,
                                            temp16b);
      temp32blen = fast_expansion_sum_zeroelim(temp16alen, temp16a,
                                              temp16blen, temp16b, temp32b);
      temp64len = fast_expansion_sum_zeroelim(temp32alen, temp32a,
                                              temp32blen, temp32b, temp64);
      finlength = fast_expansion_sum_zeroelim(finlength, finnow, temp64len,
                                              temp64, finother);
      finswap = finnow; finnow = finother; finother = finswap;
    }
    if (bdytail != 0.0) {
      temp16alen = scale_expansion_zeroelim(bytcalen, bytca, bdytail, temp16a);
      bytcatlen = scale_expansion_zeroelim(catlen, cat, bdytail, bytcat);
      temp32alen = scale_expansion_zeroelim(bytcatlen, bytcat, 2.0 * bdy,
                                            temp32a);
      temp48len = fast_expansion_sum_zeroelim(temp16alen, temp16a,
                                              temp32alen, temp32a, temp48);
      finlength = fast_expansion_sum_zeroelim(finlength, finnow, temp48len,
                                              temp48, finother);
      finswap = finnow; finnow = finother; finother = finswap;


      temp32alen = scale_expansion_zeroelim(bytcatlen, bytcat, bdytail,
                                            temp32a);
      bytcattlen = scale_expansion_zeroelim(cattlen, catt, bdytail, bytcatt);
      temp16alen = scale_expansion_zeroelim(bytcattlen, bytcatt, 2.0 * bdy,
                                            temp16a);
      temp16blen = scale_expansion_zeroelim(bytcattlen, bytcatt, bdytail,
                                            temp16b);
      temp32blen = fast_expansion_sum_zeroelim(temp16alen, temp16a,
                                              temp16blen, temp16b, temp32b);
      temp64len = fast_expansion_sum_zeroelim(temp32alen, temp32a,
                                              temp32blen, temp32b, temp64);
      finlength = fast_expansion_sum_zeroelim(finlength, finnow, temp64len,
                                              temp64, finother);
      finswap = finnow; finnow = finother; finother = finswap;
    }
  }
  if ((cdxtail != 0.0) || (cdytail != 0.0)) {
    if ((adxtail != 0.0) || (adytail != 0.0)
        || (bdxtail != 0.0) || (bdytail != 0.0)) {
      Two_Product(adxtail, bdy, ti1, ti0);
      Two_Product(adx, bdytail, tj1, tj0);
      Two_Two_Sum(ti1, ti0, tj1, tj0, u3, u[2], u[1], u[0]);
      u[3] = u3;
      negate = -ady;
      Two_Product(bdxtail, negate, ti1, ti0);
      negate = -adytail;
      Two_Product(bdx, negate, tj1, tj0);
      Two_Two_Sum(ti1, ti0, tj1, tj0, v3, v[2], v[1], v[0]);
      v[3] = v3;
      abtlen = fast_expansion_sum_zeroelim(4, u, 4, v, abt);

      Two_Product(adxtail, bdytail, ti1, ti0);
      Two_Product(bdxtail, adytail, tj1, tj0);
      Two_Two_Diff(ti1, ti0, tj1, tj0, abtt3, abtt[2], abtt[1], abtt[0]);
      abtt[3] = abtt3;
      abttlen = 4;
    } else {
      abt[0] = 0.0;
      abtlen = 1;
      abtt[0] = 0.0;
      abttlen = 1;
    }

    if (cdxtail != 0.0) {
      temp16alen = scale_expansion_zeroelim(cxtablen, cxtab, cdxtail, temp16a);
      cxtabtlen = scale_expansion_zeroelim(abtlen, abt, cdxtail, cxtabt);
      temp32alen = scale_expansion_zeroelim(cxtabtlen, cxtabt, 2.0 * cdx,
                                            temp32a);
      temp48len = fast_expansion_sum_zeroelim(temp16alen, temp16a,
                                              temp32alen, temp32a, temp48);
      finlength = fast_expansion_sum_zeroelim(finlength, finnow, temp48len,
                                              temp48, finother);
      finswap = finnow; finnow = finother; finother = finswap;
      if (adytail != 0.0) {
        temp8len = scale_expansion_zeroelim(4, bb, cdxtail, temp8);
        temp16alen = scale_expansion_zeroelim(temp8len, temp8, adytail,
                                              temp16a);
        finlength = fast_expansion_sum_zeroelim(finlength, finnow, temp16alen,
                                                temp16a, finother);
        finswap = finnow; finnow = finother; finother = finswap;
      }
      if (bdytail != 0.0) {
        temp8len = scale_expansion_zeroelim(4, aa, -cdxtail, temp8);
        temp16alen = scale_expansion_zeroelim(temp8len, temp8, bdytail,
                                              temp16a);
        finlength = fast_expansion_sum_zeroelim(finlength, finnow, temp16alen,
                                                temp16a, finother);
        finswap = finnow; finnow = finother; finother = finswap;
      }

      temp32alen = scale_expansion_zeroelim(cxtabtlen, cxtabt, cdxtail,
                                            temp32a);
      cxtabttlen = scale_expansion_zeroelim(abttlen, abtt, cdxtail, cxtabtt);
      temp16alen = scale_expansion_zeroelim(cxtabttlen, cxtabtt, 2.0 * cdx,
                                            temp16a);
      temp16blen = scale_expansion_zeroelim(cxtabttlen, cxtabtt, cdxtail,
                                            temp16b);
      temp32blen = fast_expansion_sum_zeroelim(temp16alen, temp16a,
                                              temp16blen, temp16b, temp32b);
      temp64len = fast_expansion_sum_zeroelim(temp32alen, temp32a,
                                              temp32blen, temp32b, temp64);
      finlength = fast_expansion_sum_zeroelim(finlength, finnow, temp64len,
                                              temp64, finother);
      finswap = finnow; finnow = finother; finother = finswap;
    }
    if (cdytail != 0.0) {
      temp16alen = scale_expansion_zeroelim(cytablen, cytab, cdytail, temp16a);
      cytabtlen = scale_expansion_zeroelim(abtlen, abt, cdytail, cytabt);
      temp32alen = scale_expansion_zeroelim(cytabtlen, cytabt, 2.0 * cdy,
                                            temp32a);
      temp48len = fast_expansion_sum_zeroelim(temp16alen, temp16a,
                                              temp32alen, temp32a, temp48);
      finlength = fast_expansion_sum_zeroelim(finlength, finnow, temp48len,
                                              temp48, finother);
      finswap = finnow; finnow = finother; finother = finswap;


      temp32alen = scale_expansion_zeroelim(cytabtlen, cytabt, cdytail,
                                            temp32a);
      cytabttlen = scale_expansion_zeroelim(abttlen, abtt, cdytail, cytabtt);
      temp16alen = scale_expansion_zeroelim(cytabttlen, cytabtt, 2.0 * cdy,
                                            temp16a);
      temp16blen = scale_expansion_zeroelim(cytabttlen, cytabtt, cdytail,
                                            temp16b);
      temp32blen = fast_expansion_sum_zeroelim(temp16alen, temp16a,
                                              temp16blen, temp16b, temp32b);
      temp64len = fast_expansion_sum_zeroelim(temp32alen, temp32a,
                                              temp32blen, temp32b, temp64);
      finlength = fast_expansion_sum_zeroelim(finlength, finnow, temp64len,
                                              temp64, finother);
      finswap = finnow; finnow = finother; finother = finswap;
    }
  }

  return finnow[finlength - 1];
}

#ifdef ANSI_DECLARATORS
REAL incircle(struct mesh *m, struct behavior *b,
              vertex pa, vertex pb, vertex pc, vertex pd)
#else
REAL incircle(m, b, pa, pb, pc, pd)
struct mesh *m;
struct behavior *b;
vertex pa;
vertex pb;
vertex pc;
vertex pd;
#endif

{
  REAL adx, bdx, cdx, ady, bdy, cdy;
  REAL bdxcdy, cdxbdy, cdxady, adxcdy, adxbdy, bdxady;
  REAL alift, blift, clift;
  REAL det;
  REAL permanent, errbound;

  m->incirclecount++;

  adx = pa[0] - pd[0];
  bdx = pb[0] - pd[0];
  cdx = pc[0] - pd[0];
  ady = pa[1] - pd[1];
  bdy = pb[1] - pd[1];
  cdy = pc[1] - pd[1];

  bdxcdy = bdx * cdy;
  cdxbdy = cdx * bdy;
  alift = adx * adx + ady * ady;

  cdxady = cdx * ady;
  adxcdy = adx * cdy;
  blift = bdx * bdx + bdy * bdy;

  adxbdy = adx * bdy;
  bdxady = bdx * ady;
  clift = cdx * cdx + cdy * cdy;

  det = alift * (bdxcdy - cdxbdy)
      + blift * (cdxady - adxcdy)
      + clift * (adxbdy - bdxady);

  if (b->noexact) {
    return det;
  }

  permanent = (Absolute(bdxcdy) + Absolute(cdxbdy)) * alift
            + (Absolute(cdxady) + Absolute(adxcdy)) * blift
            + (Absolute(adxbdy) + Absolute(bdxady)) * clift;
  errbound = iccerrboundA * permanent;
  if ((det > errbound) || (-det > errbound)) {
    return det;
  }

  return incircleadapt(pa, pb, pc, pd, permanent);
}


#ifdef ANSI_DECLARATORS
REAL orient3dadapt(vertex pa, vertex pb, vertex pc, vertex pd,
                   REAL aheight, REAL bheight, REAL cheight, REAL dheight,
                   REAL permanent)
#else
REAL orient3dadapt(pa, pb, pc, pd,
                   aheight, bheight, cheight, dheight, permanent)
vertex pa;
vertex pb;
vertex pc;
vertex pd;
REAL aheight;
REAL bheight;
REAL cheight;
REAL dheight;
REAL permanent;
#endif

{
  INEXACT REAL adx, bdx, cdx, ady, bdy, cdy, adheight, bdheight, cdheight;
  REAL det, errbound;

  INEXACT REAL bdxcdy1, cdxbdy1, cdxady1, adxcdy1, adxbdy1, bdxady1;
  REAL bdxcdy0, cdxbdy0, cdxady0, adxcdy0, adxbdy0, bdxady0;
  REAL bc[4], ca[4], ab[4];
  INEXACT REAL bc3, ca3, ab3;
  REAL adet[8], bdet[8], cdet[8];
  int alen, blen, clen;
  REAL abdet[16];
  int ablen;
  REAL *finnow, *finother, *finswap;
  REAL fin1[192], fin2[192];
  int finlength;

  REAL adxtail, bdxtail, cdxtail;
  REAL adytail, bdytail, cdytail;
  REAL adheighttail, bdheighttail, cdheighttail;
  INEXACT REAL at_blarge, at_clarge;
  INEXACT REAL bt_clarge, bt_alarge;
  INEXACT REAL ct_alarge, ct_blarge;
  REAL at_b[4], at_c[4], bt_c[4], bt_a[4], ct_a[4], ct_b[4];
  int at_blen, at_clen, bt_clen, bt_alen, ct_alen, ct_blen;
  INEXACT REAL bdxt_cdy1, cdxt_bdy1, cdxt_ady1;
  INEXACT REAL adxt_cdy1, adxt_bdy1, bdxt_ady1;
  REAL bdxt_cdy0, cdxt_bdy0, cdxt_ady0;
  REAL adxt_cdy0, adxt_bdy0, bdxt_ady0;
  INEXACT REAL bdyt_cdx1, cdyt_bdx1, cdyt_adx1;
  INEXACT REAL adyt_cdx1, adyt_bdx1, bdyt_adx1;
  REAL bdyt_cdx0, cdyt_bdx0, cdyt_adx0;
  REAL adyt_cdx0, adyt_bdx0, bdyt_adx0;
  REAL bct[8], cat[8], abt[8];
  int bctlen, catlen, abtlen;
  INEXACT REAL bdxt_cdyt1, cdxt_bdyt1, cdxt_adyt1;
  INEXACT REAL adxt_cdyt1, adxt_bdyt1, bdxt_adyt1;
  REAL bdxt_cdyt0, cdxt_bdyt0, cdxt_adyt0;
  REAL adxt_cdyt0, adxt_bdyt0, bdxt_adyt0;
  REAL u[4], v[12], w[16];
  INEXACT REAL u3;
  int vlength, wlength;
  REAL negate;

  INEXACT REAL bvirt;
  REAL avirt, bround, around;
  INEXACT REAL c;
  INEXACT REAL abig;
  REAL ahi, alo, bhi, blo;
  REAL err1, err2, err3;
  INEXACT REAL _i, _j, _k;
  REAL _0;

  adx = (REAL) (pa[0] - pd[0]);
  bdx = (REAL) (pb[0] - pd[0]);
  cdx = (REAL) (pc[0] - pd[0]);
  ady = (REAL) (pa[1] - pd[1]);
  bdy = (REAL) (pb[1] - pd[1]);
  cdy = (REAL) (pc[1] - pd[1]);
  adheight = (REAL) (aheight - dheight);
  bdheight = (REAL) (bheight - dheight);
  cdheight = (REAL) (cheight - dheight);

  Two_Product(bdx, cdy, bdxcdy1, bdxcdy0);
  Two_Product(cdx, bdy, cdxbdy1, cdxbdy0);
  Two_Two_Diff(bdxcdy1, bdxcdy0, cdxbdy1, cdxbdy0, bc3, bc[2], bc[1], bc[0]);
  bc[3] = bc3;
  alen = scale_expansion_zeroelim(4, bc, adheight, adet);

  Two_Product(cdx, ady, cdxady1, cdxady0);
  Two_Product(adx, cdy, adxcdy1, adxcdy0);
  Two_Two_Diff(cdxady1, cdxady0, adxcdy1, adxcdy0, ca3, ca[2], ca[1], ca[0]);
  ca[3] = ca3;
  blen = scale_expansion_zeroelim(4, ca, bdheight, bdet);

  Two_Product(adx, bdy, adxbdy1, adxbdy0);
  Two_Product(bdx, ady, bdxady1, bdxady0);
  Two_Two_Diff(adxbdy1, adxbdy0, bdxady1, bdxady0, ab3, ab[2], ab[1], ab[0]);
  ab[3] = ab3;
  clen = scale_expansion_zeroelim(4, ab, cdheight, cdet);

  ablen = fast_expansion_sum_zeroelim(alen, adet, blen, bdet, abdet);
  finlength = fast_expansion_sum_zeroelim(ablen, abdet, clen, cdet, fin1);

  det = estimate(finlength, fin1);
  errbound = o3derrboundB * permanent;
  if ((det >= errbound) || (-det >= errbound)) {
    return det;
  }

  Two_Diff_Tail(pa[0], pd[0], adx, adxtail);
  Two_Diff_Tail(pb[0], pd[0], bdx, bdxtail);
  Two_Diff_Tail(pc[0], pd[0], cdx, cdxtail);
  Two_Diff_Tail(pa[1], pd[1], ady, adytail);
  Two_Diff_Tail(pb[1], pd[1], bdy, bdytail);
  Two_Diff_Tail(pc[1], pd[1], cdy, cdytail);
  Two_Diff_Tail(aheight, dheight, adheight, adheighttail);
  Two_Diff_Tail(bheight, dheight, bdheight, bdheighttail);
  Two_Diff_Tail(cheight, dheight, cdheight, cdheighttail);

  if ((adxtail == 0.0) && (bdxtail == 0.0) && (cdxtail == 0.0) &&
      (adytail == 0.0) && (bdytail == 0.0) && (cdytail == 0.0) &&
      (adheighttail == 0.0) &&
      (bdheighttail == 0.0) &&
      (cdheighttail == 0.0)) {
    return det;
  }

  errbound = o3derrboundC * permanent + resulterrbound * Absolute(det);
  det += (adheight * ((bdx * cdytail + cdy * bdxtail) -
                      (bdy * cdxtail + cdx * bdytail)) +
          adheighttail * (bdx * cdy - bdy * cdx)) +
         (bdheight * ((cdx * adytail + ady * cdxtail) -
                      (cdy * adxtail + adx * cdytail)) +
          bdheighttail * (cdx * ady - cdy * adx)) +
         (cdheight * ((adx * bdytail + bdy * adxtail) -
                      (ady * bdxtail + bdx * adytail)) +
          cdheighttail * (adx * bdy - ady * bdx));
  if ((det >= errbound) || (-det >= errbound)) {
    return det;
  }

  finnow = fin1;
  finother = fin2;

  if (adxtail == 0.0) {
    if (adytail == 0.0) {
      at_b[0] = 0.0;
      at_blen = 1;
      at_c[0] = 0.0;
      at_clen = 1;
    } else {
      negate = -adytail;
      Two_Product(negate, bdx, at_blarge, at_b[0]);
      at_b[1] = at_blarge;
      at_blen = 2;
      Two_Product(adytail, cdx, at_clarge, at_c[0]);
      at_c[1] = at_clarge;
      at_clen = 2;
    }
  } else {
    if (adytail == 0.0) {
      Two_Product(adxtail, bdy, at_blarge, at_b[0]);
      at_b[1] = at_blarge;
      at_blen = 2;
      negate = -adxtail;
      Two_Product(negate, cdy, at_clarge, at_c[0]);
      at_c[1] = at_clarge;
      at_clen = 2;
    } else {
      Two_Product(adxtail, bdy, adxt_bdy1, adxt_bdy0);
      Two_Product(adytail, bdx, adyt_bdx1, adyt_bdx0);
      Two_Two_Diff(adxt_bdy1, adxt_bdy0, adyt_bdx1, adyt_bdx0,
                   at_blarge, at_b[2], at_b[1], at_b[0]);
      at_b[3] = at_blarge;
      at_blen = 4;
      Two_Product(adytail, cdx, adyt_cdx1, adyt_cdx0);
      Two_Product(adxtail, cdy, adxt_cdy1, adxt_cdy0);
      Two_Two_Diff(adyt_cdx1, adyt_cdx0, adxt_cdy1, adxt_cdy0,
                   at_clarge, at_c[2], at_c[1], at_c[0]);
      at_c[3] = at_clarge;
      at_clen = 4;
    }
  }
  if (bdxtail == 0.0) {
    if (bdytail == 0.0) {
      bt_c[0] = 0.0;
      bt_clen = 1;
      bt_a[0] = 0.0;
      bt_alen = 1;
    } else {
      negate = -bdytail;
      Two_Product(negate, cdx, bt_clarge, bt_c[0]);
      bt_c[1] = bt_clarge;
      bt_clen = 2;
      Two_Product(bdytail, adx, bt_alarge, bt_a[0]);
      bt_a[1] = bt_alarge;
      bt_alen = 2;
    }
  } else {
    if (bdytail == 0.0) {
      Two_Product(bdxtail, cdy, bt_clarge, bt_c[0]);
      bt_c[1] = bt_clarge;
      bt_clen = 2;
      negate = -bdxtail;
      Two_Product(negate, ady, bt_alarge, bt_a[0]);
      bt_a[1] = bt_alarge;
      bt_alen = 2;
    } else {
      Two_Product(bdxtail, cdy, bdxt_cdy1, bdxt_cdy0);
      Two_Product(bdytail, cdx, bdyt_cdx1, bdyt_cdx0);
      Two_Two_Diff(bdxt_cdy1, bdxt_cdy0, bdyt_cdx1, bdyt_cdx0,
                   bt_clarge, bt_c[2], bt_c[1], bt_c[0]);
      bt_c[3] = bt_clarge;
      bt_clen = 4;
      Two_Product(bdytail, adx, bdyt_adx1, bdyt_adx0);
      Two_Product(bdxtail, ady, bdxt_ady1, bdxt_ady0);
      Two_Two_Diff(bdyt_adx1, bdyt_adx0, bdxt_ady1, bdxt_ady0,
                  bt_alarge, bt_a[2], bt_a[1], bt_a[0]);
      bt_a[3] = bt_alarge;
      bt_alen = 4;
    }
  }
  if (cdxtail == 0.0) {
    if (cdytail == 0.0) {
      ct_a[0] = 0.0;
      ct_alen = 1;
      ct_b[0] = 0.0;
      ct_blen = 1;
    } else {
      negate = -cdytail;
      Two_Product(negate, adx, ct_alarge, ct_a[0]);
      ct_a[1] = ct_alarge;
      ct_alen = 2;
      Two_Product(cdytail, bdx, ct_blarge, ct_b[0]);
      ct_b[1] = ct_blarge;
      ct_blen = 2;
    }
  } else {
    if (cdytail == 0.0) {
      Two_Product(cdxtail, ady, ct_alarge, ct_a[0]);
      ct_a[1] = ct_alarge;
      ct_alen = 2;
      negate = -cdxtail;
      Two_Product(negate, bdy, ct_blarge, ct_b[0]);
      ct_b[1] = ct_blarge;
      ct_blen = 2;
    } else {
      Two_Product(cdxtail, ady, cdxt_ady1, cdxt_ady0);
      Two_Product(cdytail, adx, cdyt_adx1, cdyt_adx0);
      Two_Two_Diff(cdxt_ady1, cdxt_ady0, cdyt_adx1, cdyt_adx0,
                   ct_alarge, ct_a[2], ct_a[1], ct_a[0]);
      ct_a[3] = ct_alarge;
      ct_alen = 4;
      Two_Product(cdytail, bdx, cdyt_bdx1, cdyt_bdx0);
      Two_Product(cdxtail, bdy, cdxt_bdy1, cdxt_bdy0);
      Two_Two_Diff(cdyt_bdx1, cdyt_bdx0, cdxt_bdy1, cdxt_bdy0,
                   ct_blarge, ct_b[2], ct_b[1], ct_b[0]);
      ct_b[3] = ct_blarge;
      ct_blen = 4;
    }
  }

  bctlen = fast_expansion_sum_zeroelim(bt_clen, bt_c, ct_blen, ct_b, bct);
  wlength = scale_expansion_zeroelim(bctlen, bct, adheight, w);
  finlength = fast_expansion_sum_zeroelim(finlength, finnow, wlength, w,
                                          finother);
  finswap = finnow; finnow = finother; finother = finswap;

  catlen = fast_expansion_sum_zeroelim(ct_alen, ct_a, at_clen, at_c, cat);
  wlength = scale_expansion_zeroelim(catlen, cat, bdheight, w);
  finlength = fast_expansion_sum_zeroelim(finlength, finnow, wlength, w,
                                          finother);
  finswap = finnow; finnow = finother; finother = finswap;

  abtlen = fast_expansion_sum_zeroelim(at_blen, at_b, bt_alen, bt_a, abt);
  wlength = scale_expansion_zeroelim(abtlen, abt, cdheight, w);
  finlength = fast_expansion_sum_zeroelim(finlength, finnow, wlength, w,
                                          finother);
  finswap = finnow; finnow = finother; finother = finswap;

  if (adheighttail != 0.0) {
    vlength = scale_expansion_zeroelim(4, bc, adheighttail, v);
    finlength = fast_expansion_sum_zeroelim(finlength, finnow, vlength, v,
                                            finother);
    finswap = finnow; finnow = finother; finother = finswap;
  }
  if (bdheighttail != 0.0) {
    vlength = scale_expansion_zeroelim(4, ca, bdheighttail, v);
    finlength = fast_expansion_sum_zeroelim(finlength, finnow, vlength, v,
                                            finother);
    finswap = finnow; finnow = finother; finother = finswap;
  }
  if (cdheighttail != 0.0) {
    vlength = scale_expansion_zeroelim(4, ab, cdheighttail, v);
    finlength = fast_expansion_sum_zeroelim(finlength, finnow, vlength, v,
                                            finother);
    finswap = finnow; finnow = finother; finother = finswap;
  }

  if (adxtail != 0.0) {
    if (bdytail != 0.0) {
      Two_Product(adxtail, bdytail, adxt_bdyt1, adxt_bdyt0);
      Two_One_Product(adxt_bdyt1, adxt_bdyt0, cdheight, u3, u[2], u[1], u[0]);
      u[3] = u3;
      finlength = fast_expansion_sum_zeroelim(finlength, finnow, 4, u,
                                              finother);
      finswap = finnow; finnow = finother; finother = finswap;
      if (cdheighttail != 0.0) {
        Two_One_Product(adxt_bdyt1, adxt_bdyt0, cdheighttail,
                        u3, u[2], u[1], u[0]);
        u[3] = u3;
        finlength = fast_expansion_sum_zeroelim(finlength, finnow, 4, u,
                                                finother);
        finswap = finnow; finnow = finother; finother = finswap;
      }
    }
    if (cdytail != 0.0) {
      negate = -adxtail;
      Two_Product(negate, cdytail, adxt_cdyt1, adxt_cdyt0);
      Two_One_Product(adxt_cdyt1, adxt_cdyt0, bdheight, u3, u[2], u[1], u[0]);
      u[3] = u3;
      finlength = fast_expansion_sum_zeroelim(finlength, finnow, 4, u,
                                              finother);
      finswap = finnow; finnow = finother; finother = finswap;
      if (bdheighttail != 0.0) {
        Two_One_Product(adxt_cdyt1, adxt_cdyt0, bdheighttail,
                        u3, u[2], u[1], u[0]);
        u[3] = u3;
        finlength = fast_expansion_sum_zeroelim(finlength, finnow, 4, u,
                                                finother);
        finswap = finnow; finnow = finother; finother = finswap;
      }
    }
  }
  if (bdxtail != 0.0) {
    if (cdytail != 0.0) {
      Two_Product(bdxtail, cdytail, bdxt_cdyt1, bdxt_cdyt0);
      Two_One_Product(bdxt_cdyt1, bdxt_cdyt0, adheight, u3, u[2], u[1], u[0]);
      u[3] = u3;
      finlength = fast_expansion_sum_zeroelim(finlength, finnow, 4, u,
                                              finother);
      finswap = finnow; finnow = finother; finother = finswap;
      if (adheighttail != 0.0) {
        Two_One_Product(bdxt_cdyt1, bdxt_cdyt0, adheighttail,
                        u3, u[2], u[1], u[0]);
        u[3] = u3;
        finlength = fast_expansion_sum_zeroelim(finlength, finnow, 4, u,
                                                finother);
        finswap = finnow; finnow = finother; finother = finswap;
      }
    }
    if (adytail != 0.0) {
      negate = -bdxtail;
      Two_Product(negate, adytail, bdxt_adyt1, bdxt_adyt0);
      Two_One_Product(bdxt_adyt1, bdxt_adyt0, cdheight, u3, u[2], u[1], u[0]);
      u[3] = u3;
      finlength = fast_expansion_sum_zeroelim(finlength, finnow, 4, u,
                                              finother);
      finswap = finnow; finnow = finother; finother = finswap;
      if (cdheighttail != 0.0) {
        Two_One_Product(bdxt_adyt1, bdxt_adyt0, cdheighttail,
                        u3, u[2], u[1], u[0]);
        u[3] = u3;
        finlength = fast_expansion_sum_zeroelim(finlength, finnow, 4, u,
                                                finother);
        finswap = finnow; finnow = finother; finother = finswap;
      }
    }
  }
  if (cdxtail != 0.0) {
    if (adytail != 0.0) {
      Two_Product(cdxtail, adytail, cdxt_adyt1, cdxt_adyt0);
      Two_One_Product(cdxt_adyt1, cdxt_adyt0, bdheight, u3, u[2], u[1], u[0]);
      u[3] = u3;
      finlength = fast_expansion_sum_zeroelim(finlength, finnow, 4, u,
                                              finother);
      finswap = finnow; finnow = finother; finother = finswap;
      if (bdheighttail != 0.0) {
        Two_One_Product(cdxt_adyt1, cdxt_adyt0, bdheighttail,
                        u3, u[2], u[1], u[0]);
        u[3] = u3;
        finlength = fast_expansion_sum_zeroelim(finlength, finnow, 4, u,
                                                finother);
        finswap = finnow; finnow = finother; finother = finswap;
      }
    }
    if (bdytail != 0.0) {
      negate = -cdxtail;
      Two_Product(negate, bdytail, cdxt_bdyt1, cdxt_bdyt0);
      Two_One_Product(cdxt_bdyt1, cdxt_bdyt0, adheight, u3, u[2], u[1], u[0]);
      u[3] = u3;
      finlength = fast_expansion_sum_zeroelim(finlength, finnow, 4, u,
                                              finother);
      finswap = finnow; finnow = finother; finother = finswap;
      if (adheighttail != 0.0) {
        Two_One_Product(cdxt_bdyt1, cdxt_bdyt0, adheighttail,
                        u3, u[2], u[1], u[0]);
        u[3] = u3;
        finlength = fast_expansion_sum_zeroelim(finlength, finnow, 4, u,
                                                finother);
        finswap = finnow; finnow = finother; finother = finswap;
      }
    }
  }

  if (adheighttail != 0.0) {
    wlength = scale_expansion_zeroelim(bctlen, bct, adheighttail, w);
    finlength = fast_expansion_sum_zeroelim(finlength, finnow, wlength, w,
                                            finother);
    finswap = finnow; finnow = finother; finother = finswap;
  }
  if (bdheighttail != 0.0) {
    wlength = scale_expansion_zeroelim(catlen, cat, bdheighttail, w);
    finlength = fast_expansion_sum_zeroelim(finlength, finnow, wlength, w,
                                            finother);
    finswap = finnow; finnow = finother; finother = finswap;
  }
  if (cdheighttail != 0.0) {
    wlength = scale_expansion_zeroelim(abtlen, abt, cdheighttail, w);
    finlength = fast_expansion_sum_zeroelim(finlength, finnow, wlength, w,
                                            finother);
    finswap = finnow; finnow = finother; finother = finswap;
  }

  return finnow[finlength - 1];
}

#ifdef ANSI_DECLARATORS
REAL orient3d(struct mesh *m, struct behavior *b,
              vertex pa, vertex pb, vertex pc, vertex pd,
              REAL aheight, REAL bheight, REAL cheight, REAL dheight)
#else
REAL orient3d(m, b, pa, pb, pc, pd, aheight, bheight, cheight, dheight)
struct mesh *m;
struct behavior *b;
vertex pa;
vertex pb;
vertex pc;
vertex pd;
REAL aheight;
REAL bheight;
REAL cheight;
REAL dheight;
#endif

{
  REAL adx, bdx, cdx, ady, bdy, cdy, adheight, bdheight, cdheight;
  REAL bdxcdy, cdxbdy, cdxady, adxcdy, adxbdy, bdxady;
  REAL det;
  REAL permanent, errbound;

  m->orient3dcount++;

  adx = pa[0] - pd[0];
  bdx = pb[0] - pd[0];
  cdx = pc[0] - pd[0];
  ady = pa[1] - pd[1];
  bdy = pb[1] - pd[1];
  cdy = pc[1] - pd[1];
  adheight = aheight - dheight;
  bdheight = bheight - dheight;
  cdheight = cheight - dheight;

  bdxcdy = bdx * cdy;
  cdxbdy = cdx * bdy;

  cdxady = cdx * ady;
  adxcdy = adx * cdy;

  adxbdy = adx * bdy;
  bdxady = bdx * ady;

  det = adheight * (bdxcdy - cdxbdy)
      + bdheight * (cdxady - adxcdy)
      + cdheight * (adxbdy - bdxady);

  if (b->noexact) {
    return det;
  }

  permanent = (Absolute(bdxcdy) + Absolute(cdxbdy)) * Absolute(adheight)
            + (Absolute(cdxady) + Absolute(adxcdy)) * Absolute(bdheight)
            + (Absolute(adxbdy) + Absolute(bdxady)) * Absolute(cdheight);
  errbound = o3derrboundA * permanent;
  if ((det > errbound) || (-det > errbound)) {
    return det;
  }

  return orient3dadapt(pa, pb, pc, pd, aheight, bheight, cheight, dheight,
                       permanent);
}


#ifdef ANSI_DECLARATORS
REAL nonregular(struct mesh *m, struct behavior *b,
                vertex pa, vertex pb, vertex pc, vertex pd)
#else
REAL nonregular(m, b, pa, pb, pc, pd)
struct mesh *m;
struct behavior *b;
vertex pa;
vertex pb;
vertex pc;
vertex pd;
#endif

{
  if (b->weighted == 0) {
    return incircle(m, b, pa, pb, pc, pd);
  } else if (b->weighted == 1) {
    return orient3d(m, b, pa, pb, pc, pd,
                    pa[0] * pa[0] + pa[1] * pa[1] - pa[2],
                    pb[0] * pb[0] + pb[1] * pb[1] - pb[2],
                    pc[0] * pc[0] + pc[1] * pc[1] - pc[2],
                    pd[0] * pd[0] + pd[1] * pd[1] - pd[2]);
  } else {
    return orient3d(m, b, pa, pb, pc, pd, pa[2], pb[2], pc[2], pd[2]);
  }
}


#ifdef ANSI_DECLARATORS
void findcircumcenter(struct mesh *m, struct behavior *b,
                      vertex torg, vertex tdest, vertex tapex,
                      vertex circumcenter, REAL *xi, REAL *eta, int offcenter)
#else
void findcircumcenter(m, b, torg, tdest, tapex, circumcenter, xi, eta,
                      offcenter)
struct mesh *m;
struct behavior *b;
vertex torg;
vertex tdest;
vertex tapex;
vertex circumcenter;
REAL *xi;
REAL *eta;
int offcenter;
#endif

{
  REAL xdo, ydo, xao, yao;
  REAL dodist, aodist, dadist;
  REAL denominator;
  REAL dx, dy, dxoff, dyoff;

  m->circumcentercount++;

   xdo = tdest[0] - torg[0];
  ydo = tdest[1] - torg[1];
  xao = tapex[0] - torg[0];
  yao = tapex[1] - torg[1];
  dodist = xdo * xdo + ydo * ydo;
  aodist = xao * xao + yao * yao;
  dadist = (tdest[0] - tapex[0]) * (tdest[0] - tapex[0]) +
           (tdest[1] - tapex[1]) * (tdest[1] - tapex[1]);
  if (b->noexact) {
    denominator = 0.5 / (xdo * yao - xao * ydo);
  } else {
             denominator = 0.5 / counterclockwise(m, b, tdest, tapex, torg);
       m->counterclockcount--;
  }
  dx = (yao * dodist - ydo * aodist) * denominator;
  dy = (xdo * aodist - xao * dodist) * denominator;

       if ((dodist < aodist) && (dodist < dadist)) {
    if (offcenter && (b->offconstant > 0.0)) {
           dxoff = 0.5 * xdo - b->offconstant * ydo;
      dyoff = 0.5 * ydo + b->offconstant * xdo;
                if (dxoff * dxoff + dyoff * dyoff < dx * dx + dy * dy) {
        dx = dxoff;
        dy = dyoff;
      }
    }
  } else if (aodist < dadist) {
    if (offcenter && (b->offconstant > 0.0)) {
      dxoff = 0.5 * xao + b->offconstant * yao;
      dyoff = 0.5 * yao - b->offconstant * xao;
                if (dxoff * dxoff + dyoff * dyoff < dx * dx + dy * dy) {
        dx = dxoff;
        dy = dyoff;
      }
    }
  } else {
    if (offcenter && (b->offconstant > 0.0)) {
      dxoff = 0.5 * (tapex[0] - tdest[0]) -
              b->offconstant * (tapex[1] - tdest[1]);
      dyoff = 0.5 * (tapex[1] - tdest[1]) +
              b->offconstant * (tapex[0] - tdest[0]);
                if (dxoff * dxoff + dyoff * dyoff <
          (dx - xdo) * (dx - xdo) + (dy - ydo) * (dy - ydo)) {
        dx = xdo + dxoff;
        dy = ydo + dyoff;
      }
    }
  }

  circumcenter[0] = torg[0] + dx;
  circumcenter[1] = torg[1] + dy;

       *xi = (yao * dx - xao * dy) * (2.0 * denominator);
  *eta = (xdo * dy - ydo * dx) * (2.0 * denominator);
}



#ifdef ANSI_DECLARATORS
void triangleinit(struct mesh *m)
#else
void triangleinit(m)
struct mesh *m;
#endif

{
  poolzero(&m->vertices);
  poolzero(&m->triangles);
  poolzero(&m->subsegs);
  poolzero(&m->viri);
  poolzero(&m->badsubsegs);
  poolzero(&m->badtriangles);
  poolzero(&m->flipstackers);
  poolzero(&m->splaynodes);

  m->recenttri.tri = (triangle *) NULL;
  m->undeads = 0;
  m->samples = 1;
  m->checksegments = 0;
  m->checkquality = 0;
  m->incirclecount = m->counterclockcount = m->orient3dcount = 0;
  m->hyperbolacount = m->circletopcount = m->circumcentercount = 0;
  randomseed = 1;

  exactinit();
}


#ifdef ANSI_DECLARATORS
TRIULONG randomnation(unsigned int choices)
#else
TRIULONG randomnation(choices)
unsigned int choices;
#endif

{
  randomseed = (randomseed * 1366l + 150889l) % 714025l;
  return randomseed / (714025l / choices + 1);
}



#ifndef REDUCED

#ifdef ANSI_DECLARATORS
void checkmesh(struct mesh *m, struct behavior *b)
#else
void checkmesh(m, b)
struct mesh *m;
struct behavior *b;
#endif

{
  struct otri triangleloop;
  struct otri oppotri, oppooppotri;
  vertex triorg, tridest, triapex;
  vertex oppoorg, oppodest;
  int horrors;
  int saveexact;
  triangle ptr;

   saveexact = b->noexact;
  b->noexact = 0;
  if (!b->quiet) {
    printf("  Checking consistency of mesh...\n");
  }
  horrors = 0;
   traversalinit(&m->triangles);
  triangleloop.tri = triangletraverse(m);
  while (triangleloop.tri != (triangle *) NULL) {
       for (triangleloop.orient = 0; triangleloop.orient < 3;
         triangleloop.orient++) {
      org(triangleloop, triorg);
      dest(triangleloop, tridest);
      if (triangleloop.orient == 0) {
               apex(triangleloop, triapex);
        if (counterclockwise(m, b, triorg, tridest, triapex) <= 0.0) {
          printf("  !! !! Inverted ");
          printtriangle(m, b, &triangleloop);
          horrors++;
        }
      }
           sym(triangleloop, oppotri);
      if (oppotri.tri != m->dummytri) {
               sym(oppotri, oppooppotri);
        if ((triangleloop.tri != oppooppotri.tri)
            || (triangleloop.orient != oppooppotri.orient)) {
          printf("  !! !! Asymmetric triangle-triangle bond:\n");
          if (triangleloop.tri == oppooppotri.tri) {
            printf("   (Right triangle, wrong orientation)\n");
          }
          printf("    First ");
          printtriangle(m, b, &triangleloop);
          printf("    Second (nonreciprocating) ");
          printtriangle(m, b, &oppotri);
          horrors++;
        }
                      org(oppotri, oppoorg);
        dest(oppotri, oppodest);
        if ((triorg != oppodest) || (tridest != oppoorg)) {
          printf("  !! !! Mismatched edge coordinates between two triangles:\n"
                 );
          printf("    First mismatched ");
          printtriangle(m, b, &triangleloop);
          printf("    Second mismatched ");
          printtriangle(m, b, &oppotri);
          horrors++;
        }
      }
    }
    triangleloop.tri = triangletraverse(m);
  }
  if (horrors == 0) {
    if (!b->quiet) {
      printf("  In my studied opinion, the mesh appears to be consistent.\n");
    }
  } else if (horrors == 1) {
    printf("  !! !! !! !! Precisely one festering wound discovered.\n");
  } else {
    printf("  !! !! !! !! %d abominations witnessed.\n", horrors);
  }
   b->noexact = saveexact;
}

#endif


#ifndef REDUCED

#ifdef ANSI_DECLARATORS
void checkdelaunay(struct mesh *m, struct behavior *b)
#else
void checkdelaunay(m, b)
struct mesh *m;
struct behavior *b;
#endif

{
  struct otri triangleloop;
  struct otri oppotri;
  struct osub opposubseg;
  vertex triorg, tridest, triapex;
  vertex oppoapex;
  int shouldbedelaunay;
  int horrors;
  int saveexact;
  triangle ptr;
  subseg sptr;

   saveexact = b->noexact;
  b->noexact = 0;
  if (!b->quiet) {
    printf("  Checking Delaunay property of mesh...\n");
  }
  horrors = 0;
   traversalinit(&m->triangles);
  triangleloop.tri = triangletraverse(m);
  while (triangleloop.tri != (triangle *) NULL) {
       for (triangleloop.orient = 0; triangleloop.orient < 3;
         triangleloop.orient++) {
      org(triangleloop, triorg);
      dest(triangleloop, tridest);
      apex(triangleloop, triapex);
      sym(triangleloop, oppotri);
      apex(oppotri, oppoapex);
                     shouldbedelaunay = (oppotri.tri != m->dummytri) &&
            !deadtri(oppotri.tri) && (triangleloop.tri < oppotri.tri) &&
            (triorg != m->infvertex1) && (triorg != m->infvertex2) &&
            (triorg != m->infvertex3) &&
            (tridest != m->infvertex1) && (tridest != m->infvertex2) &&
            (tridest != m->infvertex3) &&
            (triapex != m->infvertex1) && (triapex != m->infvertex2) &&
            (triapex != m->infvertex3) &&
            (oppoapex != m->infvertex1) && (oppoapex != m->infvertex2) &&
            (oppoapex != m->infvertex3);
      if (m->checksegments && shouldbedelaunay) {
                      tspivot(triangleloop, opposubseg);
        if (opposubseg.ss != m->dummysub){
          shouldbedelaunay = 0;
        }
      }
      if (shouldbedelaunay) {
        if (nonregular(m, b, triorg, tridest, triapex, oppoapex) > 0.0) {
          if (!b->weighted) {
            printf("  !! !! Non-Delaunay pair of triangles:\n");
            printf("    First non-Delaunay ");
            printtriangle(m, b, &triangleloop);
            printf("    Second non-Delaunay ");
          } else {
            printf("  !! !! Non-regular pair of triangles:\n");
            printf("    First non-regular ");
            printtriangle(m, b, &triangleloop);
            printf("    Second non-regular ");
          }
          printtriangle(m, b, &oppotri);
          horrors++;
        }
      }
    }
    triangleloop.tri = triangletraverse(m);
  }
  if (horrors == 0) {
    if (!b->quiet) {
      printf(
  "  By virtue of my perceptive intelligence, I declare the mesh Delaunay.\n");
    }
  } else if (horrors == 1) {
    printf(
         "  !! !! !! !! Precisely one terrifying transgression identified.\n");
  } else {
    printf("  !! !! !! !! %d obscenities viewed with horror.\n", horrors);
  }
   b->noexact = saveexact;
}

#endif


#ifndef CDT_ONLY

#ifdef ANSI_DECLARATORS
void enqueuebadtriang(struct mesh *m, struct behavior *b,
                      struct badtriang *badtri)
#else
void enqueuebadtriang(m, b, badtri)
struct mesh *m;
struct behavior *b;
struct badtriang *badtri;
#endif

{
  REAL length, multiplier;
  int exponent, expincrement;
  int queuenumber;
  int posexponent;
  int i;

  if (b->verbose > 2) {
    printf("  Queueing bad triangle:\n");
    printf("    (%.12g, %.12g) (%.12g, %.12g) (%.12g, %.12g)\n",
           badtri->triangorg[0], badtri->triangorg[1],
           badtri->triangdest[0], badtri->triangdest[1],
           badtri->triangapex[0], badtri->triangapex[1]);
  }

    if (badtri->key >= 1.0) {
    length = badtri->key;
    posexponent = 1;
  } else {
          length = 1.0 / badtri->key;
    posexponent = 0;
  }
    exponent = 0;
  while (length > 2.0) {
       expincrement = 1;
    multiplier = 0.5;
    while (length * multiplier * multiplier > 1.0) {
      expincrement *= 2;
      multiplier *= multiplier;
    }
       exponent += expincrement;
    length *= multiplier;
  }
   exponent = 2.0 * exponent + (length > SQUAREROOTTWO);
     if (posexponent) {
    queuenumber = 2047 - exponent;
  } else {
    queuenumber = 2048 + exponent;
  }

   if (m->queuefront[queuenumber] == (struct badtriang *) NULL) {
          if (queuenumber > m->firstnonemptyq) {
           m->nextnonemptyq[queuenumber] = m->firstnonemptyq;
      m->firstnonemptyq = queuenumber;
    } else {
                i = queuenumber + 1;
      while (m->queuefront[i] == (struct badtriang *) NULL) {
        i++;
      }
           m->nextnonemptyq[queuenumber] = m->nextnonemptyq[i];
      m->nextnonemptyq[i] = queuenumber;
    }
       m->queuefront[queuenumber] = badtri;
  } else {
       m->queuetail[queuenumber]->nexttriang = badtri;
  }
   m->queuetail[queuenumber] = badtri;
   badtri->nexttriang = (struct badtriang *) NULL;
}

#endif


#ifndef CDT_ONLY

#ifdef ANSI_DECLARATORS
void enqueuebadtri(struct mesh *m, struct behavior *b, struct otri *enqtri,
                   REAL minedge, vertex enqapex, vertex enqorg, vertex enqdest)
#else
void enqueuebadtri(m, b, enqtri, minedge, enqapex, enqorg, enqdest)
struct mesh *m;
struct behavior *b;
struct otri *enqtri;
REAL minedge;
vertex enqapex;
vertex enqorg;
vertex enqdest;
#endif

{
  struct badtriang *newbad;

   newbad = (struct badtriang *) poolalloc(&m->badtriangles);
  newbad->poortri = encode(*enqtri);
  newbad->key = minedge;
  newbad->triangapex = enqapex;
  newbad->triangorg = enqorg;
  newbad->triangdest = enqdest;
  enqueuebadtriang(m, b, newbad);
}

#endif


#ifndef CDT_ONLY

#ifdef ANSI_DECLARATORS
struct badtriang *dequeuebadtriang(struct mesh *m)
#else
struct badtriang *dequeuebadtriang(m)
struct mesh *m;
#endif

{
  struct badtriang *result;

   if (m->firstnonemptyq < 0) {
    return (struct badtriang *) NULL;
  }
   result = m->queuefront[m->firstnonemptyq];
   m->queuefront[m->firstnonemptyq] = result->nexttriang;
    if (result == m->queuetail[m->firstnonemptyq]) {
    m->firstnonemptyq = m->nextnonemptyq[m->firstnonemptyq];
  }
  return result;
}

#endif


#ifndef CDT_ONLY

#ifdef ANSI_DECLARATORS
int checkseg4encroach(struct mesh *m, struct behavior *b,
                      struct osub *testsubseg)
#else
int checkseg4encroach(m, b, testsubseg)
struct mesh *m;
struct behavior *b;
struct osub *testsubseg;
#endif

{
  struct otri neighbortri;
  struct osub testsym;
  struct badsubseg *encroachedseg;
  REAL dotproduct;
  int encroached;
  int sides;
  vertex eorg, edest, eapex;
  triangle ptr;

  encroached = 0;
  sides = 0;

  sorg(*testsubseg, eorg);
  sdest(*testsubseg, edest);
   stpivot(*testsubseg, neighbortri);
   if (neighbortri.tri != m->dummytri) {
    sides++;
       apex(neighbortri, eapex);
                   dotproduct = (eorg[0] - eapex[0]) * (edest[0] - eapex[0]) +
                 (eorg[1] - eapex[1]) * (edest[1] - eapex[1]);
    if (dotproduct < 0.0) {
      if (b->conformdel ||
          (dotproduct * dotproduct >=
           (2.0 * b->goodangle - 1.0) * (2.0 * b->goodangle - 1.0) *
           ((eorg[0] - eapex[0]) * (eorg[0] - eapex[0]) +
            (eorg[1] - eapex[1]) * (eorg[1] - eapex[1])) *
           ((edest[0] - eapex[0]) * (edest[0] - eapex[0]) +
            (edest[1] - eapex[1]) * (edest[1] - eapex[1])))) {
        encroached = 1;
      }
    }
  }
   ssym(*testsubseg, testsym);
  stpivot(testsym, neighbortri);
   if (neighbortri.tri != m->dummytri) {
    sides++;
       apex(neighbortri, eapex);
          dotproduct = (eorg[0] - eapex[0]) * (edest[0] - eapex[0]) +
                 (eorg[1] - eapex[1]) * (edest[1] - eapex[1]);
    if (dotproduct < 0.0) {
      if (b->conformdel ||
          (dotproduct * dotproduct >=
           (2.0 * b->goodangle - 1.0) * (2.0 * b->goodangle - 1.0) *
           ((eorg[0] - eapex[0]) * (eorg[0] - eapex[0]) +
            (eorg[1] - eapex[1]) * (eorg[1] - eapex[1])) *
           ((edest[0] - eapex[0]) * (edest[0] - eapex[0]) +
            (edest[1] - eapex[1]) * (edest[1] - eapex[1])))) {
        encroached += 2;
      }
    }
  }

  if (encroached && (!b->nobisect || ((b->nobisect == 1) && (sides == 2)))) {
    if (b->verbose > 2) {
      printf(
        "  Queueing encroached subsegment (%.12g, %.12g) (%.12g, %.12g).\n",
        eorg[0], eorg[1], edest[0], edest[1]);
    }
          encroachedseg = (struct badsubseg *) poolalloc(&m->badsubsegs);
    if (encroached == 1) {
      encroachedseg->encsubseg = sencode(*testsubseg);
      encroachedseg->subsegorg = eorg;
      encroachedseg->subsegdest = edest;
    } else {
      encroachedseg->encsubseg = sencode(testsym);
      encroachedseg->subsegorg = edest;
      encroachedseg->subsegdest = eorg;
    }
  }

  return encroached;
}

#endif


#ifndef CDT_ONLY

#ifdef ANSI_DECLARATORS
void testtriangle(struct mesh *m, struct behavior *b, struct otri *testtri)
#else
void testtriangle(m, b, testtri)
struct mesh *m;
struct behavior *b;
struct otri *testtri;
#endif

{
  struct otri tri1, tri2;
  struct osub testsub;
  vertex torg, tdest, tapex;
  vertex base1, base2;
  vertex org1, dest1, org2, dest2;
  vertex joinvertex;
  REAL dxod, dyod, dxda, dyda, dxao, dyao;
  REAL dxod2, dyod2, dxda2, dyda2, dxao2, dyao2;
  REAL apexlen, orglen, destlen, minedge;
  REAL angle;
  REAL area;
  REAL dist1, dist2;
  subseg sptr;
  triangle ptr;

  org(*testtri, torg);
  dest(*testtri, tdest);
  apex(*testtri, tapex);
  dxod = torg[0] - tdest[0];
  dyod = torg[1] - tdest[1];
  dxda = tdest[0] - tapex[0];
  dyda = tdest[1] - tapex[1];
  dxao = tapex[0] - torg[0];
  dyao = tapex[1] - torg[1];
  dxod2 = dxod * dxod;
  dyod2 = dyod * dyod;
  dxda2 = dxda * dxda;
  dyda2 = dyda * dyda;
  dxao2 = dxao * dxao;
  dyao2 = dyao * dyao;
   apexlen = dxod2 + dyod2;
  orglen = dxda2 + dyda2;
  destlen = dxao2 + dyao2;

  if ((apexlen < orglen) && (apexlen < destlen)) {
       minedge = apexlen;
       angle = dxda * dxao + dyda * dyao;
    angle = angle * angle / (orglen * destlen);
    base1 = torg;
    base2 = tdest;
    otricopy(*testtri, tri1);
  } else if (orglen < destlen) {
       minedge = orglen;
       angle = dxod * dxao + dyod * dyao;
    angle = angle * angle / (apexlen * destlen);
    base1 = tdest;
    base2 = tapex;
    lnext(*testtri, tri1);
  } else {
       minedge = destlen;
       angle = dxod * dxda + dyod * dyda;
    angle = angle * angle / (apexlen * orglen);
    base1 = tapex;
    base2 = torg;
    lprev(*testtri, tri1);
  }

  if (b->vararea || b->fixedarea || b->usertest) {
       area = 0.5 * (dxod * dyda - dyod * dxda);
    if (b->fixedarea && (area > b->maxarea)) {
           enqueuebadtri(m, b, testtri, minedge, tapex, torg, tdest);
      return;
    }

       if ((b->vararea) && (area > areabound(*testtri)) &&
        (areabound(*testtri) > 0.0)) {
           enqueuebadtri(m, b, testtri, minedge, tapex, torg, tdest);
      return;
    }

    if (b->usertest) {
           if (triunsuitable(torg, tdest, tapex, area)) {
        enqueuebadtri(m, b, testtri, minedge, tapex, torg, tdest);
        return;
      }
    }
  }

   if (angle > b->goodangle) {
                               if ((vertextype(base1) == SEGMENTVERTEX) &&
        (vertextype(base2) == SEGMENTVERTEX)) {
                tspivot(tri1, testsub);
      if (testsub.ss == m->dummysub) {
               otricopy(tri1, tri2);
        do {
          oprevself(tri1);
          tspivot(tri1, testsub);
        } while (testsub.ss == m->dummysub);
               segorg(testsub, org1);
        segdest(testsub, dest1);
               do {
          dnextself(tri2);
          tspivot(tri2, testsub);
        } while (testsub.ss == m->dummysub);
               segorg(testsub, org2);
        segdest(testsub, dest2);
               joinvertex = (vertex) NULL;
        if ((dest1[0] == org2[0]) && (dest1[1] == org2[1])) {
          joinvertex = dest1;
        } else if ((org1[0] == dest2[0]) && (org1[1] == dest2[1])) {
          joinvertex = org1;
        }
        if (joinvertex != (vertex) NULL) {
                            dist1 = ((base1[0] - joinvertex[0]) * (base1[0] - joinvertex[0]) +
                   (base1[1] - joinvertex[1]) * (base1[1] - joinvertex[1]));
          dist2 = ((base2[0] - joinvertex[0]) * (base2[0] - joinvertex[0]) +
                   (base2[1] - joinvertex[1]) * (base2[1] - joinvertex[1]));
                   if ((dist1 < 1.001 * dist2) && (dist1 > 0.999 * dist2)) {
                       return;
          }
        }
      }
    }

       enqueuebadtri(m, b, testtri, minedge, tapex, torg, tdest);
  }
}

#endif




#ifdef ANSI_DECLARATORS
void makevertexmap(struct mesh *m, struct behavior *b)
#else
void makevertexmap(m, b)
struct mesh *m;
struct behavior *b;
#endif

{
  struct otri triangleloop;
  vertex triorg;

  if (b->verbose) {
    printf("    Constructing mapping from vertices to triangles.\n");
  }
  traversalinit(&m->triangles);
  triangleloop.tri = triangletraverse(m);
  while (triangleloop.tri != (triangle *) NULL) {
       for (triangleloop.orient = 0; triangleloop.orient < 3;
         triangleloop.orient++) {
      org(triangleloop, triorg);
      setvertex2tri(triorg, encode(triangleloop));
    }
    triangleloop.tri = triangletraverse(m);
  }
}


#ifdef ANSI_DECLARATORS
enum locateresult preciselocate(struct mesh *m, struct behavior *b,
                                vertex searchpoint, struct otri *searchtri,
                                int stopatsubsegment)
#else
enum locateresult preciselocate(m, b, searchpoint, searchtri, stopatsubsegment)
struct mesh *m;
struct behavior *b;
vertex searchpoint;
struct otri *searchtri;
int stopatsubsegment;
#endif

{
  struct otri backtracktri;
  struct osub checkedge;
  vertex forg, fdest, fapex;
  REAL orgorient, destorient;
  int moveleft;
  triangle ptr;
  subseg sptr;

  if (b->verbose > 2) {
    printf("  Searching for point (%.12g, %.12g).\n",
           searchpoint[0], searchpoint[1]);
  }
   org(*searchtri, forg);
  dest(*searchtri, fdest);
  apex(*searchtri, fapex);
  while (1) {
    if (b->verbose > 2) {
      printf("    At (%.12g, %.12g) (%.12g, %.12g) (%.12g, %.12g)\n",
             forg[0], forg[1], fdest[0], fdest[1], fapex[0], fapex[1]);
    }
       if ((fapex[0] == searchpoint[0]) && (fapex[1] == searchpoint[1])) {
      lprevself(*searchtri);
      return ONVERTEX;
    }
          destorient = counterclockwise(m, b, forg, fapex, searchpoint);
          orgorient = counterclockwise(m, b, fapex, fdest, searchpoint);
    if (destorient > 0.0) {
      if (orgorient > 0.0) {
                                           moveleft = (fapex[0] - searchpoint[0]) * (fdest[0] - forg[0]) +
                   (fapex[1] - searchpoint[1]) * (fdest[1] - forg[1]) > 0.0;
      } else {
        moveleft = 1;
      }
    } else {
      if (orgorient > 0.0) {
        moveleft = 0;
      } else {
                      if (destorient == 0.0) {
          lprevself(*searchtri);
          return ONEDGE;
        }
        if (orgorient == 0.0) {
          lnextself(*searchtri);
          return ONEDGE;
        }
        return INTRIANGLE;
      }
    }

             if (moveleft) {
      lprev(*searchtri, backtracktri);
      fdest = fapex;
    } else {
      lnext(*searchtri, backtracktri);
      forg = fapex;
    }
    sym(backtracktri, *searchtri);

    if (m->checksegments && stopatsubsegment) {
           tspivot(backtracktri, checkedge);
      if (checkedge.ss != m->dummysub) {
               otricopy(backtracktri, *searchtri);
        return OUTSIDE;
      }
    }
       if (searchtri->tri == m->dummytri) {
           otricopy(backtracktri, *searchtri);
      return OUTSIDE;
    }

    apex(*searchtri, fapex);
  }
}


#ifdef ANSI_DECLARATORS
enum locateresult locate(struct mesh *m, struct behavior *b,
                         vertex searchpoint, struct otri *searchtri)
#else
enum locateresult locate(m, b, searchpoint, searchtri)
struct mesh *m;
struct behavior *b;
vertex searchpoint;
struct otri *searchtri;
#endif

{
  VOID **sampleblock;
  char *firsttri;
  struct otri sampletri;
  vertex torg, tdest;
  TRIULONG alignptr;
  REAL searchdist, dist;
  REAL ahead;
  long samplesperblock, totalsamplesleft, samplesleft;
  long population, totalpopulation;
  triangle ptr;

  if (b->verbose > 2) {
    printf("  Randomly sampling for a triangle near point (%.12g, %.12g).\n",
           searchpoint[0], searchpoint[1]);
  }
    org(*searchtri, torg);
  searchdist = (searchpoint[0] - torg[0]) * (searchpoint[0] - torg[0]) +
               (searchpoint[1] - torg[1]) * (searchpoint[1] - torg[1]);
  if (b->verbose > 2) {
    printf("    Boundary triangle has origin (%.12g, %.12g).\n",
           torg[0], torg[1]);
  }

    if (m->recenttri.tri != (triangle *) NULL) {
    if (!deadtri(m->recenttri.tri)) {
      org(m->recenttri, torg);
      if ((torg[0] == searchpoint[0]) && (torg[1] == searchpoint[1])) {
        otricopy(m->recenttri, *searchtri);
        return ONVERTEX;
      }
      dist = (searchpoint[0] - torg[0]) * (searchpoint[0] - torg[0]) +
             (searchpoint[1] - torg[1]) * (searchpoint[1] - torg[1]);
      if (dist < searchdist) {
        otricopy(m->recenttri, *searchtri);
        searchdist = dist;
        if (b->verbose > 2) {
          printf("    Choosing recent triangle with origin (%.12g, %.12g).\n",
                 torg[0], torg[1]);
        }
      }
    }
  }

      while (SAMPLEFACTOR * m->samples * m->samples * m->samples <
         m->triangles.items) {
    m->samples++;
  }

      samplesperblock = (m->samples * TRIPERBLOCK - 1) / m->triangles.maxitems + 1;
    samplesleft = (m->samples * m->triangles.itemsfirstblock - 1) /
                m->triangles.maxitems + 1;
  totalsamplesleft = m->samples;
  population = m->triangles.itemsfirstblock;
  totalpopulation = m->triangles.maxitems;
  sampleblock = m->triangles.firstblock;
  sampletri.orient = 0;
  while (totalsamplesleft > 0) {
       if (population > totalpopulation) {
      population = totalpopulation;
    }
       alignptr = (TRIULONG) (sampleblock + 1);
    firsttri = (char *) (alignptr +
                         (TRIULONG) m->triangles.alignbytes -
                         (alignptr %
                          (TRIULONG) m->triangles.alignbytes));

       do {
      sampletri.tri = (triangle *) (firsttri +
                                    (randomnation((unsigned int) population) *
                                     m->triangles.itembytes));
      if (!deadtri(sampletri.tri)) {
        org(sampletri, torg);
        dist = (searchpoint[0] - torg[0]) * (searchpoint[0] - torg[0]) +
               (searchpoint[1] - torg[1]) * (searchpoint[1] - torg[1]);
        if (dist < searchdist) {
          otricopy(sampletri, *searchtri);
          searchdist = dist;
          if (b->verbose > 2) {
            printf("    Choosing triangle with origin (%.12g, %.12g).\n",
                   torg[0], torg[1]);
          }
        }
      }

      samplesleft--;
      totalsamplesleft--;
    } while ((samplesleft > 0) && (totalsamplesleft > 0));

    if (totalsamplesleft > 0) {
      sampleblock = (VOID **) *sampleblock;
      samplesleft = samplesperblock;
      totalpopulation -= population;
      population = TRIPERBLOCK;
    }
  }

   org(*searchtri, torg);
  dest(*searchtri, tdest);
   if ((torg[0] == searchpoint[0]) && (torg[1] == searchpoint[1])) {
    return ONVERTEX;
  }
  if ((tdest[0] == searchpoint[0]) && (tdest[1] == searchpoint[1])) {
    lnextself(*searchtri);
    return ONVERTEX;
  }
   ahead = counterclockwise(m, b, torg, tdest, searchpoint);
  if (ahead < 0.0) {
          symself(*searchtri);
  } else if (ahead == 0.0) {
       if (((torg[0] < searchpoint[0]) == (searchpoint[0] < tdest[0])) &&
        ((torg[1] < searchpoint[1]) == (searchpoint[1] < tdest[1]))) {
      return ONEDGE;
    }
  }
  return preciselocate(m, b, searchpoint, searchtri, 0);
}




#ifdef ANSI_DECLARATORS
void insertsubseg(struct mesh *m, struct behavior *b, struct otri *tri,
                  int subsegmark)
#else
void insertsubseg(m, b, tri, subsegmark)
struct mesh *m;
struct behavior *b;
struct otri *tri;
int subsegmark;
#endif

{
  struct otri oppotri;
  struct osub newsubseg;
  vertex triorg, tridest;
  triangle ptr;
  subseg sptr;

  org(*tri, triorg);
  dest(*tri, tridest);
   if (vertexmark(triorg) == 0) {
    setvertexmark(triorg, subsegmark);
  }
  if (vertexmark(tridest) == 0) {
    setvertexmark(tridest, subsegmark);
  }
   tspivot(*tri, newsubseg);
  if (newsubseg.ss == m->dummysub) {
       makesubseg(m, &newsubseg);
    setsorg(newsubseg, tridest);
    setsdest(newsubseg, triorg);
    setsegorg(newsubseg, tridest);
    setsegdest(newsubseg, triorg);
                tsbond(*tri, newsubseg);
    sym(*tri, oppotri);
    ssymself(newsubseg);
    tsbond(oppotri, newsubseg);
    setmark(newsubseg, subsegmark);
    if (b->verbose > 2) {
      printf("  Inserting new ");
      printsubseg(m, b, &newsubseg);
    }
  } else {
    if (mark(newsubseg) == 0) {
      setmark(newsubseg, subsegmark);
    }
  }
}



#ifdef ANSI_DECLARATORS
void flip(struct mesh *m, struct behavior *b, struct otri *flipedge)
#else
void flip(m, b, flipedge)
struct mesh *m;
struct behavior *b;
struct otri *flipedge;
#endif

{
  struct otri botleft, botright;
  struct otri topleft, topright;
  struct otri top;
  struct otri botlcasing, botrcasing;
  struct otri toplcasing, toprcasing;
  struct osub botlsubseg, botrsubseg;
  struct osub toplsubseg, toprsubseg;
  vertex leftvertex, rightvertex, botvertex;
  vertex farvertex;
  triangle ptr;
  subseg sptr;

   org(*flipedge, rightvertex);
  dest(*flipedge, leftvertex);
  apex(*flipedge, botvertex);
  sym(*flipedge, top);
#ifdef SELF_CHECK
  if (top.tri == m->dummytri) {
    printf("Internal error in flip():  Attempt to flip on boundary.\n");
    lnextself(*flipedge);
    return;
  }
  if (m->checksegments) {
    tspivot(*flipedge, toplsubseg);
    if (toplsubseg.ss != m->dummysub) {
      printf("Internal error in flip():  Attempt to flip a segment.\n");
      lnextself(*flipedge);
      return;
    }
  }
#endif
  apex(top, farvertex);

   lprev(top, topleft);
  sym(topleft, toplcasing);
  lnext(top, topright);
  sym(topright, toprcasing);
  lnext(*flipedge, botleft);
  sym(botleft, botlcasing);
  lprev(*flipedge, botright);
  sym(botright, botrcasing);
   bond(topleft, botlcasing);
  bond(botleft, botrcasing);
  bond(botright, toprcasing);
  bond(topright, toplcasing);

  if (m->checksegments) {
       tspivot(topleft, toplsubseg);
    tspivot(botleft, botlsubseg);
    tspivot(botright, botrsubseg);
    tspivot(topright, toprsubseg);
    if (toplsubseg.ss == m->dummysub) {
      tsdissolve(topright);
    } else {
      tsbond(topright, toplsubseg);
    }
    if (botlsubseg.ss == m->dummysub) {
      tsdissolve(topleft);
    } else {
      tsbond(topleft, botlsubseg);
    }
    if (botrsubseg.ss == m->dummysub) {
      tsdissolve(botleft);
    } else {
      tsbond(botleft, botrsubseg);
    }
    if (toprsubseg.ss == m->dummysub) {
      tsdissolve(botright);
    } else {
      tsbond(botright, toprsubseg);
    }
  }

   setorg(*flipedge, farvertex);
  setdest(*flipedge, botvertex);
  setapex(*flipedge, rightvertex);
  setorg(top, botvertex);
  setdest(top, farvertex);
  setapex(top, leftvertex);
  if (b->verbose > 2) {
    printf("  Edge flip results in left ");
    printtriangle(m, b, &top);
    printf("  and right ");
    printtriangle(m, b, flipedge);
  }
}


#ifdef ANSI_DECLARATORS
void unflip(struct mesh *m, struct behavior *b, struct otri *flipedge)
#else
void unflip(m, b, flipedge)
struct mesh *m;
struct behavior *b;
struct otri *flipedge;
#endif

{
  struct otri botleft, botright;
  struct otri topleft, topright;
  struct otri top;
  struct otri botlcasing, botrcasing;
  struct otri toplcasing, toprcasing;
  struct osub botlsubseg, botrsubseg;
  struct osub toplsubseg, toprsubseg;
  vertex leftvertex, rightvertex, botvertex;
  vertex farvertex;
  triangle ptr;
  subseg sptr;

   org(*flipedge, rightvertex);
  dest(*flipedge, leftvertex);
  apex(*flipedge, botvertex);
  sym(*flipedge, top);
#ifdef SELF_CHECK
  if (top.tri == m->dummytri) {
    printf("Internal error in unflip():  Attempt to flip on boundary.\n");
    lnextself(*flipedge);
    return;
  }
  if (m->checksegments) {
    tspivot(*flipedge, toplsubseg);
    if (toplsubseg.ss != m->dummysub) {
      printf("Internal error in unflip():  Attempt to flip a subsegment.\n");
      lnextself(*flipedge);
      return;
    }
  }
#endif
  apex(top, farvertex);

   lprev(top, topleft);
  sym(topleft, toplcasing);
  lnext(top, topright);
  sym(topright, toprcasing);
  lnext(*flipedge, botleft);
  sym(botleft, botlcasing);
  lprev(*flipedge, botright);
  sym(botright, botrcasing);
   bond(topleft, toprcasing);
  bond(botleft, toplcasing);
  bond(botright, botlcasing);
  bond(topright, botrcasing);

  if (m->checksegments) {
       tspivot(topleft, toplsubseg);
    tspivot(botleft, botlsubseg);
    tspivot(botright, botrsubseg);
    tspivot(topright, toprsubseg);
    if (toplsubseg.ss == m->dummysub) {
      tsdissolve(botleft);
    } else {
      tsbond(botleft, toplsubseg);
    }
    if (botlsubseg.ss == m->dummysub) {
      tsdissolve(botright);
    } else {
      tsbond(botright, botlsubseg);
    }
    if (botrsubseg.ss == m->dummysub) {
      tsdissolve(topright);
    } else {
      tsbond(topright, botrsubseg);
    }
    if (toprsubseg.ss == m->dummysub) {
      tsdissolve(topleft);
    } else {
      tsbond(topleft, toprsubseg);
    }
  }

   setorg(*flipedge, botvertex);
  setdest(*flipedge, farvertex);
  setapex(*flipedge, leftvertex);
  setorg(top, farvertex);
  setdest(top, botvertex);
  setapex(top, rightvertex);
  if (b->verbose > 2) {
    printf("  Edge unflip results in left ");
    printtriangle(m, b, flipedge);
    printf("  and right ");
    printtriangle(m, b, &top);
  }
}


#ifdef ANSI_DECLARATORS
enum insertvertexresult insertvertex(struct mesh *m, struct behavior *b,
                                     vertex newvertex, struct otri *searchtri,
                                     struct osub *splitseg,
                                     int segmentflaws, int triflaws)
#else
enum insertvertexresult insertvertex(m, b, newvertex, searchtri, splitseg,
                                     segmentflaws, triflaws)
struct mesh *m;
struct behavior *b;
vertex newvertex;
struct otri *searchtri;
struct osub *splitseg;
int segmentflaws;
int triflaws;
#endif

{
  struct otri horiz;
  struct otri top;
  struct otri botleft, botright;
  struct otri topleft, topright;
  struct otri newbotleft, newbotright;
  struct otri newtopright;
  struct otri botlcasing, botrcasing;
  struct otri toplcasing, toprcasing;
  struct otri testtri;
  struct osub botlsubseg, botrsubseg;
  struct osub toplsubseg, toprsubseg;
  struct osub brokensubseg;
  struct osub checksubseg;
  struct osub rightsubseg;
  struct osub newsubseg;
  struct badsubseg *encroached;
  struct flipstacker *newflip;
  vertex first;
  vertex leftvertex, rightvertex, botvertex, topvertex, farvertex;
  vertex segmentorg, segmentdest;
  REAL attrib;
  REAL area;
  enum insertvertexresult success;
  enum locateresult intersect;
  int doflip;
  int mirrorflag;
  int enq;
  int i;
  triangle ptr;
  subseg sptr;

  if (b->verbose > 1) {
    printf("  Inserting (%.12g, %.12g).\n", newvertex[0], newvertex[1]);
  }

  if (splitseg == (struct osub *) NULL) {
          if (searchtri->tri == m->dummytri) {
           horiz.tri = m->dummytri;
      horiz.orient = 0;
      symself(horiz);
           intersect = locate(m, b, newvertex, &horiz);
    } else {
           otricopy(*searchtri, horiz);
      intersect = preciselocate(m, b, newvertex, &horiz, 1);
    }
  } else {
          otricopy(*searchtri, horiz);
    intersect = ONEDGE;
  }

  if (intersect == ONVERTEX) {
          otricopy(horiz, *searchtri);
    otricopy(horiz, m->recenttri);
    return DUPLICATEVERTEX;
  }
  if ((intersect == ONEDGE) || (intersect == OUTSIDE)) {
       if (m->checksegments && (splitseg == (struct osub *) NULL)) {
           tspivot(horiz, brokensubseg);
      if (brokensubseg.ss != m->dummysub) {
               if (segmentflaws) {
          enq = b->nobisect != 2;
          if (enq && (b->nobisect == 1)) {
                                  sym(horiz, testtri);
            enq = testtri.tri != m->dummytri;
          }
          if (enq) {
                       encroached = (struct badsubseg *) poolalloc(&m->badsubsegs);
            encroached->encsubseg = sencode(brokensubseg);
            sorg(brokensubseg, encroached->subsegorg);
            sdest(brokensubseg, encroached->subsegdest);
            if (b->verbose > 2) {
              printf(
          "  Queueing encroached subsegment (%.12g, %.12g) (%.12g, %.12g).\n",
                     encroached->subsegorg[0], encroached->subsegorg[1],
                     encroached->subsegdest[0], encroached->subsegdest[1]);
            }
          }
        }
                      otricopy(horiz, *searchtri);
        otricopy(horiz, m->recenttri);
        return VIOLATINGVERTEX;
      }
    }

          lprev(horiz, botright);
    sym(botright, botrcasing);
    sym(horiz, topright);
       mirrorflag = topright.tri != m->dummytri;
    if (mirrorflag) {
      lnextself(topright);
      sym(topright, toprcasing);
      maketriangle(m, b, &newtopright);
    } else {
           m->hullsize++;
    }
    maketriangle(m, b, &newbotright);

       org(horiz, rightvertex);
    dest(horiz, leftvertex);
    apex(horiz, botvertex);
    setorg(newbotright, botvertex);
    setdest(newbotright, rightvertex);
    setapex(newbotright, newvertex);
    setorg(horiz, newvertex);
    for (i = 0; i < m->eextras; i++) {
           setelemattribute(newbotright, i, elemattribute(botright, i));
    }
    if (b->vararea) {
           setareabound(newbotright, areabound(botright));
    }
    if (mirrorflag) {
      dest(topright, topvertex);
      setorg(newtopright, rightvertex);
      setdest(newtopright, topvertex);
      setapex(newtopright, newvertex);
      setorg(topright, newvertex);
      for (i = 0; i < m->eextras; i++) {
               setelemattribute(newtopright, i, elemattribute(topright, i));
      }
      if (b->vararea) {
               setareabound(newtopright, areabound(topright));
      }
    }

          if (m->checksegments) {
      tspivot(botright, botrsubseg);
      if (botrsubseg.ss != m->dummysub) {
        tsdissolve(botright);
        tsbond(newbotright, botrsubseg);
      }
      if (mirrorflag) {
        tspivot(topright, toprsubseg);
        if (toprsubseg.ss != m->dummysub) {
          tsdissolve(topright);
          tsbond(newtopright, toprsubseg);
        }
      }
    }

       bond(newbotright, botrcasing);
    lprevself(newbotright);
    bond(newbotright, botright);
    lprevself(newbotright);
    if (mirrorflag) {
      bond(newtopright, toprcasing);
      lnextself(newtopright);
      bond(newtopright, topright);
      lnextself(newtopright);
      bond(newtopright, newbotright);
    }

    if (splitseg != (struct osub *) NULL) {
           setsdest(*splitseg, newvertex);
      segorg(*splitseg, segmentorg);
      segdest(*splitseg, segmentdest);
      ssymself(*splitseg);
      spivot(*splitseg, rightsubseg);
      insertsubseg(m, b, &newbotright, mark(*splitseg));
      tspivot(newbotright, newsubseg);
      setsegorg(newsubseg, segmentorg);
      setsegdest(newsubseg, segmentdest);
      sbond(*splitseg, newsubseg);
      ssymself(newsubseg);
      sbond(newsubseg, rightsubseg);
      ssymself(*splitseg);
                if (vertexmark(newvertex) == 0) {
        setvertexmark(newvertex, mark(*splitseg));
      }
    }

    if (m->checkquality) {
      poolrestart(&m->flipstackers);
      m->lastflip = (struct flipstacker *) poolalloc(&m->flipstackers);
      m->lastflip->flippedtri = encode(horiz);
      m->lastflip->prevflip = (struct flipstacker *) &insertvertex;
    }

#ifdef SELF_CHECK
    if (counterclockwise(m, b, rightvertex, leftvertex, botvertex) < 0.0) {
      printf("Internal error in insertvertex():\n");
      printf(
            "  Clockwise triangle prior to edge vertex insertion (bottom).\n");
    }
    if (mirrorflag) {
      if (counterclockwise(m, b, leftvertex, rightvertex, topvertex) < 0.0) {
        printf("Internal error in insertvertex():\n");
        printf("  Clockwise triangle prior to edge vertex insertion (top).\n");
      }
      if (counterclockwise(m, b, rightvertex, topvertex, newvertex) < 0.0) {
        printf("Internal error in insertvertex():\n");
        printf(
            "  Clockwise triangle after edge vertex insertion (top right).\n");
      }
      if (counterclockwise(m, b, topvertex, leftvertex, newvertex) < 0.0) {
        printf("Internal error in insertvertex():\n");
        printf(
            "  Clockwise triangle after edge vertex insertion (top left).\n");
      }
    }
    if (counterclockwise(m, b, leftvertex, botvertex, newvertex) < 0.0) {
      printf("Internal error in insertvertex():\n");
      printf(
          "  Clockwise triangle after edge vertex insertion (bottom left).\n");
    }
    if (counterclockwise(m, b, botvertex, rightvertex, newvertex) < 0.0) {
      printf("Internal error in insertvertex():\n");
      printf(
        "  Clockwise triangle after edge vertex insertion (bottom right).\n");
    }
#endif
    if (b->verbose > 2) {
      printf("  Updating bottom left ");
      printtriangle(m, b, &botright);
      if (mirrorflag) {
        printf("  Updating top left ");
        printtriangle(m, b, &topright);
        printf("  Creating top right ");
        printtriangle(m, b, &newtopright);
      }
      printf("  Creating bottom right ");
      printtriangle(m, b, &newbotright);
    }

          lnextself(horiz);
  } else {
       lnext(horiz, botleft);
    lprev(horiz, botright);
    sym(botleft, botlcasing);
    sym(botright, botrcasing);
    maketriangle(m, b, &newbotleft);
    maketriangle(m, b, &newbotright);

       org(horiz, rightvertex);
    dest(horiz, leftvertex);
    apex(horiz, botvertex);
    setorg(newbotleft, leftvertex);
    setdest(newbotleft, botvertex);
    setapex(newbotleft, newvertex);
    setorg(newbotright, botvertex);
    setdest(newbotright, rightvertex);
    setapex(newbotright, newvertex);
    setapex(horiz, newvertex);
    for (i = 0; i < m->eextras; i++) {
           attrib = elemattribute(horiz, i);
      setelemattribute(newbotleft, i, attrib);
      setelemattribute(newbotright, i, attrib);
    }
    if (b->vararea) {
           area = areabound(horiz);
      setareabound(newbotleft, area);
      setareabound(newbotright, area);
    }

          if (m->checksegments) {
      tspivot(botleft, botlsubseg);
      if (botlsubseg.ss != m->dummysub) {
        tsdissolve(botleft);
        tsbond(newbotleft, botlsubseg);
      }
      tspivot(botright, botrsubseg);
      if (botrsubseg.ss != m->dummysub) {
        tsdissolve(botright);
        tsbond(newbotright, botrsubseg);
      }
    }

       bond(newbotleft, botlcasing);
    bond(newbotright, botrcasing);
    lnextself(newbotleft);
    lprevself(newbotright);
    bond(newbotleft, newbotright);
    lnextself(newbotleft);
    bond(botleft, newbotleft);
    lprevself(newbotright);
    bond(botright, newbotright);

    if (m->checkquality) {
      poolrestart(&m->flipstackers);
      m->lastflip = (struct flipstacker *) poolalloc(&m->flipstackers);
      m->lastflip->flippedtri = encode(horiz);
      m->lastflip->prevflip = (struct flipstacker *) NULL;
    }

#ifdef SELF_CHECK
    if (counterclockwise(m, b, rightvertex, leftvertex, botvertex) < 0.0) {
      printf("Internal error in insertvertex():\n");
      printf("  Clockwise triangle prior to vertex insertion.\n");
    }
    if (counterclockwise(m, b, rightvertex, leftvertex, newvertex) < 0.0) {
      printf("Internal error in insertvertex():\n");
      printf("  Clockwise triangle after vertex insertion (top).\n");
    }
    if (counterclockwise(m, b, leftvertex, botvertex, newvertex) < 0.0) {
      printf("Internal error in insertvertex():\n");
      printf("  Clockwise triangle after vertex insertion (left).\n");
    }
    if (counterclockwise(m, b, botvertex, rightvertex, newvertex) < 0.0) {
      printf("Internal error in insertvertex():\n");
      printf("  Clockwise triangle after vertex insertion (right).\n");
    }
#endif
    if (b->verbose > 2) {
      printf("  Updating top ");
      printtriangle(m, b, &horiz);
      printf("  Creating left ");
      printtriangle(m, b, &newbotleft);
      printf("  Creating right ");
      printtriangle(m, b, &newbotright);
    }
  }

    success = SUCCESSFULVERTEX;
      org(horiz, first);
  rightvertex = first;
  dest(horiz, leftvertex);
   while (1) {
       doflip = 1;

    if (m->checksegments) {
           tspivot(horiz, checksubseg);
      if (checksubseg.ss != m->dummysub) {
               doflip = 0;
#ifndef CDT_ONLY
        if (segmentflaws) {
                   if (checkseg4encroach(m, b, &checksubseg)) {
            success = ENCROACHINGVERTEX;
          }
        }
#endif
      }
    }

    if (doflip) {
           sym(horiz, top);
      if (top.tri == m->dummytri) {
               doflip = 0;
      } else {
               apex(top, farvertex);
                                           if ((leftvertex == m->infvertex1) || (leftvertex == m->infvertex2) ||
            (leftvertex == m->infvertex3)) {
                                              doflip = counterclockwise(m, b, newvertex, rightvertex, farvertex)
                   > 0.0;
        } else if ((rightvertex == m->infvertex1) ||
                   (rightvertex == m->infvertex2) ||
                   (rightvertex == m->infvertex3)) {
                                              doflip = counterclockwise(m, b, farvertex, leftvertex, newvertex)
                   > 0.0;
        } else if ((farvertex == m->infvertex1) ||
                   (farvertex == m->infvertex2) ||
                   (farvertex == m->infvertex3)) {
                            doflip = 0;
        } else {
                   doflip = incircle(m, b, leftvertex, newvertex, rightvertex,
                            farvertex) > 0.0;
        }
        if (doflip) {
                                     lprev(top, topleft);
          sym(topleft, toplcasing);
          lnext(top, topright);
          sym(topright, toprcasing);
          lnext(horiz, botleft);
          sym(botleft, botlcasing);
          lprev(horiz, botright);
          sym(botright, botrcasing);
                   bond(topleft, botlcasing);
          bond(botleft, botrcasing);
          bond(botright, toprcasing);
          bond(topright, toplcasing);
          if (m->checksegments) {
                       tspivot(topleft, toplsubseg);
            tspivot(botleft, botlsubseg);
            tspivot(botright, botrsubseg);
            tspivot(topright, toprsubseg);
            if (toplsubseg.ss == m->dummysub) {
              tsdissolve(topright);
            } else {
              tsbond(topright, toplsubseg);
            }
            if (botlsubseg.ss == m->dummysub) {
              tsdissolve(topleft);
            } else {
              tsbond(topleft, botlsubseg);
            }
            if (botrsubseg.ss == m->dummysub) {
              tsdissolve(botleft);
            } else {
              tsbond(botleft, botrsubseg);
            }
            if (toprsubseg.ss == m->dummysub) {
              tsdissolve(botright);
            } else {
              tsbond(botright, toprsubseg);
            }
          }
                   setorg(horiz, farvertex);
          setdest(horiz, newvertex);
          setapex(horiz, rightvertex);
          setorg(top, newvertex);
          setdest(top, farvertex);
          setapex(top, leftvertex);
          for (i = 0; i < m->eextras; i++) {
                       attrib = 0.5 * (elemattribute(top, i) + elemattribute(horiz, i));
            setelemattribute(top, i, attrib);
            setelemattribute(horiz, i, attrib);
          }
          if (b->vararea) {
            if ((areabound(top) <= 0.0) || (areabound(horiz) <= 0.0)) {
              area = -1.0;
            } else {
                                                     area = 0.5 * (areabound(top) + areabound(horiz));
            }
            setareabound(top, area);
            setareabound(horiz, area);
          }

          if (m->checkquality) {
            newflip = (struct flipstacker *) poolalloc(&m->flipstackers);
            newflip->flippedtri = encode(horiz);
            newflip->prevflip = m->lastflip;
            m->lastflip = newflip;
          }

#ifdef SELF_CHECK
          if (newvertex != (vertex) NULL) {
            if (counterclockwise(m, b, leftvertex, newvertex, rightvertex) <
                0.0) {
              printf("Internal error in insertvertex():\n");
              printf("  Clockwise triangle prior to edge flip (bottom).\n");
            }
                                            if (counterclockwise(m, b, farvertex, leftvertex, newvertex) <
                0.0) {
              printf("Internal error in insertvertex():\n");
              printf("  Clockwise triangle after edge flip (left).\n");
            }
            if (counterclockwise(m, b, newvertex, rightvertex, farvertex) <
                0.0) {
              printf("Internal error in insertvertex():\n");
              printf("  Clockwise triangle after edge flip (right).\n");
            }
          }
#endif
          if (b->verbose > 2) {
            printf("  Edge flip results in left ");
            lnextself(topleft);
            printtriangle(m, b, &topleft);
            printf("  and right ");
            printtriangle(m, b, &horiz);
          }
                                     lprevself(horiz);
          leftvertex = farvertex;
        }
      }
    }
    if (!doflip) {
     #ifndef CDT_ONLY
      if (triflaws) {
               testtriangle(m, b, &horiz);
      }
#endif
           lnextself(horiz);
      sym(horiz, testtri);
                     if ((leftvertex == first) || (testtri.tri == m->dummytri)) {
               lnext(horiz, *searchtri);
        lnext(horiz, m->recenttri);
        return success;
      }
           lnext(testtri, horiz);
      rightvertex = leftvertex;
      dest(horiz, leftvertex);
    }
  }
}


#ifdef ANSI_DECLARATORS
void triangulatepolygon(struct mesh *m, struct behavior *b,
                        struct otri *firstedge, struct otri *lastedge,
                        int edgecount, int doflip, int triflaws)
#else
void triangulatepolygon(m, b, firstedge, lastedge, edgecount, doflip, triflaws)
struct mesh *m;
struct behavior *b;
struct otri *firstedge;
struct otri *lastedge;
int edgecount;
int doflip;
int triflaws;
#endif

{
  struct otri testtri;
  struct otri besttri;
  struct otri tempedge;
  vertex leftbasevertex, rightbasevertex;
  vertex testvertex;
  vertex bestvertex;
  int bestnumber;
  int i;
  triangle ptr;

   apex(*lastedge, leftbasevertex);
  dest(*firstedge, rightbasevertex);
  if (b->verbose > 2) {
    printf("  Triangulating interior polygon at edge\n");
    printf("    (%.12g, %.12g) (%.12g, %.12g)\n", leftbasevertex[0],
           leftbasevertex[1], rightbasevertex[0], rightbasevertex[1]);
  }
   onext(*firstedge, besttri);
  dest(besttri, bestvertex);
  otricopy(besttri, testtri);
  bestnumber = 1;
  for (i = 2; i <= edgecount - 2; i++) {
    onextself(testtri);
    dest(testtri, testvertex);
       if (incircle(m, b, leftbasevertex, rightbasevertex, bestvertex,
                 testvertex) > 0.0) {
      otricopy(testtri, besttri);
      bestvertex = testvertex;
      bestnumber = i;
    }
  }
  if (b->verbose > 2) {
    printf("    Connecting edge to (%.12g, %.12g)\n", bestvertex[0],
           bestvertex[1]);
  }
  if (bestnumber > 1) {
       oprev(besttri, tempedge);
    triangulatepolygon(m, b, firstedge, &tempedge, bestnumber + 1, 1,
                       triflaws);
  }
  if (bestnumber < edgecount - 2) {
       sym(besttri, tempedge);
    triangulatepolygon(m, b, &besttri, lastedge, edgecount - bestnumber, 1,
                       triflaws);
       sym(tempedge, besttri);
  }
  if (doflip) {
       flip(m, b, &besttri);
#ifndef CDT_ONLY
    if (triflaws) {
           sym(besttri, testtri);
      testtriangle(m, b, &testtri);
    }
#endif
  }
   otricopy(besttri, *lastedge);
}


#ifndef CDT_ONLY

#ifdef ANSI_DECLARATORS
void deletevertex(struct mesh *m, struct behavior *b, struct otri *deltri)
#else
void deletevertex(m, b, deltri)
struct mesh *m;
struct behavior *b;
struct otri *deltri;
#endif

{
  struct otri countingtri;
  struct otri firstedge, lastedge;
  struct otri deltriright;
  struct otri lefttri, righttri;
  struct otri leftcasing, rightcasing;
  struct osub leftsubseg, rightsubseg;
  vertex delvertex;
  vertex neworg;
  int edgecount;
  triangle ptr;
  subseg sptr;

  org(*deltri, delvertex);
  if (b->verbose > 1) {
    printf("  Deleting (%.12g, %.12g).\n", delvertex[0], delvertex[1]);
  }
  vertexdealloc(m, delvertex);

   onext(*deltri, countingtri);
  edgecount = 1;
  while (!otriequal(*deltri, countingtri)) {
#ifdef SELF_CHECK
    if (countingtri.tri == m->dummytri) {
      printf("Internal error in deletevertex():\n");
      printf("  Attempt to delete boundary vertex.\n");
      internalerror();
    }
#endif
    edgecount++;
    onextself(countingtri);
  }

#ifdef SELF_CHECK
  if (edgecount < 3) {
    printf("Internal error in deletevertex():\n  Vertex has degree %d.\n",
           edgecount);
    internalerror();
  }
#endif
  if (edgecount > 3) {
             onext(*deltri, firstedge);
    oprev(*deltri, lastedge);
    triangulatepolygon(m, b, &firstedge, &lastedge, edgecount, 0,
                       !b->nobisect);
  }
   lprev(*deltri, deltriright);
  dnext(*deltri, lefttri);
  sym(lefttri, leftcasing);
  oprev(deltriright, righttri);
  sym(righttri, rightcasing);
  bond(*deltri, leftcasing);
  bond(deltriright, rightcasing);
  tspivot(lefttri, leftsubseg);
  if (leftsubseg.ss != m->dummysub) {
    tsbond(*deltri, leftsubseg);
  }
  tspivot(righttri, rightsubseg);
  if (rightsubseg.ss != m->dummysub) {
    tsbond(deltriright, rightsubseg);
  }

   org(lefttri, neworg);
  setorg(*deltri, neworg);
  if (!b->nobisect) {
    testtriangle(m, b, deltri);
  }

   triangledealloc(m, lefttri.tri);
  triangledealloc(m, righttri.tri);
}

#endif


#ifndef CDT_ONLY

#ifdef ANSI_DECLARATORS
void undovertex(struct mesh *m, struct behavior *b)
#else
void undovertex(m, b)
struct mesh *m;
struct behavior *b;
#endif

{
  struct otri fliptri;
  struct otri botleft, botright, topright;
  struct otri botlcasing, botrcasing, toprcasing;
  struct otri gluetri;
  struct osub botlsubseg, botrsubseg, toprsubseg;
  vertex botvertex, rightvertex;
  triangle ptr;
  subseg sptr;

    while (m->lastflip != (struct flipstacker *) NULL) {
       decode(m->lastflip->flippedtri, fliptri);

                if (m->lastflip->prevflip == (struct flipstacker *) NULL) {
                dprev(fliptri, botleft);
      lnextself(botleft);
      onext(fliptri, botright);
      lprevself(botright);
      sym(botleft, botlcasing);
      sym(botright, botrcasing);
      dest(botleft, botvertex);

      setapex(fliptri, botvertex);
      lnextself(fliptri);
      bond(fliptri, botlcasing);
      tspivot(botleft, botlsubseg);
      tsbond(fliptri, botlsubseg);
      lnextself(fliptri);
      bond(fliptri, botrcasing);
      tspivot(botright, botrsubseg);
      tsbond(fliptri, botrsubseg);

           triangledealloc(m, botleft.tri);
      triangledealloc(m, botright.tri);
    } else if (m->lastflip->prevflip == (struct flipstacker *) &insertvertex) {
                lprev(fliptri, gluetri);
      sym(gluetri, botright);
      lnextself(botright);
      sym(botright, botrcasing);
      dest(botright, rightvertex);

      setorg(fliptri, rightvertex);
      bond(gluetri, botrcasing);
      tspivot(botright, botrsubseg);
      tsbond(gluetri, botrsubseg);

           triangledealloc(m, botright.tri);

      sym(fliptri, gluetri);
      if (gluetri.tri != m->dummytri) {
        lnextself(gluetri);
        dnext(gluetri, topright);
        sym(topright, toprcasing);

        setorg(gluetri, rightvertex);
        bond(gluetri, toprcasing);
        tspivot(topright, toprsubseg);
        tsbond(gluetri, toprsubseg);

               triangledealloc(m, topright.tri);
      }

           m->lastflip->prevflip = (struct flipstacker *) NULL;
    } else {
           unflip(m, b, &fliptri);
    }

       m->lastflip = m->lastflip->prevflip;
  }
}

#endif





#ifdef ANSI_DECLARATORS
void vertexsort(vertex *sortarray, int arraysize)
#else
void vertexsort(sortarray, arraysize)
vertex *sortarray;
int arraysize;
#endif

{
  int left, right;
  int pivot;
  REAL pivotx, pivoty;
  vertex temp;

  if (arraysize == 2) {
       if ((sortarray[0][0] > sortarray[1][0]) ||
        ((sortarray[0][0] == sortarray[1][0]) &&
         (sortarray[0][1] > sortarray[1][1]))) {
      temp = sortarray[1];
      sortarray[1] = sortarray[0];
      sortarray[0] = temp;
    }
    return;
  }
   pivot = (int) randomnation((unsigned int) arraysize);
  pivotx = sortarray[pivot][0];
  pivoty = sortarray[pivot][1];
   left = -1;
  right = arraysize;
  while (left < right) {
       do {
      left++;
    } while ((left <= right) && ((sortarray[left][0] < pivotx) ||
                                 ((sortarray[left][0] == pivotx) &&
                                  (sortarray[left][1] < pivoty))));
       do {
      right--;
    } while ((left <= right) && ((sortarray[right][0] > pivotx) ||
                                 ((sortarray[right][0] == pivotx) &&
                                  (sortarray[right][1] > pivoty))));
    if (left < right) {
           temp = sortarray[left];
      sortarray[left] = sortarray[right];
      sortarray[right] = temp;
    }
  }
  if (left > 1) {
       vertexsort(sortarray, left);
  }
  if (right < arraysize - 2) {
       vertexsort(&sortarray[right + 1], arraysize - right - 1);
  }
}


#ifdef ANSI_DECLARATORS
void vertexmedian(vertex *sortarray, int arraysize, int median, int axis)
#else
void vertexmedian(sortarray, arraysize, median, axis)
vertex *sortarray;
int arraysize;
int median;
int axis;
#endif

{
  int left, right;
  int pivot;
  REAL pivot1, pivot2;
  vertex temp;

  if (arraysize == 2) {
       if ((sortarray[0][axis] > sortarray[1][axis]) ||
        ((sortarray[0][axis] == sortarray[1][axis]) &&
         (sortarray[0][1 - axis] > sortarray[1][1 - axis]))) {
      temp = sortarray[1];
      sortarray[1] = sortarray[0];
      sortarray[0] = temp;
    }
    return;
  }
   pivot = (int) randomnation((unsigned int) arraysize);
  pivot1 = sortarray[pivot][axis];
  pivot2 = sortarray[pivot][1 - axis];
   left = -1;
  right = arraysize;
  while (left < right) {
       do {
      left++;
    } while ((left <= right) && ((sortarray[left][axis] < pivot1) ||
                                 ((sortarray[left][axis] == pivot1) &&
                                  (sortarray[left][1 - axis] < pivot2))));
       do {
      right--;
    } while ((left <= right) && ((sortarray[right][axis] > pivot1) ||
                                 ((sortarray[right][axis] == pivot1) &&
                                  (sortarray[right][1 - axis] > pivot2))));
    if (left < right) {
           temp = sortarray[left];
      sortarray[left] = sortarray[right];
      sortarray[right] = temp;
    }
  }
    if (left > median) {
       vertexmedian(sortarray, left, median, axis);
  }
  if (right < median - 1) {
       vertexmedian(&sortarray[right + 1], arraysize - right - 1,
                 median - right - 1, axis);
  }
}


#ifdef ANSI_DECLARATORS
void alternateaxes(vertex *sortarray, int arraysize, int axis)
#else
void alternateaxes(sortarray, arraysize, axis)
vertex *sortarray;
int arraysize;
int axis;
#endif

{
  int divider;

  divider = arraysize >> 1;
  if (arraysize <= 3) {
          axis = 0;
  }
   vertexmedian(sortarray, arraysize, divider, axis);
   if (arraysize - divider >= 2) {
    if (divider >= 2) {
      alternateaxes(sortarray, divider, 1 - axis);
    }
    alternateaxes(&sortarray[divider], arraysize - divider, 1 - axis);
  }
}


#ifdef ANSI_DECLARATORS
void mergehulls(struct mesh *m, struct behavior *b, struct otri *farleft,
                struct otri *innerleft, struct otri *innerright,
                struct otri *farright, int axis)
#else
void mergehulls(m, b, farleft, innerleft, innerright, farright, axis)
struct mesh *m;
struct behavior *b;
struct otri *farleft;
struct otri *innerleft;
struct otri *innerright;
struct otri *farright;
int axis;
#endif

{
  struct otri leftcand, rightcand;
  struct otri baseedge;
  struct otri nextedge;
  struct otri sidecasing, topcasing, outercasing;
  struct otri checkedge;
  vertex innerleftdest;
  vertex innerrightorg;
  vertex innerleftapex, innerrightapex;
  vertex farleftpt, farrightpt;
  vertex farleftapex, farrightapex;
  vertex lowerleft, lowerright;
  vertex upperleft, upperright;
  vertex nextapex;
  vertex checkvertex;
  int changemade;
  int badedge;
  int leftfinished, rightfinished;
  triangle ptr;

  dest(*innerleft, innerleftdest);
  apex(*innerleft, innerleftapex);
  org(*innerright, innerrightorg);
  apex(*innerright, innerrightapex);
   if (b->dwyer && (axis == 1)) {
    org(*farleft, farleftpt);
    apex(*farleft, farleftapex);
    dest(*farright, farrightpt);
    apex(*farright, farrightapex);
             while (farleftapex[1] < farleftpt[1]) {
      lnextself(*farleft);
      symself(*farleft);
      farleftpt = farleftapex;
      apex(*farleft, farleftapex);
    }
    sym(*innerleft, checkedge);
    apex(checkedge, checkvertex);
    while (checkvertex[1] > innerleftdest[1]) {
      lnext(checkedge, *innerleft);
      innerleftapex = innerleftdest;
      innerleftdest = checkvertex;
      sym(*innerleft, checkedge);
      apex(checkedge, checkvertex);
    }
    while (innerrightapex[1] < innerrightorg[1]) {
      lnextself(*innerright);
      symself(*innerright);
      innerrightorg = innerrightapex;
      apex(*innerright, innerrightapex);
    }
    sym(*farright, checkedge);
    apex(checkedge, checkvertex);
    while (checkvertex[1] > farrightpt[1]) {
      lnext(checkedge, *farright);
      farrightapex = farrightpt;
      farrightpt = checkvertex;
      sym(*farright, checkedge);
      apex(checkedge, checkvertex);
    }
  }
   do {
    changemade = 0;
       if (counterclockwise(m, b, innerleftdest, innerleftapex, innerrightorg) >
        0.0) {
      lprevself(*innerleft);
      symself(*innerleft);
      innerleftdest = innerleftapex;
      apex(*innerleft, innerleftapex);
      changemade = 1;
    }
       if (counterclockwise(m, b, innerrightapex, innerrightorg, innerleftdest) >
        0.0) {
      lnextself(*innerright);
      symself(*innerright);
      innerrightorg = innerrightapex;
      apex(*innerright, innerrightapex);
      changemade = 1;
    }
  } while (changemade);
   sym(*innerleft, leftcand);
  sym(*innerright, rightcand);
   maketriangle(m, b, &baseedge);
   bond(baseedge, *innerleft);
  lnextself(baseedge);
  bond(baseedge, *innerright);
  lnextself(baseedge);
  setorg(baseedge, innerrightorg);
  setdest(baseedge, innerleftdest);
   if (b->verbose > 2) {
    printf("  Creating base bounding ");
    printtriangle(m, b, &baseedge);
  }
   org(*farleft, farleftpt);
  if (innerleftdest == farleftpt) {
    lnext(baseedge, *farleft);
  }
  dest(*farright, farrightpt);
  if (innerrightorg == farrightpt) {
    lprev(baseedge, *farright);
  }
   lowerleft = innerleftdest;
  lowerright = innerrightorg;
   apex(leftcand, upperleft);
  apex(rightcand, upperright);
   while (1) {
                leftfinished = counterclockwise(m, b, upperleft, lowerleft, lowerright) <=
                   0.0;
    rightfinished = counterclockwise(m, b, upperright, lowerleft, lowerright)
                 <= 0.0;
    if (leftfinished && rightfinished) {
           maketriangle(m, b, &nextedge);
      setorg(nextedge, lowerleft);
      setdest(nextedge, lowerright);
                bond(nextedge, baseedge);
      lnextself(nextedge);
      bond(nextedge, rightcand);
      lnextself(nextedge);
      bond(nextedge, leftcand);
      if (b->verbose > 2) {
        printf("  Creating top bounding ");
        printtriangle(m, b, &nextedge);
      }
           if (b->dwyer && (axis == 1)) {
        org(*farleft, farleftpt);
        apex(*farleft, farleftapex);
        dest(*farright, farrightpt);
        apex(*farright, farrightapex);
        sym(*farleft, checkedge);
        apex(checkedge, checkvertex);
                             while (checkvertex[0] < farleftpt[0]) {
          lprev(checkedge, *farleft);
          farleftapex = farleftpt;
          farleftpt = checkvertex;
          sym(*farleft, checkedge);
          apex(checkedge, checkvertex);
        }
        while (farrightapex[0] > farrightpt[0]) {
          lprevself(*farright);
          symself(*farright);
          farrightpt = farrightapex;
          apex(*farright, farrightapex);
        }
      }
      return;
    }
       if (!leftfinished) {
           lprev(leftcand, nextedge);
      symself(nextedge);
      apex(nextedge, nextapex);
                if (nextapex != (vertex) NULL) {
               badedge = incircle(m, b, lowerleft, lowerright, upperleft, nextapex) >
                  0.0;
        while (badedge) {
                            lnextself(nextedge);
          sym(nextedge, topcasing);
          lnextself(nextedge);
          sym(nextedge, sidecasing);
          bond(nextedge, topcasing);
          bond(leftcand, sidecasing);
          lnextself(leftcand);
          sym(leftcand, outercasing);
          lprevself(nextedge);
          bond(nextedge, outercasing);
                   setorg(leftcand, lowerleft);
          setdest(leftcand, NULL);
          setapex(leftcand, nextapex);
          setorg(nextedge, NULL);
          setdest(nextedge, upperleft);
          setapex(nextedge, nextapex);
                   upperleft = nextapex;
                   otricopy(sidecasing, nextedge);
          apex(nextedge, nextapex);
          if (nextapex != (vertex) NULL) {
                       badedge = incircle(m, b, lowerleft, lowerright, upperleft,
                               nextapex) > 0.0;
          } else {
                       badedge = 0;
          }
        }
      }
    }
       if (!rightfinished) {
           lnext(rightcand, nextedge);
      symself(nextedge);
      apex(nextedge, nextapex);
                if (nextapex != (vertex) NULL) {
               badedge = incircle(m, b, lowerleft, lowerright, upperright, nextapex) >
                  0.0;
        while (badedge) {
                            lprevself(nextedge);
          sym(nextedge, topcasing);
          lprevself(nextedge);
          sym(nextedge, sidecasing);
          bond(nextedge, topcasing);
          bond(rightcand, sidecasing);
          lprevself(rightcand);
          sym(rightcand, outercasing);
          lnextself(nextedge);
          bond(nextedge, outercasing);
                   setorg(rightcand, NULL);
          setdest(rightcand, lowerright);
          setapex(rightcand, nextapex);
          setorg(nextedge, upperright);
          setdest(nextedge, NULL);
          setapex(nextedge, nextapex);
                   upperright = nextapex;
                   otricopy(sidecasing, nextedge);
          apex(nextedge, nextapex);
          if (nextapex != (vertex) NULL) {
                       badedge = incircle(m, b, lowerleft, lowerright, upperright,
                               nextapex) > 0.0;
          } else {
                       badedge = 0;
          }
        }
      }
    }
    if (leftfinished || (!rightfinished &&
           (incircle(m, b, upperleft, lowerleft, lowerright, upperright) >
            0.0))) {
                bond(baseedge, rightcand);
      lprev(rightcand, baseedge);
      setdest(baseedge, lowerleft);
      lowerright = upperright;
      sym(baseedge, rightcand);
      apex(rightcand, upperright);
    } else {
                bond(baseedge, leftcand);
      lnext(leftcand, baseedge);
      setorg(baseedge, lowerright);
      lowerleft = upperleft;
      sym(baseedge, leftcand);
      apex(leftcand, upperleft);
    }
    if (b->verbose > 2) {
      printf("  Connecting ");
      printtriangle(m, b, &baseedge);
    }
  }
}


#ifdef ANSI_DECLARATORS
void divconqrecurse(struct mesh *m, struct behavior *b, vertex *sortarray,
                    int vertices, int axis,
                    struct otri *farleft, struct otri *farright)
#else
void divconqrecurse(m, b, sortarray, vertices, axis, farleft, farright)
struct mesh *m;
struct behavior *b;
vertex *sortarray;
int vertices;
int axis;
struct otri *farleft;
struct otri *farright;
#endif

{
  struct otri midtri, tri1, tri2, tri3;
  struct otri innerleft, innerright;
  REAL area;
  int divider;

  if (b->verbose > 2) {
    printf("  Triangulating %d vertices.\n", vertices);
  }
  if (vertices == 2) {
          maketriangle(m, b, farleft);
    setorg(*farleft, sortarray[0]);
    setdest(*farleft, sortarray[1]);
       maketriangle(m, b, farright);
    setorg(*farright, sortarray[1]);
    setdest(*farright, sortarray[0]);
       bond(*farleft, *farright);
    lprevself(*farleft);
    lnextself(*farright);
    bond(*farleft, *farright);
    lprevself(*farleft);
    lnextself(*farright);
    bond(*farleft, *farright);
    if (b->verbose > 2) {
      printf("  Creating ");
      printtriangle(m, b, farleft);
      printf("  Creating ");
      printtriangle(m, b, farright);
    }
       lprev(*farright, *farleft);
    return;
  } else if (vertices == 3) {
             maketriangle(m, b, &midtri);
    maketriangle(m, b, &tri1);
    maketriangle(m, b, &tri2);
    maketriangle(m, b, &tri3);
    area = counterclockwise(m, b, sortarray[0], sortarray[1], sortarray[2]);
    if (area == 0.0) {
           setorg(midtri, sortarray[0]);
      setdest(midtri, sortarray[1]);
      setorg(tri1, sortarray[1]);
      setdest(tri1, sortarray[0]);
      setorg(tri2, sortarray[2]);
      setdest(tri2, sortarray[1]);
      setorg(tri3, sortarray[1]);
      setdest(tri3, sortarray[2]);
           bond(midtri, tri1);
      bond(tri2, tri3);
      lnextself(midtri);
      lprevself(tri1);
      lnextself(tri2);
      lprevself(tri3);
      bond(midtri, tri3);
      bond(tri1, tri2);
      lnextself(midtri);
      lprevself(tri1);
      lnextself(tri2);
      lprevself(tri3);
      bond(midtri, tri1);
      bond(tri2, tri3);
           otricopy(tri1, *farleft);
           otricopy(tri2, *farright);
    } else {
                setorg(midtri, sortarray[0]);
      setdest(tri1, sortarray[0]);
      setorg(tri3, sortarray[0]);
           if (area > 0.0) {
               setdest(midtri, sortarray[1]);
        setorg(tri1, sortarray[1]);
        setdest(tri2, sortarray[1]);
        setapex(midtri, sortarray[2]);
        setorg(tri2, sortarray[2]);
        setdest(tri3, sortarray[2]);
      } else {
               setdest(midtri, sortarray[2]);
        setorg(tri1, sortarray[2]);
        setdest(tri2, sortarray[2]);
        setapex(midtri, sortarray[1]);
        setorg(tri2, sortarray[1]);
        setdest(tri3, sortarray[1]);
      }
           bond(midtri, tri1);
      lnextself(midtri);
      bond(midtri, tri2);
      lnextself(midtri);
      bond(midtri, tri3);
      lprevself(tri1);
      lnextself(tri2);
      bond(tri1, tri2);
      lprevself(tri1);
      lprevself(tri3);
      bond(tri1, tri3);
      lnextself(tri2);
      lprevself(tri3);
      bond(tri2, tri3);
           otricopy(tri1, *farleft);
           if (area > 0.0) {
        otricopy(tri2, *farright);
      } else {
        lnext(*farleft, *farright);
      }
    }
    if (b->verbose > 2) {
      printf("  Creating ");
      printtriangle(m, b, &midtri);
      printf("  Creating ");
      printtriangle(m, b, &tri1);
      printf("  Creating ");
      printtriangle(m, b, &tri2);
      printf("  Creating ");
      printtriangle(m, b, &tri3);
    }
    return;
  } else {
       divider = vertices >> 1;
       divconqrecurse(m, b, sortarray, divider, 1 - axis, farleft, &innerleft);
    divconqrecurse(m, b, &sortarray[divider], vertices - divider, 1 - axis,
                   &innerright, farright);
    if (b->verbose > 1) {
      printf("  Joining triangulations with %d and %d vertices.\n", divider,
             vertices - divider);
    }
       mergehulls(m, b, farleft, &innerleft, &innerright, farright, axis);
  }
}

#ifdef ANSI_DECLARATORS
long removeghosts(struct mesh *m, struct behavior *b, struct otri *startghost)
#else
long removeghosts(m, b, startghost)
struct mesh *m;
struct behavior *b;
struct otri *startghost;
#endif

{
  struct otri searchedge;
  struct otri dissolveedge;
  struct otri deadtriangle;
  vertex markorg;
  long hullsize;
  triangle ptr;

  if (b->verbose) {
    printf("  Removing ghost triangles.\n");
  }
   lprev(*startghost, searchedge);
  symself(searchedge);
  m->dummytri[0] = encode(searchedge);
   otricopy(*startghost, dissolveedge);
  hullsize = 0;
  do {
    hullsize++;
    lnext(dissolveedge, deadtriangle);
    lprevself(dissolveedge);
    symself(dissolveedge);
          if (!b->poly) {
           if (dissolveedge.tri != m->dummytri) {
        org(dissolveedge, markorg);
        if (vertexmark(markorg) == 0) {
          setvertexmark(markorg, 1);
        }
      }
    }
       dissolve(dissolveedge);
       sym(deadtriangle, dissolveedge);
       triangledealloc(m, deadtriangle.tri);
  } while (!otriequal(dissolveedge, *startghost));
  return hullsize;
}


#ifdef ANSI_DECLARATORS
long divconqdelaunay(struct mesh *m, struct behavior *b)
#else
long divconqdelaunay(m, b)
struct mesh *m;
struct behavior *b;
#endif

{
  vertex *sortarray;
  struct otri hullleft, hullright;
  int divider;
  int i, j;

  if (b->verbose) {
    printf("  Sorting vertices.\n");
  }

   sortarray = (vertex *) trimalloc(m->invertices * (int) sizeof(vertex));
  traversalinit(&m->vertices);
  for (i = 0; i < m->invertices; i++) {
    sortarray[i] = vertextraverse(m);
  }
   vertexsort(sortarray, m->invertices);
   i = 0;
  for (j = 1; j < m->invertices; j++) {
    if ((sortarray[i][0] == sortarray[j][0])
        && (sortarray[i][1] == sortarray[j][1])) {
      if (!b->quiet) {
        printf(
"Warning:  A duplicate vertex at (%.12g, %.12g) appeared and was ignored.\n",
               sortarray[j][0], sortarray[j][1]);
      }
      setvertextype(sortarray[j], UNDEADVERTEX);
      m->undeads++;
    } else {
      i++;
      sortarray[i] = sortarray[j];
    }
  }
  i++;
  if (b->dwyer) {
       divider = i >> 1;
    if (i - divider >= 2) {
      if (divider >= 2) {
        alternateaxes(sortarray, divider, 1);
      }
      alternateaxes(&sortarray[divider], i - divider, 1);
    }
  }

  if (b->verbose) {
    printf("  Forming triangulation.\n");
  }

   divconqrecurse(m, b, sortarray, i, 0, &hullleft, &hullright);
  trifree((VOID *) sortarray);

  return removeghosts(m, b, &hullleft);
}




#ifndef REDUCED

#ifdef ANSI_DECLARATORS
void boundingbox(struct mesh *m, struct behavior *b)
#else
void boundingbox(m, b)
struct mesh *m;
struct behavior *b;
#endif

{
  struct otri inftri;
  REAL width;

  if (b->verbose) {
    printf("  Creating triangular bounding box.\n");
  }
   width = m->xmax - m->xmin;
  if (m->ymax - m->ymin > width) {
    width = m->ymax - m->ymin;
  }
  if (width == 0.0) {
    width = 1.0;
  }
   m->infvertex1 = (vertex) trimalloc(m->vertices.itembytes);
  m->infvertex2 = (vertex) trimalloc(m->vertices.itembytes);
  m->infvertex3 = (vertex) trimalloc(m->vertices.itembytes);
  m->infvertex1[0] = m->xmin - 50.0 * width;
  m->infvertex1[1] = m->ymin - 40.0 * width;
  m->infvertex2[0] = m->xmax + 50.0 * width;
  m->infvertex2[1] = m->ymin - 40.0 * width;
  m->infvertex3[0] = 0.5 * (m->xmin + m->xmax);
  m->infvertex3[1] = m->ymax + 60.0 * width;

   maketriangle(m, b, &inftri);
  setorg(inftri, m->infvertex1);
  setdest(inftri, m->infvertex2);
  setapex(inftri, m->infvertex3);
    m->dummytri[0] = (triangle) inftri.tri;
  if (b->verbose > 2) {
    printf("  Creating ");
    printtriangle(m, b, &inftri);
  }
}

#endif


#ifndef REDUCED

#ifdef ANSI_DECLARATORS
long removebox(struct mesh *m, struct behavior *b)
#else
long removebox(m, b)
struct mesh *m;
struct behavior *b;
#endif

{
  struct otri deadtriangle;
  struct otri searchedge;
  struct otri checkedge;
  struct otri nextedge, finaledge, dissolveedge;
  vertex markorg;
  long hullsize;
  triangle ptr;

  if (b->verbose) {
    printf("  Removing triangular bounding box.\n");
  }
   nextedge.tri = m->dummytri;
  nextedge.orient = 0;
  symself(nextedge);
   lprev(nextedge, finaledge);
  lnextself(nextedge);
  symself(nextedge);
    lprev(nextedge, searchedge);
  symself(searchedge);
    lnext(nextedge, checkedge);
  symself(checkedge);
  if (checkedge.tri == m->dummytri) {
             lprevself(searchedge);
    symself(searchedge);
  }
    m->dummytri[0] = encode(searchedge);
  hullsize = -2l;
  while (!otriequal(nextedge, finaledge)) {
    hullsize++;
    lprev(nextedge, dissolveedge);
    symself(dissolveedge);
          if (!b->poly) {
                          if (dissolveedge.tri != m->dummytri) {
        org(dissolveedge, markorg);
        if (vertexmark(markorg) == 0) {
          setvertexmark(markorg, 1);
        }
      }
    }
       dissolve(dissolveedge);
    lnext(nextedge, deadtriangle);
    sym(deadtriangle, nextedge);
       triangledealloc(m, deadtriangle.tri);
       if (nextedge.tri == m->dummytri) {
           otricopy(dissolveedge, nextedge);
    }
  }
  triangledealloc(m, finaledge.tri);

  trifree((VOID *) m->infvertex1);
  trifree((VOID *) m->infvertex2);
  trifree((VOID *) m->infvertex3);

  return hullsize;
}

#endif


#ifndef REDUCED

#ifdef ANSI_DECLARATORS
long incrementaldelaunay(struct mesh *m, struct behavior *b)
#else
long incrementaldelaunay(m, b)
struct mesh *m;
struct behavior *b;
#endif

{
  struct otri starttri;
  vertex vertexloop;

   boundingbox(m, b);
  if (b->verbose) {
    printf("  Incrementally inserting vertices.\n");
  }
  traversalinit(&m->vertices);
  vertexloop = vertextraverse(m);
  while (vertexloop != (vertex) NULL) {
    starttri.tri = m->dummytri;
    if (insertvertex(m, b, vertexloop, &starttri, (struct osub *) NULL, 0, 0)
        == DUPLICATEVERTEX) {
      if (!b->quiet) {
        printf(
"Warning:  A duplicate vertex at (%.12g, %.12g) appeared and was ignored.\n",
               vertexloop[0], vertexloop[1]);
      }
      setvertextype(vertexloop, UNDEADVERTEX);
      m->undeads++;
    }
    vertexloop = vertextraverse(m);
  }
   return removebox(m, b);
}

#endif



#ifndef REDUCED

#ifdef ANSI_DECLARATORS
void eventheapinsert(struct event **heap, int heapsize, struct event *newevent)
#else
void eventheapinsert(heap, heapsize, newevent)
struct event **heap;
int heapsize;
struct event *newevent;
#endif

{
  REAL eventx, eventy;
  int eventnum;
  int parent;
  int notdone;

  eventx = newevent->xkey;
  eventy = newevent->ykey;
  eventnum = heapsize;
  notdone = eventnum > 0;
  while (notdone) {
    parent = (eventnum - 1) >> 1;
    if ((heap[parent]->ykey < eventy) ||
        ((heap[parent]->ykey == eventy)
         && (heap[parent]->xkey <= eventx))) {
      notdone = 0;
    } else {
      heap[eventnum] = heap[parent];
      heap[eventnum]->heapposition = eventnum;

      eventnum = parent;
      notdone = eventnum > 0;
    }
  }
  heap[eventnum] = newevent;
  newevent->heapposition = eventnum;
}

#endif

#ifndef REDUCED

#ifdef ANSI_DECLARATORS
void eventheapify(struct event **heap, int heapsize, int eventnum)
#else
void eventheapify(heap, heapsize, eventnum)
struct event **heap;
int heapsize;
int eventnum;
#endif

{
  struct event *thisevent;
  REAL eventx, eventy;
  int leftchild, rightchild;
  int smallest;
  int notdone;

  thisevent = heap[eventnum];
  eventx = thisevent->xkey;
  eventy = thisevent->ykey;
  leftchild = 2 * eventnum + 1;
  notdone = leftchild < heapsize;
  while (notdone) {
    if ((heap[leftchild]->ykey < eventy) ||
        ((heap[leftchild]->ykey == eventy)
         && (heap[leftchild]->xkey < eventx))) {
      smallest = leftchild;
    } else {
      smallest = eventnum;
    }
    rightchild = leftchild + 1;
    if (rightchild < heapsize) {
      if ((heap[rightchild]->ykey < heap[smallest]->ykey) ||
          ((heap[rightchild]->ykey == heap[smallest]->ykey)
           && (heap[rightchild]->xkey < heap[smallest]->xkey))) {
        smallest = rightchild;
      }
    }
    if (smallest == eventnum) {
      notdone = 0;
    } else {
      heap[eventnum] = heap[smallest];
      heap[eventnum]->heapposition = eventnum;
      heap[smallest] = thisevent;
      thisevent->heapposition = smallest;

      eventnum = smallest;
      leftchild = 2 * eventnum + 1;
      notdone = leftchild < heapsize;
    }
  }
}

#endif

#ifndef REDUCED

#ifdef ANSI_DECLARATORS
void eventheapdelete(struct event **heap, int heapsize, int eventnum)
#else
void eventheapdelete(heap, heapsize, eventnum)
struct event **heap;
int heapsize;
int eventnum;
#endif

{
  struct event *moveevent;
  REAL eventx, eventy;
  int parent;
  int notdone;

  moveevent = heap[heapsize - 1];
  if (eventnum > 0) {
    eventx = moveevent->xkey;
    eventy = moveevent->ykey;
    do {
      parent = (eventnum - 1) >> 1;
      if ((heap[parent]->ykey < eventy) ||
          ((heap[parent]->ykey == eventy)
           && (heap[parent]->xkey <= eventx))) {
        notdone = 0;
      } else {
        heap[eventnum] = heap[parent];
        heap[eventnum]->heapposition = eventnum;

        eventnum = parent;
        notdone = eventnum > 0;
      }
    } while (notdone);
  }
  heap[eventnum] = moveevent;
  moveevent->heapposition = eventnum;
  eventheapify(heap, heapsize - 1, eventnum);
}

#endif

#ifndef REDUCED

#ifdef ANSI_DECLARATORS
void createeventheap(struct mesh *m, struct event ***eventheap,
                     struct event **events, struct event **freeevents)
#else
void createeventheap(m, eventheap, events, freeevents)
struct mesh *m;
struct event ***eventheap;
struct event **events;
struct event **freeevents;
#endif

{
  vertex thisvertex;
  int maxevents;
  int i;

  maxevents = (3 * m->invertices) / 2;
  *eventheap = (struct event **) trimalloc(maxevents *
                                           (int) sizeof(struct event *));
  *events = (struct event *) trimalloc(maxevents * (int) sizeof(struct event));
  traversalinit(&m->vertices);
  for (i = 0; i < m->invertices; i++) {
    thisvertex = vertextraverse(m);
    (*events)[i].eventptr = (VOID *) thisvertex;
    (*events)[i].xkey = thisvertex[0];
    (*events)[i].ykey = thisvertex[1];
    eventheapinsert(*eventheap, i, *events + i);
  }
  *freeevents = (struct event *) NULL;
  for (i = maxevents - 1; i >= m->invertices; i--) {
    (*events)[i].eventptr = (VOID *) *freeevents;
    *freeevents = *events + i;
  }
}

#endif

#ifndef REDUCED

#ifdef ANSI_DECLARATORS
int rightofhyperbola(struct mesh *m, struct otri *fronttri, vertex newsite)
#else
int rightofhyperbola(m, fronttri, newsite)
struct mesh *m;
struct otri *fronttri;
vertex newsite;
#endif

{
  vertex leftvertex, rightvertex;
  REAL dxa, dya, dxb, dyb;

  m->hyperbolacount++;

  dest(*fronttri, leftvertex);
  apex(*fronttri, rightvertex);
  if ((leftvertex[1] < rightvertex[1]) ||
      ((leftvertex[1] == rightvertex[1]) &&
       (leftvertex[0] < rightvertex[0]))) {
    if (newsite[0] >= rightvertex[0]) {
      return 1;
    }
  } else {
    if (newsite[0] <= leftvertex[0]) {
      return 0;
    }
  }
  dxa = leftvertex[0] - newsite[0];
  dya = leftvertex[1] - newsite[1];
  dxb = rightvertex[0] - newsite[0];
  dyb = rightvertex[1] - newsite[1];
  return dya * (dxb * dxb + dyb * dyb) > dyb * (dxa * dxa + dya * dya);
}

#endif

#ifndef REDUCED

#ifdef ANSI_DECLARATORS
REAL circletop(struct mesh *m, vertex pa, vertex pb, vertex pc, REAL ccwabc)
#else
REAL circletop(m, pa, pb, pc, ccwabc)
struct mesh *m;
vertex pa;
vertex pb;
vertex pc;
REAL ccwabc;
#endif

{
  REAL xac, yac, xbc, ybc, xab, yab;
  REAL aclen2, bclen2, ablen2;

  m->circletopcount++;

  xac = pa[0] - pc[0];
  yac = pa[1] - pc[1];
  xbc = pb[0] - pc[0];
  ybc = pb[1] - pc[1];
  xab = pa[0] - pb[0];
  yab = pa[1] - pb[1];
  aclen2 = xac * xac + yac * yac;
  bclen2 = xbc * xbc + ybc * ybc;
  ablen2 = xab * xab + yab * yab;
  return pc[1] + (xac * bclen2 - xbc * aclen2 + sqrt(aclen2 * bclen2 * ablen2))
               / (2.0 * ccwabc);
}

#endif

#ifndef REDUCED

#ifdef ANSI_DECLARATORS
void check4deadevent(struct otri *checktri, struct event **freeevents,
                     struct event **eventheap, int *heapsize)
#else
void check4deadevent(checktri, freeevents, eventheap, heapsize)
struct otri *checktri;
struct event **freeevents;
struct event **eventheap;
int *heapsize;
#endif

{
  struct event *deadevent;
  vertex eventvertex;
  int eventnum;

  org(*checktri, eventvertex);
  if (eventvertex != (vertex) NULL) {
    deadevent = (struct event *) eventvertex;
    eventnum = deadevent->heapposition;
    deadevent->eventptr = (VOID *) *freeevents;
    *freeevents = deadevent;
    eventheapdelete(eventheap, *heapsize, eventnum);
    (*heapsize)--;
    setorg(*checktri, NULL);
  }
}

#endif

#ifndef REDUCED

#ifdef ANSI_DECLARATORS
struct splaynode *splay(struct mesh *m, struct splaynode *splaytree,
                        vertex searchpoint, struct otri *searchtri)
#else
struct splaynode *splay(m, splaytree, searchpoint, searchtri)
struct mesh *m;
struct splaynode *splaytree;
vertex searchpoint;
struct otri *searchtri;
#endif

{
  struct splaynode *child, *grandchild;
  struct splaynode *lefttree, *righttree;
  struct splaynode *leftright;
  vertex checkvertex;
  int rightofroot, rightofchild;

  if (splaytree == (struct splaynode *) NULL) {
    return (struct splaynode *) NULL;
  }
  dest(splaytree->keyedge, checkvertex);
  if (checkvertex == splaytree->keydest) {
    rightofroot = rightofhyperbola(m, &splaytree->keyedge, searchpoint);
    if (rightofroot) {
      otricopy(splaytree->keyedge, *searchtri);
      child = splaytree->rchild;
    } else {
      child = splaytree->lchild;
    }
    if (child == (struct splaynode *) NULL) {
      return splaytree;
    }
    dest(child->keyedge, checkvertex);
    if (checkvertex != child->keydest) {
      child = splay(m, child, searchpoint, searchtri);
      if (child == (struct splaynode *) NULL) {
        if (rightofroot) {
          splaytree->rchild = (struct splaynode *) NULL;
        } else {
          splaytree->lchild = (struct splaynode *) NULL;
        }
        return splaytree;
      }
    }
    rightofchild = rightofhyperbola(m, &child->keyedge, searchpoint);
    if (rightofchild) {
      otricopy(child->keyedge, *searchtri);
      grandchild = splay(m, child->rchild, searchpoint, searchtri);
      child->rchild = grandchild;
    } else {
      grandchild = splay(m, child->lchild, searchpoint, searchtri);
      child->lchild = grandchild;
    }
    if (grandchild == (struct splaynode *) NULL) {
      if (rightofroot) {
        splaytree->rchild = child->lchild;
        child->lchild = splaytree;
      } else {
        splaytree->lchild = child->rchild;
        child->rchild = splaytree;
      }
      return child;
    }
    if (rightofchild) {
      if (rightofroot) {
        splaytree->rchild = child->lchild;
        child->lchild = splaytree;
      } else {
        splaytree->lchild = grandchild->rchild;
        grandchild->rchild = splaytree;
      }
      child->rchild = grandchild->lchild;
      grandchild->lchild = child;
    } else {
      if (rightofroot) {
        splaytree->rchild = grandchild->lchild;
        grandchild->lchild = splaytree;
      } else {
        splaytree->lchild = child->rchild;
        child->rchild = splaytree;
      }
      child->lchild = grandchild->rchild;
      grandchild->rchild = child;
    }
    return grandchild;
  } else {
    lefttree = splay(m, splaytree->lchild, searchpoint, searchtri);
    righttree = splay(m, splaytree->rchild, searchpoint, searchtri);

    pooldealloc(&m->splaynodes, (VOID *) splaytree);
    if (lefttree == (struct splaynode *) NULL) {
      return righttree;
    } else if (righttree == (struct splaynode *) NULL) {
      return lefttree;
    } else if (lefttree->rchild == (struct splaynode *) NULL) {
      lefttree->rchild = righttree->lchild;
      righttree->lchild = lefttree;
      return righttree;
    } else if (righttree->lchild == (struct splaynode *) NULL) {
      righttree->lchild = lefttree->rchild;
      lefttree->rchild = righttree;
      return lefttree;
    } else {
      leftright = lefttree->rchild;
      while (leftright->rchild != (struct splaynode *) NULL) {
        leftright = leftright->rchild;
      }
      leftright->rchild = righttree;
      return lefttree;
    }
  }
}

#endif

#ifndef REDUCED

#ifdef ANSI_DECLARATORS
struct splaynode *splayinsert(struct mesh *m, struct splaynode *splayroot,
                              struct otri *newkey, vertex searchpoint)
#else
struct splaynode *splayinsert(m, splayroot, newkey, searchpoint)
struct mesh *m;
struct splaynode *splayroot;
struct otri *newkey;
vertex searchpoint;
#endif

{
  struct splaynode *newsplaynode;

  newsplaynode = (struct splaynode *) poolalloc(&m->splaynodes);
  otricopy(*newkey, newsplaynode->keyedge);
  dest(*newkey, newsplaynode->keydest);
  if (splayroot == (struct splaynode *) NULL) {
    newsplaynode->lchild = (struct splaynode *) NULL;
    newsplaynode->rchild = (struct splaynode *) NULL;
  } else if (rightofhyperbola(m, &splayroot->keyedge, searchpoint)) {
    newsplaynode->lchild = splayroot;
    newsplaynode->rchild = splayroot->rchild;
    splayroot->rchild = (struct splaynode *) NULL;
  } else {
    newsplaynode->lchild = splayroot->lchild;
    newsplaynode->rchild = splayroot;
    splayroot->lchild = (struct splaynode *) NULL;
  }
  return newsplaynode;
}

#endif

#ifndef REDUCED

#ifdef ANSI_DECLARATORS
struct splaynode *circletopinsert(struct mesh *m, struct behavior *b,
                                  struct splaynode *splayroot,
                                  struct otri *newkey,
                                  vertex pa, vertex pb, vertex pc, REAL topy)
#else
struct splaynode *circletopinsert(m, b, splayroot, newkey, pa, pb, pc, topy)
struct mesh *m;
struct behavior *b;
struct splaynode *splayroot;
struct otri *newkey;
vertex pa;
vertex pb;
vertex pc;
REAL topy;
#endif

{
  REAL ccwabc;
  REAL xac, yac, xbc, ybc;
  REAL aclen2, bclen2;
  REAL searchpoint[2];
  struct otri dummytri;

  ccwabc = counterclockwise(m, b, pa, pb, pc);
  xac = pa[0] - pc[0];
  yac = pa[1] - pc[1];
  xbc = pb[0] - pc[0];
  ybc = pb[1] - pc[1];
  aclen2 = xac * xac + yac * yac;
  bclen2 = xbc * xbc + ybc * ybc;
  searchpoint[0] = pc[0] - (yac * bclen2 - ybc * aclen2) / (2.0 * ccwabc);
  searchpoint[1] = topy;
  return splayinsert(m, splay(m, splayroot, (vertex) searchpoint, &dummytri),
                     newkey, (vertex) searchpoint);
}

#endif

#ifndef REDUCED

#ifdef ANSI_DECLARATORS
struct splaynode *frontlocate(struct mesh *m, struct splaynode *splayroot,
                              struct otri *bottommost, vertex searchvertex,
                              struct otri *searchtri, int *farright)
#else
struct splaynode *frontlocate(m, splayroot, bottommost, searchvertex,
                              searchtri, farright)
struct mesh *m;
struct splaynode *splayroot;
struct otri *bottommost;
vertex searchvertex;
struct otri *searchtri;
int *farright;
#endif

{
  int farrightflag;
  triangle ptr;

  otricopy(*bottommost, *searchtri);
  splayroot = splay(m, splayroot, searchvertex, searchtri);

  farrightflag = 0;
  while (!farrightflag && rightofhyperbola(m, searchtri, searchvertex)) {
    onextself(*searchtri);
    farrightflag = otriequal(*searchtri, *bottommost);
  }
  *farright = farrightflag;
  return splayroot;
}

#endif

#ifndef REDUCED

#ifdef ANSI_DECLARATORS
long sweeplinedelaunay(struct mesh *m, struct behavior *b)
#else
long sweeplinedelaunay(m, b)
struct mesh *m;
struct behavior *b;
#endif

{
  struct event **eventheap;
  struct event *events;
  struct event *freeevents;
  struct event *nextevent;
  struct event *newevent;
  struct splaynode *splayroot;
  struct otri bottommost;
  struct otri searchtri;
  struct otri fliptri;
  struct otri lefttri, righttri, farlefttri, farrighttri;
  struct otri inserttri;
  vertex firstvertex, secondvertex;
  vertex nextvertex, lastvertex;
  vertex connectvertex;
  vertex leftvertex, midvertex, rightvertex;
  REAL lefttest, righttest;
  int heapsize;
  int check4events, farrightflag;
  triangle ptr;

  poolinit(&m->splaynodes, sizeof(struct splaynode), SPLAYNODEPERBLOCK,
           SPLAYNODEPERBLOCK, 0);
  splayroot = (struct splaynode *) NULL;

  if (b->verbose) {
    printf("  Placing vertices in event heap.\n");
  }
  createeventheap(m, &eventheap, &events, &freeevents);
  heapsize = m->invertices;

  if (b->verbose) {
    printf("  Forming triangulation.\n");
  }
  maketriangle(m, b, &lefttri);
  maketriangle(m, b, &righttri);
  bond(lefttri, righttri);
  lnextself(lefttri);
  lprevself(righttri);
  bond(lefttri, righttri);
  lnextself(lefttri);
  lprevself(righttri);
  bond(lefttri, righttri);
  firstvertex = (vertex) eventheap[0]->eventptr;
  eventheap[0]->eventptr = (VOID *) freeevents;
  freeevents = eventheap[0];
  eventheapdelete(eventheap, heapsize, 0);
  heapsize--;
  do {
    if (heapsize == 0) {
      printf("Error:  Input vertices are all identical.\n");
      triexit(1);
    }
    secondvertex = (vertex) eventheap[0]->eventptr;
    eventheap[0]->eventptr = (VOID *) freeevents;
    freeevents = eventheap[0];
    eventheapdelete(eventheap, heapsize, 0);
    heapsize--;
    if ((firstvertex[0] == secondvertex[0]) &&
        (firstvertex[1] == secondvertex[1])) {
      if (!b->quiet) {
        printf(
"Warning:  A duplicate vertex at (%.12g, %.12g) appeared and was ignored.\n",
               secondvertex[0], secondvertex[1]);
      }
      setvertextype(secondvertex, UNDEADVERTEX);
      m->undeads++;
    }
  } while ((firstvertex[0] == secondvertex[0]) &&
           (firstvertex[1] == secondvertex[1]));
  setorg(lefttri, firstvertex);
  setdest(lefttri, secondvertex);
  setorg(righttri, secondvertex);
  setdest(righttri, firstvertex);
  lprev(lefttri, bottommost);
  lastvertex = secondvertex;
  while (heapsize > 0) {
    nextevent = eventheap[0];
    eventheapdelete(eventheap, heapsize, 0);
    heapsize--;
    check4events = 1;
    if (nextevent->xkey < m->xmin) {
      decode(nextevent->eventptr, fliptri);
      oprev(fliptri, farlefttri);
      check4deadevent(&farlefttri, &freeevents, eventheap, &heapsize);
      onext(fliptri, farrighttri);
      check4deadevent(&farrighttri, &freeevents, eventheap, &heapsize);

      if (otriequal(farlefttri, bottommost)) {
        lprev(fliptri, bottommost);
      }
      flip(m, b, &fliptri);
      setapex(fliptri, NULL);
      lprev(fliptri, lefttri);
      lnext(fliptri, righttri);
      sym(lefttri, farlefttri);

      if (randomnation(SAMPLERATE) == 0) {
        symself(fliptri);
        dest(fliptri, leftvertex);
        apex(fliptri, midvertex);
        org(fliptri, rightvertex);
        splayroot = circletopinsert(m, b, splayroot, &lefttri, leftvertex,
                                    midvertex, rightvertex, nextevent->ykey);
      }
    } else {
      nextvertex = (vertex) nextevent->eventptr;
      if ((nextvertex[0] == lastvertex[0]) &&
          (nextvertex[1] == lastvertex[1])) {
        if (!b->quiet) {
          printf(
"Warning:  A duplicate vertex at (%.12g, %.12g) appeared and was ignored.\n",
                 nextvertex[0], nextvertex[1]);
        }
        setvertextype(nextvertex, UNDEADVERTEX);
        m->undeads++;
        check4events = 0;
      } else {
        lastvertex = nextvertex;

        splayroot = frontlocate(m, splayroot, &bottommost, nextvertex,
                                &searchtri, &farrightflag);

        check4deadevent(&searchtri, &freeevents, eventheap, &heapsize);

        otricopy(searchtri, farrighttri);
        sym(searchtri, farlefttri);
        maketriangle(m, b, &lefttri);
        maketriangle(m, b, &righttri);
        dest(farrighttri, connectvertex);
        setorg(lefttri, connectvertex);
        setdest(lefttri, nextvertex);
        setorg(righttri, nextvertex);
        setdest(righttri, connectvertex);
        bond(lefttri, righttri);
        lnextself(lefttri);
        lprevself(righttri);
        bond(lefttri, righttri);
        lnextself(lefttri);
        lprevself(righttri);
        bond(lefttri, farlefttri);
        bond(righttri, farrighttri);
        if (!farrightflag && otriequal(farrighttri, bottommost)) {
          otricopy(lefttri, bottommost);
        }

        if (randomnation(SAMPLERATE) == 0) {
          splayroot = splayinsert(m, splayroot, &lefttri, nextvertex);
        } else if (randomnation(SAMPLERATE) == 0) {
          lnext(righttri, inserttri);
          splayroot = splayinsert(m, splayroot, &inserttri, nextvertex);
        }
      }
    }
    nextevent->eventptr = (VOID *) freeevents;
    freeevents = nextevent;

    if (check4events) {
      apex(farlefttri, leftvertex);
      dest(lefttri, midvertex);
      apex(lefttri, rightvertex);
      lefttest = counterclockwise(m, b, leftvertex, midvertex, rightvertex);
      if (lefttest > 0.0) {
        newevent = freeevents;
        freeevents = (struct event *) freeevents->eventptr;
        newevent->xkey = m->xminextreme;
        newevent->ykey = circletop(m, leftvertex, midvertex, rightvertex,
                                   lefttest);
        newevent->eventptr = (VOID *) encode(lefttri);
        eventheapinsert(eventheap, heapsize, newevent);
        heapsize++;
        setorg(lefttri, newevent);
      }
      apex(righttri, leftvertex);
      org(righttri, midvertex);
      apex(farrighttri, rightvertex);
      righttest = counterclockwise(m, b, leftvertex, midvertex, rightvertex);
      if (righttest > 0.0) {
        newevent = freeevents;
        freeevents = (struct event *) freeevents->eventptr;
        newevent->xkey = m->xminextreme;
        newevent->ykey = circletop(m, leftvertex, midvertex, rightvertex,
                                   righttest);
        newevent->eventptr = (VOID *) encode(farrighttri);
        eventheapinsert(eventheap, heapsize, newevent);
        heapsize++;
        setorg(farrighttri, newevent);
      }
    }
  }

  pooldeinit(&m->splaynodes);
  lprevself(bottommost);
  return removeghosts(m, b, &bottommost);
}

#endif




#ifdef ANSI_DECLARATORS
long delaunay(struct mesh *m, struct behavior *b)
#else
long delaunay(m, b)
struct mesh *m;
struct behavior *b;
#endif

{
  long hulledges;

  m->eextras = 0;
  initializetrisubpools(m, b);

#ifdef REDUCED
  if (!b->quiet) {
    printf(
      "Constructing Delaunay triangulation by divide-and-conquer method.\n");
  }
  hulledges = divconqdelaunay(m, b);
#else
  if (!b->quiet) {
    printf("Constructing Delaunay triangulation ");
    if (b->incremental) {
      printf("by incremental method.\n");
    } else if (b->sweepline) {
      printf("by sweepline method.\n");
    } else {
      printf("by divide-and-conquer method.\n");
    }
  }
  if (b->incremental) {
    hulledges = incrementaldelaunay(m, b);
  } else if (b->sweepline) {
    hulledges = sweeplinedelaunay(m, b);
  } else {
    hulledges = divconqdelaunay(m, b);
  }
#endif

  if (m->triangles.items == 0) {
       return 0l;
  } else {
    return hulledges;
  }
}


#ifndef CDT_ONLY

#ifdef TRILIBRARY

#ifdef ANSI_DECLARATORS
int reconstruct(struct mesh *m, struct behavior *b, int *trianglelist,
                REAL *triangleattriblist, REAL *trianglearealist,
                int elements, int corners, int attribs,
                int *segmentlist,int *segmentmarkerlist, int numberofsegments)
#else
int reconstruct(m, b, trianglelist, triangleattriblist, trianglearealist,
                elements, corners, attribs, segmentlist, segmentmarkerlist,
                numberofsegments)
struct mesh *m;
struct behavior *b;
int *trianglelist;
REAL *triangleattriblist;
REAL *trianglearealist;
int elements;
int corners;
int attribs;
int *segmentlist;
int *segmentmarkerlist;
int numberofsegments;
#endif

#else

#ifdef ANSI_DECLARATORS
long reconstruct(struct mesh *m, struct behavior *b, char *elefilename,
                 char *areafilename, char *polyfilename, FILE *polyfile)
#else
long reconstruct(m, b, elefilename, areafilename, polyfilename, polyfile)
struct mesh *m;
struct behavior *b;
char *elefilename;
char *areafilename;
char *polyfilename;
FILE *polyfile;
#endif

#endif

{
#ifdef TRILIBRARY
  int vertexindex;
  int attribindex;
#else
  FILE *elefile;
  FILE *areafile;
  char inputline[INPUTLINESIZE];
  char *stringptr;
  int areaelements;
#endif
  struct otri triangleloop;
  struct otri triangleleft;
  struct otri checktri;
  struct otri checkleft;
  struct otri checkneighbor;
  struct osub subsegloop;
  triangle *vertexarray;
  triangle *prevlink;
  triangle nexttri;
  vertex tdest, tapex;
  vertex checkdest, checkapex;
  vertex shorg;
  vertex killvertex;
  vertex segmentorg, segmentdest;
  REAL area;
  int corner[3];
  int end[2];
  int killvertexindex;
  int incorners;
  int segmentmarkers;
  int boundmarker;
  int aroundvertex;
  long hullsize;
  int notfound;
  long elementnumber, segmentnumber;
  int i, j;
  triangle ptr;

#ifdef TRILIBRARY
  m->inelements = elements;
  incorners = corners;
  if (incorners < 3) {
    printf("Error:  Triangles must have at least 3 vertices.\n");
    triexit(1);
  }
  m->eextras = attribs;
#else
   if (!b->quiet) {
    printf("Opening %s.\n", elefilename);
  }
  elefile = fopen(elefilename, "r");
  if (elefile == (FILE *) NULL) {
    printf("  Error:  Cannot access file %s.\n", elefilename);
    triexit(1);
  }
    stringptr = readline(inputline, elefile, elefilename);
  m->inelements = (int) strtol(stringptr, &stringptr, 0);
  stringptr = findfield(stringptr);
  if (*stringptr == '\0') {
    incorners = 3;
  } else {
    incorners = (int) strtol(stringptr, &stringptr, 0);
    if (incorners < 3) {
      printf("Error:  Triangles in %s must have at least 3 vertices.\n",
             elefilename);
      triexit(1);
    }
  }
  stringptr = findfield(stringptr);
  if (*stringptr == '\0') {
    m->eextras = 0;
  } else {
    m->eextras = (int) strtol(stringptr, &stringptr, 0);
  }
#endif

  initializetrisubpools(m, b);

   for (elementnumber = 1; elementnumber <= m->inelements; elementnumber++) {
    maketriangle(m, b, &triangleloop);
       triangleloop.tri[3] = (triangle) triangleloop.tri;
  }

  segmentmarkers = 0;
  if (b->poly) {
#ifdef TRILIBRARY
    m->insegments = numberofsegments;
    segmentmarkers = segmentmarkerlist != (int *) NULL;
#else
          stringptr = readline(inputline, polyfile, b->inpolyfilename);
    m->insegments = (int) strtol(stringptr, &stringptr, 0);
    stringptr = findfield(stringptr);
    if (*stringptr != '\0') {
      segmentmarkers = (int) strtol(stringptr, &stringptr, 0);
    }
#endif

       for (segmentnumber = 1; segmentnumber <= m->insegments; segmentnumber++) {
      makesubseg(m, &subsegloop);
           subsegloop.ss[2] = (subseg) subsegloop.ss;
    }
  }

#ifdef TRILIBRARY
  vertexindex = 0;
  attribindex = 0;
#else
  if (b->vararea) {
       if (!b->quiet) {
      printf("Opening %s.\n", areafilename);
    }
    areafile = fopen(areafilename, "r");
    if (areafile == (FILE *) NULL) {
      printf("  Error:  Cannot access file %s.\n", areafilename);
      triexit(1);
    }
    stringptr = readline(inputline, areafile, areafilename);
    areaelements = (int) strtol(stringptr, &stringptr, 0);
    if (areaelements != m->inelements) {
      printf("Error:  %s and %s disagree on number of triangles.\n",
             elefilename, areafilename);
      triexit(1);
    }
  }
#endif

  if (!b->quiet) {
    printf("Reconstructing mesh.\n");
  }
     vertexarray = (triangle *) trimalloc(m->vertices.items *
                                       (int) sizeof(triangle));
   for (i = 0; i < m->vertices.items; i++) {
    vertexarray[i] = (triangle) m->dummytri;
  }

  if (b->verbose) {
    printf("  Assembling triangles.\n");
  }
    traversalinit(&m->triangles);
  triangleloop.tri = triangletraverse(m);
  elementnumber = b->firstnumber;
  while (triangleloop.tri != (triangle *) NULL) {
#ifdef TRILIBRARY
       for (j = 0; j < 3; j++) {
      corner[j] = trianglelist[vertexindex++];
      if ((corner[j] < b->firstnumber) ||
          (corner[j] >= b->firstnumber + m->invertices)) {
        printf("Error:  Triangle %ld has an invalid vertex index.\n",
               elementnumber);
        triexit(1);
      }
    }
#else
       stringptr = readline(inputline, elefile, elefilename);
    for (j = 0; j < 3; j++) {
      stringptr = findfield(stringptr);
      if (*stringptr == '\0') {
        printf("Error:  Triangle %ld is missing vertex %d in %s.\n",
               elementnumber, j + 1, elefilename);
        triexit(1);
      } else {
        corner[j] = (int) strtol(stringptr, &stringptr, 0);
        if ((corner[j] < b->firstnumber) ||
            (corner[j] >= b->firstnumber + m->invertices)) {
          printf("Error:  Triangle %ld has an invalid vertex index.\n",
                 elementnumber);
          triexit(1);
        }
      }
    }
#endif

       for (j = 3; j < incorners; j++) {
#ifdef TRILIBRARY
      killvertexindex = trianglelist[vertexindex++];
#else
      stringptr = findfield(stringptr);
      if (*stringptr != '\0') {
        killvertexindex = (int) strtol(stringptr, &stringptr, 0);
#endif
        if ((killvertexindex >= b->firstnumber) &&
            (killvertexindex < b->firstnumber + m->invertices)) {
                   killvertex = getvertex(m, b, killvertexindex);
          if (vertextype(killvertex) != DEADVERTEX) {
            vertexdealloc(m, killvertex);
          }
        }
#ifndef TRILIBRARY
      }
#endif
    }

       for (j = 0; j < m->eextras; j++) {
#ifdef TRILIBRARY
      setelemattribute(triangleloop, j, triangleattriblist[attribindex++]);
#else
      stringptr = findfield(stringptr);
      if (*stringptr == '\0') {
        setelemattribute(triangleloop, j, 0);
      } else {
        setelemattribute(triangleloop, j,
                         (REAL) strtod(stringptr, &stringptr));
      }
#endif
    }

    if (b->vararea) {
#ifdef TRILIBRARY
      area = trianglearealist[elementnumber - b->firstnumber];
#else
           stringptr = readline(inputline, areafile, areafilename);
      stringptr = findfield(stringptr);
      if (*stringptr == '\0') {
        area = -1.0;
      } else {
        area = (REAL) strtod(stringptr, &stringptr);
      }
#endif
      setareabound(triangleloop, area);
    }

       triangleloop.orient = 0;
    setorg(triangleloop, getvertex(m, b, corner[0]));
    setdest(triangleloop, getvertex(m, b, corner[1]));
    setapex(triangleloop, getvertex(m, b, corner[2]));
       for (triangleloop.orient = 0; triangleloop.orient < 3;
         triangleloop.orient++) {
           aroundvertex = corner[triangleloop.orient];
           nexttri = vertexarray[aroundvertex - b->firstnumber];
           triangleloop.tri[6 + triangleloop.orient] = nexttri;
           vertexarray[aroundvertex - b->firstnumber] = encode(triangleloop);
      decode(nexttri, checktri);
      if (checktri.tri != m->dummytri) {
        dest(triangleloop, tdest);
        apex(triangleloop, tapex);
               do {
          dest(checktri, checkdest);
          apex(checktri, checkapex);
          if (tapex == checkdest) {
                       lprev(triangleloop, triangleleft);
            bond(triangleleft, checktri);
          }
          if (tdest == checkapex) {
                       lprev(checktri, checkleft);
            bond(triangleloop, checkleft);
          }
                   nexttri = checktri.tri[6 + checktri.orient];
          decode(nexttri, checktri);
        } while (checktri.tri != m->dummytri);
      }
    }
    triangleloop.tri = triangletraverse(m);
    elementnumber++;
  }

#ifdef TRILIBRARY
  vertexindex = 0;
#else
  fclose(elefile);
  if (b->vararea) {
    fclose(areafile);
  }
#endif

  hullsize = 0;
  if (b->poly) {
    if (b->verbose) {
      printf("  Marking segments in triangulation.\n");
    }
          boundmarker = 0;
    traversalinit(&m->subsegs);
    subsegloop.ss = subsegtraverse(m);
    segmentnumber = b->firstnumber;
    while (subsegloop.ss != (subseg *) NULL) {
#ifdef TRILIBRARY
      end[0] = segmentlist[vertexindex++];
      end[1] = segmentlist[vertexindex++];
      if (segmentmarkers) {
        boundmarker = segmentmarkerlist[segmentnumber - b->firstnumber];
      }
#else
           stringptr = readline(inputline, polyfile, b->inpolyfilename);
           stringptr = findfield(stringptr);
      if (*stringptr == '\0') {
        printf("Error:  Segment %ld has no endpoints in %s.\n", segmentnumber,
               polyfilename);
        triexit(1);
      } else {
        end[0] = (int) strtol(stringptr, &stringptr, 0);
      }
      stringptr = findfield(stringptr);
      if (*stringptr == '\0') {
        printf("Error:  Segment %ld is missing its second endpoint in %s.\n",
               segmentnumber, polyfilename);
        triexit(1);
      } else {
        end[1] = (int) strtol(stringptr, &stringptr, 0);
      }
      if (segmentmarkers) {
        stringptr = findfield(stringptr);
        if (*stringptr == '\0') {
          boundmarker = 0;
        } else {
          boundmarker = (int) strtol(stringptr, &stringptr, 0);
        }
      }
#endif
      for (j = 0; j < 2; j++) {
        if ((end[j] < b->firstnumber) ||
            (end[j] >= b->firstnumber + m->invertices)) {
          printf("Error:  Segment %ld has an invalid vertex index.\n",
                 segmentnumber);
          triexit(1);
        }
      }

           subsegloop.ssorient = 0;
      segmentorg = getvertex(m, b, end[0]);
      segmentdest = getvertex(m, b, end[1]);
      setsorg(subsegloop, segmentorg);
      setsdest(subsegloop, segmentdest);
      setsegorg(subsegloop, segmentorg);
      setsegdest(subsegloop, segmentdest);
      setmark(subsegloop, boundmarker);
           for (subsegloop.ssorient = 0; subsegloop.ssorient < 2;
           subsegloop.ssorient++) {
               aroundvertex = end[1 - subsegloop.ssorient];
               prevlink = &vertexarray[aroundvertex - b->firstnumber];
        nexttri = vertexarray[aroundvertex - b->firstnumber];
        decode(nexttri, checktri);
        sorg(subsegloop, shorg);
        notfound = 1;
                                                         while (notfound && (checktri.tri != m->dummytri)) {
          dest(checktri, checkdest);
          if (shorg == checkdest) {
                       *prevlink = checktri.tri[6 + checktri.orient];
                       tsbond(checktri, subsegloop);
                       sym(checktri, checkneighbor);
            if (checkneighbor.tri == m->dummytri) {
                                                     insertsubseg(m, b, &checktri, 1);
              hullsize++;
            }
            notfound = 0;
          }
                   prevlink = &checktri.tri[6 + checktri.orient];
          nexttri = checktri.tri[6 + checktri.orient];
          decode(nexttri, checktri);
        }
      }
      subsegloop.ss = subsegtraverse(m);
      segmentnumber++;
    }
  }

    for (i = 0; i < m->vertices.items; i++) {
       nexttri = vertexarray[i];
    decode(nexttri, checktri);
    while (checktri.tri != m->dummytri) {
                nexttri = checktri.tri[6 + checktri.orient];
           tsdissolve(checktri);
      sym(checktri, checkneighbor);
      if (checkneighbor.tri == m->dummytri) {
        insertsubseg(m, b, &checktri, 1);
        hullsize++;
      }
      decode(nexttri, checktri);
    }
  }

  trifree((VOID *) vertexarray);
  return hullsize;
}

#endif




#ifdef ANSI_DECLARATORS
enum finddirectionresult finddirection(struct mesh *m, struct behavior *b,
                                       struct otri *searchtri,
                                       vertex searchpoint)
#else
enum finddirectionresult finddirection(m, b, searchtri, searchpoint)
struct mesh *m;
struct behavior *b;
struct otri *searchtri;
vertex searchpoint;
#endif

{
  struct otri checktri;
  vertex startvertex;
  vertex leftvertex, rightvertex;
  REAL leftccw, rightccw;
  int leftflag, rightflag;
  triangle ptr;

  org(*searchtri, startvertex);
  dest(*searchtri, rightvertex);
  apex(*searchtri, leftvertex);
   leftccw = counterclockwise(m, b, searchpoint, startvertex, leftvertex);
  leftflag = leftccw > 0.0;
   rightccw = counterclockwise(m, b, startvertex, searchpoint, rightvertex);
  rightflag = rightccw > 0.0;
  if (leftflag && rightflag) {
          onext(*searchtri, checktri);
    if (checktri.tri == m->dummytri) {
      leftflag = 0;
    } else {
      rightflag = 0;
    }
  }
  while (leftflag) {
       onextself(*searchtri);
    if (searchtri->tri == m->dummytri) {
      printf("Internal error in finddirection():  Unable to find a\n");
      printf("  triangle leading from (%.12g, %.12g) to", startvertex[0],
             startvertex[1]);
      printf("  (%.12g, %.12g).\n", searchpoint[0], searchpoint[1]);
      internalerror();
    }
    apex(*searchtri, leftvertex);
    rightccw = leftccw;
    leftccw = counterclockwise(m, b, searchpoint, startvertex, leftvertex);
    leftflag = leftccw > 0.0;
  }
  while (rightflag) {
       oprevself(*searchtri);
    if (searchtri->tri == m->dummytri) {
      printf("Internal error in finddirection():  Unable to find a\n");
      printf("  triangle leading from (%.12g, %.12g) to", startvertex[0],
             startvertex[1]);
      printf("  (%.12g, %.12g).\n", searchpoint[0], searchpoint[1]);
      internalerror();
    }
    dest(*searchtri, rightvertex);
    leftccw = rightccw;
    rightccw = counterclockwise(m, b, startvertex, searchpoint, rightvertex);
    rightflag = rightccw > 0.0;
  }
  if (leftccw == 0.0) {
    return LEFTCOLLINEAR;
  } else if (rightccw == 0.0) {
    return RIGHTCOLLINEAR;
  } else {
    return WITHIN;
  }
}


#ifdef ANSI_DECLARATORS
void segmentintersection(struct mesh *m, struct behavior *b,
                         struct otri *splittri, struct osub *splitsubseg,
                         vertex endpoint2)
#else
void segmentintersection(m, b, splittri, splitsubseg, endpoint2)
struct mesh *m;
struct behavior *b;
struct otri *splittri;
struct osub *splitsubseg;
vertex endpoint2;
#endif

{
  struct osub opposubseg;
  vertex endpoint1;
  vertex torg, tdest;
  vertex leftvertex, rightvertex;
  vertex newvertex;
  enum insertvertexresult success;
  enum finddirectionresult collinear;
  REAL ex, ey;
  REAL tx, ty;
  REAL etx, ety;
  REAL split, denom;
  int i;
  triangle ptr;
  subseg sptr;

   apex(*splittri, endpoint1);
  org(*splittri, torg);
  dest(*splittri, tdest);
   tx = tdest[0] - torg[0];
  ty = tdest[1] - torg[1];
  ex = endpoint2[0] - endpoint1[0];
  ey = endpoint2[1] - endpoint1[1];
  etx = torg[0] - endpoint2[0];
  ety = torg[1] - endpoint2[1];
  denom = ty * ex - tx * ey;
  if (denom == 0.0) {
    printf("Internal error in segmentintersection():");
    printf("  Attempt to find intersection of parallel segments.\n");
    internalerror();
  }
  split = (ey * etx - ex * ety) / denom;
   newvertex = (vertex) poolalloc(&m->vertices);
   for (i = 0; i < 2 + m->nextras; i++) {
    newvertex[i] = torg[i] + split * (tdest[i] - torg[i]);
  }
  setvertexmark(newvertex, mark(*splitsubseg));
  setvertextype(newvertex, INPUTVERTEX);
  if (b->verbose > 1) {
    printf(
  "  Splitting subsegment (%.12g, %.12g) (%.12g, %.12g) at (%.12g, %.12g).\n",
           torg[0], torg[1], tdest[0], tdest[1], newvertex[0], newvertex[1]);
  }
   success = insertvertex(m, b, newvertex, splittri, splitsubseg, 0, 0);
  if (success != SUCCESSFULVERTEX) {
    printf("Internal error in segmentintersection():\n");
    printf("  Failure to split a segment.\n");
    internalerror();
  }
   setvertex2tri(newvertex, encode(*splittri));
  if (m->steinerleft > 0) {
    m->steinerleft--;
  }

   ssymself(*splitsubseg);
  spivot(*splitsubseg, opposubseg);
  sdissolve(*splitsubseg);
  sdissolve(opposubseg);
  do {
    setsegorg(*splitsubseg, newvertex);
    snextself(*splitsubseg);
  } while (splitsubseg->ss != m->dummysub);
  do {
    setsegorg(opposubseg, newvertex);
    snextself(opposubseg);
  } while (opposubseg.ss != m->dummysub);

    collinear = finddirection(m, b, splittri, endpoint1);
  dest(*splittri, rightvertex);
  apex(*splittri, leftvertex);
  if ((leftvertex[0] == endpoint1[0]) && (leftvertex[1] == endpoint1[1])) {
    onextself(*splittri);
  } else if ((rightvertex[0] != endpoint1[0]) ||
             (rightvertex[1] != endpoint1[1])) {
    printf("Internal error in segmentintersection():\n");
    printf("  Topological inconsistency after splitting a segment.\n");
    internalerror();
  }
 }


#ifdef ANSI_DECLARATORS
int scoutsegment(struct mesh *m, struct behavior *b, struct otri *searchtri,
                 vertex endpoint2, int newmark)
#else
int scoutsegment(m, b, searchtri, endpoint2, newmark)
struct mesh *m;
struct behavior *b;
struct otri *searchtri;
vertex endpoint2;
int newmark;
#endif

{
  struct otri crosstri;
  struct osub crosssubseg;
  vertex leftvertex, rightvertex;
  enum finddirectionresult collinear;
  subseg sptr;

  collinear = finddirection(m, b, searchtri, endpoint2);
  dest(*searchtri, rightvertex);
  apex(*searchtri, leftvertex);
  if (((leftvertex[0] == endpoint2[0]) && (leftvertex[1] == endpoint2[1])) ||
      ((rightvertex[0] == endpoint2[0]) && (rightvertex[1] == endpoint2[1]))) {
       if ((leftvertex[0] == endpoint2[0]) && (leftvertex[1] == endpoint2[1])) {
      lprevself(*searchtri);
    }
       insertsubseg(m, b, searchtri, newmark);
    return 1;
  } else if (collinear == LEFTCOLLINEAR) {
          lprevself(*searchtri);
    insertsubseg(m, b, searchtri, newmark);
       return scoutsegment(m, b, searchtri, endpoint2, newmark);
  } else if (collinear == RIGHTCOLLINEAR) {
       insertsubseg(m, b, searchtri, newmark);
       lnextself(*searchtri);
       return scoutsegment(m, b, searchtri, endpoint2, newmark);
  } else {
    lnext(*searchtri, crosstri);
    tspivot(crosstri, crosssubseg);
       if (crosssubseg.ss == m->dummysub) {
      return 0;
    } else {
           segmentintersection(m, b, &crosstri, &crosssubseg, endpoint2);
      otricopy(crosstri, *searchtri);
      insertsubseg(m, b, searchtri, newmark);
           return scoutsegment(m, b, searchtri, endpoint2, newmark);
    }
  }
}


#ifndef REDUCED
#ifndef CDT_ONLY

#ifdef ANSI_DECLARATORS
void conformingedge(struct mesh *m, struct behavior *b,
                    vertex endpoint1, vertex endpoint2, int newmark)
#else
void conformingedge(m, b, endpoint1, endpoint2, newmark)
struct mesh *m;
struct behavior *b;
vertex endpoint1;
vertex endpoint2;
int newmark;
#endif

{
  struct otri searchtri1, searchtri2;
  struct osub brokensubseg;
  vertex newvertex;
  vertex midvertex1, midvertex2;
  enum insertvertexresult success;
  int i;
  subseg sptr;

  if (b->verbose > 2) {
    printf("Forcing segment into triangulation by recursive splitting:\n");
    printf("  (%.12g, %.12g) (%.12g, %.12g)\n", endpoint1[0], endpoint1[1],
           endpoint2[0], endpoint2[1]);
  }
   newvertex = (vertex) poolalloc(&m->vertices);
   for (i = 0; i < 2 + m->nextras; i++) {
    newvertex[i] = 0.5 * (endpoint1[i] + endpoint2[i]);
  }
  setvertexmark(newvertex, newmark);
  setvertextype(newvertex, SEGMENTVERTEX);
   searchtri1.tri = m->dummytri;
   success = insertvertex(m, b, newvertex, &searchtri1, (struct osub *) NULL,
                         0, 0);
  if (success == DUPLICATEVERTEX) {
    if (b->verbose > 2) {
      printf("  Segment intersects existing vertex (%.12g, %.12g).\n",
             newvertex[0], newvertex[1]);
    }
       vertexdealloc(m, newvertex);
    org(searchtri1, newvertex);
  } else {
    if (success == VIOLATINGVERTEX) {
      if (b->verbose > 2) {
        printf("  Two segments intersect at (%.12g, %.12g).\n",
               newvertex[0], newvertex[1]);
      }
           tspivot(searchtri1, brokensubseg);
      success = insertvertex(m, b, newvertex, &searchtri1, &brokensubseg,
                             0, 0);
      if (success != SUCCESSFULVERTEX) {
        printf("Internal error in conformingedge():\n");
        printf("  Failure to split a segment.\n");
        internalerror();
      }
    }
       if (m->steinerleft > 0) {
      m->steinerleft--;
    }
  }
  otricopy(searchtri1, searchtri2);
       finddirection(m, b, &searchtri2, endpoint2);
  if (!scoutsegment(m, b, &searchtri1, endpoint1, newmark)) {
          org(searchtri1, midvertex1);
    conformingedge(m, b, midvertex1, endpoint1, newmark);
  }
  if (!scoutsegment(m, b, &searchtri2, endpoint2, newmark)) {
          org(searchtri2, midvertex2);
    conformingedge(m, b, midvertex2, endpoint2, newmark);
  }
}

#endif
#endif


#ifdef ANSI_DECLARATORS
void delaunayfixup(struct mesh *m, struct behavior *b,
                   struct otri *fixuptri, int leftside)
#else
void delaunayfixup(m, b, fixuptri, leftside)
struct mesh *m;
struct behavior *b;
struct otri *fixuptri;
int leftside;
#endif

{
  struct otri neartri;
  struct otri fartri;
  struct osub faredge;
  vertex nearvertex, leftvertex, rightvertex, farvertex;
  triangle ptr;
  subseg sptr;

  lnext(*fixuptri, neartri);
  sym(neartri, fartri);
   if (fartri.tri == m->dummytri) {
    return;
  }
  tspivot(neartri, faredge);
  if (faredge.ss != m->dummysub) {
    return;
  }
   apex(neartri, nearvertex);
  org(neartri, leftvertex);
  dest(neartri, rightvertex);
  apex(fartri, farvertex);
   if (leftside) {
    if (counterclockwise(m, b, nearvertex, leftvertex, farvertex) <= 0.0) {
                return;
    }
  } else {
    if (counterclockwise(m, b, farvertex, rightvertex, nearvertex) <= 0.0) {
                return;
    }
  }
  if (counterclockwise(m, b, rightvertex, leftvertex, farvertex) > 0.0) {
                if (incircle(m, b, leftvertex, farvertex, rightvertex, nearvertex) <=
        0.0) {
      return;
    }
     }
  flip(m, b, &neartri);
  lprevself(*fixuptri);
   delaunayfixup(m, b, fixuptri, leftside);
  delaunayfixup(m, b, &fartri, leftside);
}


#ifdef ANSI_DECLARATORS
void constrainededge(struct mesh *m, struct behavior *b,
                     struct otri *starttri, vertex endpoint2, int newmark)
#else
void constrainededge(m, b, starttri, endpoint2, newmark)
struct mesh *m;
struct behavior *b;
struct otri *starttri;
vertex endpoint2;
int newmark;
#endif

{
  struct otri fixuptri, fixuptri2;
  struct osub crosssubseg;
  vertex endpoint1;
  vertex farvertex;
  REAL area;
  int collision;
  int done;
  triangle ptr;
  subseg sptr;

  org(*starttri, endpoint1);
  lnext(*starttri, fixuptri);
  flip(m, b, &fixuptri);
    collision = 0;
  done = 0;
  do {
    org(fixuptri, farvertex);
          if ((farvertex[0] == endpoint2[0]) && (farvertex[1] == endpoint2[1])) {
      oprev(fixuptri, fixuptri2);
           delaunayfixup(m, b, &fixuptri, 0);
      delaunayfixup(m, b, &fixuptri2, 1);
      done = 1;
    } else {
                     area = counterclockwise(m, b, endpoint1, endpoint2, farvertex);
      if (area == 0.0) {
               collision = 1;
        oprev(fixuptri, fixuptri2);
               delaunayfixup(m, b, &fixuptri, 0);
        delaunayfixup(m, b, &fixuptri2, 1);
        done = 1;
      } else {
        if (area > 0.0) {
          oprev(fixuptri, fixuptri2);
                            delaunayfixup(m, b, &fixuptri2, 1);
                                     lprevself(fixuptri);
        } else {
          delaunayfixup(m, b, &fixuptri, 0);
                                     oprevself(fixuptri);
        }
               tspivot(fixuptri, crosssubseg);
        if (crosssubseg.ss == m->dummysub) {
          flip(m, b, &fixuptri);
        } else {
                   collision = 1;
                   segmentintersection(m, b, &fixuptri, &crosssubseg, endpoint2);
          done = 1;
        }
      }
    }
  } while (!done);
   insertsubseg(m, b, &fixuptri, newmark);
    if (collision) {
       if (!scoutsegment(m, b, &fixuptri, endpoint2, newmark)) {
      constrainededge(m, b, &fixuptri, endpoint2, newmark);
    }
  }
}


#ifdef ANSI_DECLARATORS
void insertsegment(struct mesh *m, struct behavior *b,
                   vertex endpoint1, vertex endpoint2, int newmark)
#else
void insertsegment(m, b, endpoint1, endpoint2, newmark)
struct mesh *m;
struct behavior *b;
vertex endpoint1;
vertex endpoint2;
int newmark;
#endif

{
  struct otri searchtri1, searchtri2;
  triangle encodedtri;
  vertex checkvertex;
  triangle ptr;

  if (b->verbose > 1) {
    printf("  Connecting (%.12g, %.12g) to (%.12g, %.12g).\n",
           endpoint1[0], endpoint1[1], endpoint2[0], endpoint2[1]);
  }

   checkvertex = (vertex) NULL;
  encodedtri = vertex2tri(endpoint1);
  if (encodedtri != (triangle) NULL) {
    decode(encodedtri, searchtri1);
    org(searchtri1, checkvertex);
  }
  if (checkvertex != endpoint1) {
       searchtri1.tri = m->dummytri;
    searchtri1.orient = 0;
    symself(searchtri1);
       if (locate(m, b, endpoint1, &searchtri1) != ONVERTEX) {
      printf(
        "Internal error in insertsegment():  Unable to locate PSLG vertex\n");
      printf("  (%.12g, %.12g) in triangulation.\n",
             endpoint1[0], endpoint1[1]);
      internalerror();
    }
  }
   otricopy(searchtri1, m->recenttri);
    if (scoutsegment(m, b, &searchtri1, endpoint2, newmark)) {
       return;
  }
    org(searchtri1, endpoint1);

   checkvertex = (vertex) NULL;
  encodedtri = vertex2tri(endpoint2);
  if (encodedtri != (triangle) NULL) {
    decode(encodedtri, searchtri2);
    org(searchtri2, checkvertex);
  }
  if (checkvertex != endpoint2) {
       searchtri2.tri = m->dummytri;
    searchtri2.orient = 0;
    symself(searchtri2);
       if (locate(m, b, endpoint2, &searchtri2) != ONVERTEX) {
      printf(
        "Internal error in insertsegment():  Unable to locate PSLG vertex\n");
      printf("  (%.12g, %.12g) in triangulation.\n",
             endpoint2[0], endpoint2[1]);
      internalerror();
    }
  }
   otricopy(searchtri2, m->recenttri);
    if (scoutsegment(m, b, &searchtri2, endpoint1, newmark)) {
       return;
  }
    org(searchtri2, endpoint2);

#ifndef REDUCED
#ifndef CDT_ONLY
  if (b->splitseg) {
       conformingedge(m, b, endpoint1, endpoint2, newmark);
  } else {
#endif
#endif
       constrainededge(m, b, &searchtri1, endpoint2, newmark);
#ifndef REDUCED
#ifndef CDT_ONLY
  }
#endif
#endif
}


#ifdef ANSI_DECLARATORS
void markhull(struct mesh *m, struct behavior *b)
#else
void markhull(m, b)
struct mesh *m;
struct behavior *b;
#endif

{
  struct otri hulltri;
  struct otri nexttri;
  struct otri starttri;
  triangle ptr;

   hulltri.tri = m->dummytri;
  hulltri.orient = 0;
  symself(hulltri);
   otricopy(hulltri, starttri);
   do {
       insertsubseg(m, b, &hulltri, 1);
       lnextself(hulltri);
    oprev(hulltri, nexttri);
    while (nexttri.tri != m->dummytri) {
      otricopy(nexttri, hulltri);
      oprev(hulltri, nexttri);
    }
  } while (!otriequal(hulltri, starttri));
}


#ifdef TRILIBRARY

#ifdef ANSI_DECLARATORS
void formskeleton(struct mesh *m, struct behavior *b, int *segmentlist,
                  int *segmentmarkerlist, int numberofsegments)
#else
void formskeleton(m, b, segmentlist, segmentmarkerlist, numberofsegments)
struct mesh *m;
struct behavior *b;
int *segmentlist;
int *segmentmarkerlist;
int numberofsegments;
#endif

#else

#ifdef ANSI_DECLARATORS
void formskeleton(struct mesh *m, struct behavior *b,
                  FILE *polyfile, char *polyfilename)
#else
void formskeleton(m, b, polyfile, polyfilename)
struct mesh *m;
struct behavior *b;
FILE *polyfile;
char *polyfilename;
#endif

#endif

{
#ifdef TRILIBRARY
  char polyfilename[6];
  int index;
#else
  char inputline[INPUTLINESIZE];
  char *stringptr;
#endif
  vertex endpoint1, endpoint2;
  int segmentmarkers;
  int end1, end2;
  int boundmarker;
  int i;

  if (b->poly) {
    if (!b->quiet) {
      printf("Recovering segments in Delaunay triangulation.\n");
    }
#ifdef TRILIBRARY
    strcpy(polyfilename, "input");
    m->insegments = numberofsegments;
    segmentmarkers = segmentmarkerlist != (int *) NULL;
    index = 0;
#else
          stringptr = readline(inputline, polyfile, polyfilename);
    m->insegments = (int) strtol(stringptr, &stringptr, 0);
    stringptr = findfield(stringptr);
    if (*stringptr == '\0') {
      segmentmarkers = 0;
    } else {
      segmentmarkers = (int) strtol(stringptr, &stringptr, 0);
    }
#endif
          if (m->triangles.items == 0) {
      return;
    }

          if (m->insegments > 0) {
      makevertexmap(m, b);
      if (b->verbose) {
        printf("  Recovering PSLG segments.\n");
      }
    }

    boundmarker = 0;
       for (i = 0; i < m->insegments; i++) {
#ifdef TRILIBRARY
      end1 = segmentlist[index++];
      end2 = segmentlist[index++];
      if (segmentmarkers) {
        boundmarker = segmentmarkerlist[i];
      }
#else
      stringptr = readline(inputline, polyfile, b->inpolyfilename);
      stringptr = findfield(stringptr);
      if (*stringptr == '\0') {
        printf("Error:  Segment %d has no endpoints in %s.\n",
               b->firstnumber + i, polyfilename);
        triexit(1);
      } else {
        end1 = (int) strtol(stringptr, &stringptr, 0);
      }
      stringptr = findfield(stringptr);
      if (*stringptr == '\0') {
        printf("Error:  Segment %d is missing its second endpoint in %s.\n",
               b->firstnumber + i, polyfilename);
        triexit(1);
      } else {
        end2 = (int) strtol(stringptr, &stringptr, 0);
      }
      if (segmentmarkers) {
        stringptr = findfield(stringptr);
        if (*stringptr == '\0') {
          boundmarker = 0;
        } else {
          boundmarker = (int) strtol(stringptr, &stringptr, 0);
        }
      }
#endif
      if ((end1 < b->firstnumber) ||
          (end1 >= b->firstnumber + m->invertices)) {
        if (!b->quiet) {
          printf("Warning:  Invalid first endpoint of segment %d in %s.\n",
                 b->firstnumber + i, polyfilename);
        }
      } else if ((end2 < b->firstnumber) ||
                 (end2 >= b->firstnumber + m->invertices)) {
        if (!b->quiet) {
          printf("Warning:  Invalid second endpoint of segment %d in %s.\n",
                 b->firstnumber + i, polyfilename);
        }
      } else {
               endpoint1 = getvertex(m, b, end1);
        endpoint2 = getvertex(m, b, end2);
        if ((endpoint1[0] == endpoint2[0]) && (endpoint1[1] == endpoint2[1])) {
          if (!b->quiet) {
            printf("Warning:  Endpoints of segment %d are coincident in %s.\n",
                   b->firstnumber + i, polyfilename);
          }
        } else {
          insertsegment(m, b, endpoint1, endpoint2, boundmarker);
        }
      }
    }
  } else {
    m->insegments = 0;
  }
  if (b->convex || !b->poly) {
       if (b->verbose) {
      printf("  Enclosing convex hull with segments.\n");
    }
    markhull(m, b);
  }
}




#ifdef ANSI_DECLARATORS
void infecthull(struct mesh *m, struct behavior *b)
#else
void infecthull(m, b)
struct mesh *m;
struct behavior *b;
#endif

{
  struct otri hulltri;
  struct otri nexttri;
  struct otri starttri;
  struct osub hullsubseg;
  triangle **deadtriangle;
  vertex horg, hdest;
  triangle ptr;
  subseg sptr;

  if (b->verbose) {
    printf("  Marking concavities (external triangles) for elimination.\n");
  }
   hulltri.tri = m->dummytri;
  hulltri.orient = 0;
  symself(hulltri);
   otricopy(hulltri, starttri);
   do {
       if (!infected(hulltri)) {
           tspivot(hulltri, hullsubseg);
      if (hullsubseg.ss == m->dummysub) {
               if (!infected(hulltri)) {
          infect(hulltri);
          deadtriangle = (triangle **) poolalloc(&m->viri);
          *deadtriangle = hulltri.tri;
        }
      } else {
               if (mark(hullsubseg) == 0) {
          setmark(hullsubseg, 1);
          org(hulltri, horg);
          dest(hulltri, hdest);
          if (vertexmark(horg) == 0) {
            setvertexmark(horg, 1);
          }
          if (vertexmark(hdest) == 0) {
            setvertexmark(hdest, 1);
          }
        }
      }
    }
       lnextself(hulltri);
    oprev(hulltri, nexttri);
    while (nexttri.tri != m->dummytri) {
      otricopy(nexttri, hulltri);
      oprev(hulltri, nexttri);
    }
  } while (!otriequal(hulltri, starttri));
}


#ifdef ANSI_DECLARATORS
void plague(struct mesh *m, struct behavior *b)
#else
void plague(m, b)
struct mesh *m;
struct behavior *b;
#endif

{
  struct otri testtri;
  struct otri neighbor;
  triangle **virusloop;
  triangle **deadtriangle;
  struct osub neighborsubseg;
  vertex testvertex;
  vertex norg, ndest;
  vertex deadorg, deaddest, deadapex;
  int killorg;
  triangle ptr;
  subseg sptr;

  if (b->verbose) {
    printf("  Marking neighbors of marked triangles.\n");
  }
    traversalinit(&m->viri);
  virusloop = (triangle **) traverse(&m->viri);
  while (virusloop != (triangle **) NULL) {
    testtri.tri = *virusloop;
                uninfect(testtri);
    if (b->verbose > 2) {
                testtri.orient = 0;
      org(testtri, deadorg);
      dest(testtri, deaddest);
      apex(testtri, deadapex);
      printf("    Checking (%.12g, %.12g) (%.12g, %.12g) (%.12g, %.12g)\n",
             deadorg[0], deadorg[1], deaddest[0], deaddest[1],
             deadapex[0], deadapex[1]);
    }
       for (testtri.orient = 0; testtri.orient < 3; testtri.orient++) {
           sym(testtri, neighbor);
           tspivot(testtri, neighborsubseg);
           if ((neighbor.tri == m->dummytri) || infected(neighbor)) {
        if (neighborsubseg.ss != m->dummysub) {
                                     subsegdealloc(m, neighborsubseg.ss);
          if (neighbor.tri != m->dummytri) {
                                  uninfect(neighbor);
            tsdissolve(neighbor);
            infect(neighbor);
          }
        }
      } else {
        if (neighborsubseg.ss == m->dummysub) {
                            if (b->verbose > 2) {
            org(neighbor, deadorg);
            dest(neighbor, deaddest);
            apex(neighbor, deadapex);
            printf(
              "    Marking (%.12g, %.12g) (%.12g, %.12g) (%.12g, %.12g)\n",
                   deadorg[0], deadorg[1], deaddest[0], deaddest[1],
                   deadapex[0], deadapex[1]);
          }
          infect(neighbor);
                   deadtriangle = (triangle **) poolalloc(&m->viri);
          *deadtriangle = neighbor.tri;
        } else {
                   stdissolve(neighborsubseg);
                   if (mark(neighborsubseg) == 0) {
            setmark(neighborsubseg, 1);
          }
          org(neighbor, norg);
          dest(neighbor, ndest);
          if (vertexmark(norg) == 0) {
            setvertexmark(norg, 1);
          }
          if (vertexmark(ndest) == 0) {
            setvertexmark(ndest, 1);
          }
        }
      }
    }
          infect(testtri);
    virusloop = (triangle **) traverse(&m->viri);
  }

  if (b->verbose) {
    printf("  Deleting marked triangles.\n");
  }

  traversalinit(&m->viri);
  virusloop = (triangle **) traverse(&m->viri);
  while (virusloop != (triangle **) NULL) {
    testtri.tri = *virusloop;

             for (testtri.orient = 0; testtri.orient < 3; testtri.orient++) {
      org(testtri, testvertex);
           if (testvertex != (vertex) NULL) {
        killorg = 1;
               setorg(testtri, NULL);
               onext(testtri, neighbor);
               while ((neighbor.tri != m->dummytri) &&
               (!otriequal(neighbor, testtri))) {
          if (infected(neighbor)) {
                       setorg(neighbor, NULL);
          } else {
                       killorg = 0;
          }
                   onextself(neighbor);
        }
               if (neighbor.tri == m->dummytri) {
                   oprev(testtri, neighbor);
                   while (neighbor.tri != m->dummytri) {
            if (infected(neighbor)) {
                         setorg(neighbor, NULL);
            } else {
                           killorg = 0;
            }
                       oprevself(neighbor);
          }
        }
        if (killorg) {
          if (b->verbose > 1) {
            printf("    Deleting vertex (%.12g, %.12g)\n",
                   testvertex[0], testvertex[1]);
          }
          setvertextype(testvertex, UNDEADVERTEX);
          m->undeads++;
        }
      }
    }

          for (testtri.orient = 0; testtri.orient < 3; testtri.orient++) {
      sym(testtri, neighbor);
      if (neighbor.tri == m->dummytri) {
                             m->hullsize--;
      } else {
               dissolve(neighbor);
                      m->hullsize++;
      }
    }
       triangledealloc(m, testtri.tri);
    virusloop = (triangle **) traverse(&m->viri);
  }
   poolrestart(&m->viri);
}


#ifdef ANSI_DECLARATORS
void regionplague(struct mesh *m, struct behavior *b,
                  REAL attribute, REAL area)
#else
void regionplague(m, b, attribute, area)
struct mesh *m;
struct behavior *b;
REAL attribute;
REAL area;
#endif

{
  struct otri testtri;
  struct otri neighbor;
  triangle **virusloop;
  triangle **regiontri;
  struct osub neighborsubseg;
  vertex regionorg, regiondest, regionapex;
  triangle ptr;
  subseg sptr;

  if (b->verbose > 1) {
    printf("  Marking neighbors of marked triangles.\n");
  }
     traversalinit(&m->viri);
  virusloop = (triangle **) traverse(&m->viri);
  while (virusloop != (triangle **) NULL) {
    testtri.tri = *virusloop;
                uninfect(testtri);
    if (b->regionattrib) {
           setelemattribute(testtri, m->eextras, attribute);
    }
    if (b->vararea) {
           setareabound(testtri, area);
    }
    if (b->verbose > 2) {
                testtri.orient = 0;
      org(testtri, regionorg);
      dest(testtri, regiondest);
      apex(testtri, regionapex);
      printf("    Checking (%.12g, %.12g) (%.12g, %.12g) (%.12g, %.12g)\n",
             regionorg[0], regionorg[1], regiondest[0], regiondest[1],
             regionapex[0], regionapex[1]);
    }
       for (testtri.orient = 0; testtri.orient < 3; testtri.orient++) {
           sym(testtri, neighbor);
           tspivot(testtri, neighborsubseg);
                if ((neighbor.tri != m->dummytri) && !infected(neighbor)
          && (neighborsubseg.ss == m->dummysub)) {
        if (b->verbose > 2) {
          org(neighbor, regionorg);
          dest(neighbor, regiondest);
          apex(neighbor, regionapex);
          printf("    Marking (%.12g, %.12g) (%.12g, %.12g) (%.12g, %.12g)\n",
                 regionorg[0], regionorg[1], regiondest[0], regiondest[1],
                 regionapex[0], regionapex[1]);
        }
               infect(neighbor);
               regiontri = (triangle **) poolalloc(&m->viri);
        *regiontri = neighbor.tri;
      }
    }
          infect(testtri);
    virusloop = (triangle **) traverse(&m->viri);
  }

   if (b->verbose > 1) {
    printf("  Unmarking marked triangles.\n");
  }
  traversalinit(&m->viri);
  virusloop = (triangle **) traverse(&m->viri);
  while (virusloop != (triangle **) NULL) {
    testtri.tri = *virusloop;
    uninfect(testtri);
    virusloop = (triangle **) traverse(&m->viri);
  }
   poolrestart(&m->viri);
}


#ifdef ANSI_DECLARATORS
void carveholes(struct mesh *m, struct behavior *b, REAL *holelist, int holes,
                REAL *regionlist, int regions)
#else
void carveholes(m, b, holelist, holes, regionlist, regions)
struct mesh *m;
struct behavior *b;
REAL *holelist;
int holes;
REAL *regionlist;
int regions;
#endif

{
  struct otri searchtri;
  struct otri triangleloop;
  struct otri *regiontris;
  triangle **holetri;
  triangle **regiontri;
  vertex searchorg, searchdest;
  enum locateresult intersect;
  int i;
  triangle ptr;

  if (!(b->quiet || (b->noholes && b->convex))) {
    printf("Removing unwanted triangles.\n");
    if (b->verbose && (holes > 0)) {
      printf("  Marking holes for elimination.\n");
    }
  }

  if (regions > 0) {
       regiontris = (struct otri *) trimalloc(regions *
                                           (int) sizeof(struct otri));
  } else {
    regiontris = (struct otri *) NULL;
  }

  if (((holes > 0) && !b->noholes) || !b->convex || (regions > 0)) {
          poolinit(&m->viri, sizeof(triangle *), VIRUSPERBLOCK, VIRUSPERBLOCK, 0);
  }

  if (!b->convex) {
          infecthull(m, b);
  }

  if ((holes > 0) && !b->noholes) {
       for (i = 0; i < 2 * holes; i += 2) {
           if ((holelist[i] >= m->xmin) && (holelist[i] <= m->xmax)
          && (holelist[i + 1] >= m->ymin) && (holelist[i + 1] <= m->ymax)) {
               searchtri.tri = m->dummytri;
        searchtri.orient = 0;
        symself(searchtri);
                             org(searchtri, searchorg);
        dest(searchtri, searchdest);
        if (counterclockwise(m, b, searchorg, searchdest, &holelist[i]) >
            0.0) {
                   intersect = locate(m, b, &holelist[i], &searchtri);
          if ((intersect != OUTSIDE) && (!infected(searchtri))) {
                                  infect(searchtri);
            holetri = (triangle **) poolalloc(&m->viri);
            *holetri = searchtri.tri;
          }
        }
      }
    }
  }

        if (regions > 0) {
       for (i = 0; i < regions; i++) {
      regiontris[i].tri = m->dummytri;
           if ((regionlist[4 * i] >= m->xmin) && (regionlist[4 * i] <= m->xmax) &&
          (regionlist[4 * i + 1] >= m->ymin) &&
          (regionlist[4 * i + 1] <= m->ymax)) {
               searchtri.tri = m->dummytri;
        searchtri.orient = 0;
        symself(searchtri);
                             org(searchtri, searchorg);
        dest(searchtri, searchdest);
        if (counterclockwise(m, b, searchorg, searchdest, &regionlist[4 * i]) >
            0.0) {
                   intersect = locate(m, b, &regionlist[4 * i], &searchtri);
          if ((intersect != OUTSIDE) && (!infected(searchtri))) {
                                  otricopy(searchtri, regiontris[i]);
          }
        }
      }
    }
  }

  if (m->viri.items > 0) {
       plague(m, b);
  }

  if (regions > 0) {
    if (!b->quiet) {
      if (b->regionattrib) {
        if (b->vararea) {
          printf("Spreading regional attributes and area constraints.\n");
        } else {
          printf("Spreading regional attributes.\n");
        }
      } else {
        printf("Spreading regional area constraints.\n");
      }
    }
    if (b->regionattrib && !b->refine) {
           traversalinit(&m->triangles);
      triangleloop.orient = 0;
      triangleloop.tri = triangletraverse(m);
      while (triangleloop.tri != (triangle *) NULL) {
        setelemattribute(triangleloop, m->eextras, 0.0);
        triangleloop.tri = triangletraverse(m);
      }
    }
    for (i = 0; i < regions; i++) {
      if (regiontris[i].tri != m->dummytri) {
                      if (!deadtri(regiontris[i].tri)) {
                   infect(regiontris[i]);
          regiontri = (triangle **) poolalloc(&m->viri);
          *regiontri = regiontris[i].tri;
                   regionplague(m, b, regionlist[4 * i + 2], regionlist[4 * i + 3]);
                 }
      }
    }
    if (b->regionattrib && !b->refine) {
           m->eextras++;
    }
  }

   if (((holes > 0) && !b->noholes) || !b->convex || (regions > 0)) {
    pooldeinit(&m->viri);
  }
  if (regions > 0) {
    trifree((VOID *) regiontris);
  }
}




#ifndef CDT_ONLY

#ifdef ANSI_DECLARATORS
void tallyencs(struct mesh *m, struct behavior *b)
#else
void tallyencs(m, b)
struct mesh *m;
struct behavior *b;
#endif

{
  struct osub subsegloop;
  int dummy;

  traversalinit(&m->subsegs);
  subsegloop.ssorient = 0;
  subsegloop.ss = subsegtraverse(m);
  while (subsegloop.ss != (subseg *) NULL) {
       dummy = checkseg4encroach(m, b, &subsegloop);
    subsegloop.ss = subsegtraverse(m);
  }
}

#endif


#ifndef CDT_ONLY

void precisionerror()
{
  printf("Try increasing the area criterion and/or reducing the minimum\n");
  printf("  allowable angle so that tiny triangles are not created.\n");
#ifdef SINGLE
  printf("Alternatively, try recompiling me with double precision\n");
  printf("  arithmetic (by removing \"#define SINGLE\" from the\n");
  printf("  source file or \"-DSINGLE\" from the makefile).\n");
#endif
}

#endif


#ifndef CDT_ONLY

#ifdef ANSI_DECLARATORS
void splitencsegs(struct mesh *m, struct behavior *b, int triflaws)
#else
void splitencsegs(m, b, triflaws)
struct mesh *m;
struct behavior *b;
int triflaws;
#endif

{
  struct otri enctri;
  struct otri testtri;
  struct osub testsh;
  struct osub currentenc;
  struct badsubseg *encloop;
  vertex eorg, edest, eapex;
  vertex newvertex;
  enum insertvertexresult success;
  REAL segmentlength, nearestpoweroftwo;
  REAL split;
  REAL multiplier, divisor;
  int acuteorg, acuteorg2, acutedest, acutedest2;
  int dummy;
  int i;
  triangle ptr;
  subseg sptr;

    while ((m->badsubsegs.items > 0) && (m->steinerleft != 0)) {
    traversalinit(&m->badsubsegs);
    encloop = badsubsegtraverse(m);
    while ((encloop != (struct badsubseg *) NULL) && (m->steinerleft != 0)) {
      sdecode(encloop->encsubseg, currentenc);
      sorg(currentenc, eorg);
      sdest(currentenc, edest);
                          if (!deadsubseg(currentenc.ss) &&
          (eorg == encloop->subsegorg) && (edest == encloop->subsegdest)) {

               stpivot(currentenc, enctri);
        lnext(enctri, testtri);
        tspivot(testtri, testsh);
        acuteorg = testsh.ss != m->dummysub;
               lnextself(testtri);
        tspivot(testtri, testsh);
        acutedest = testsh.ss != m->dummysub;

                             if (!b->conformdel && !acuteorg && !acutedest) {
          apex(enctri, eapex);
          while ((vertextype(eapex) == FREEVERTEX) &&
                 ((eorg[0] - eapex[0]) * (edest[0] - eapex[0]) +
                  (eorg[1] - eapex[1]) * (edest[1] - eapex[1]) < 0.0)) {
            deletevertex(m, b, &testtri);
            stpivot(currentenc, enctri);
            apex(enctri, eapex);
            lprev(enctri, testtri);
          }
        }

                      sym(enctri, testtri);
        if (testtri.tri != m->dummytri) {
                   lnextself(testtri);
          tspivot(testtri, testsh);
          acutedest2 = testsh.ss != m->dummysub;
          acutedest = acutedest || acutedest2;
                   lnextself(testtri);
          tspivot(testtri, testsh);
          acuteorg2 = testsh.ss != m->dummysub;
          acuteorg = acuteorg || acuteorg2;

                   if (!b->conformdel && !acuteorg2 && !acutedest2) {
            org(testtri, eapex);
            while ((vertextype(eapex) == FREEVERTEX) &&
                   ((eorg[0] - eapex[0]) * (edest[0] - eapex[0]) +
                    (eorg[1] - eapex[1]) * (edest[1] - eapex[1]) < 0.0)) {
              deletevertex(m, b, &testtri);
              sym(enctri, testtri);
              apex(testtri, eapex);
              lprevself(testtri);
            }
          }
        }

                      if (acuteorg || acutedest) {
          segmentlength = sqrt((edest[0] - eorg[0]) * (edest[0] - eorg[0]) +
                               (edest[1] - eorg[1]) * (edest[1] - eorg[1]));
                            nearestpoweroftwo = 1.0;
          while (segmentlength > 3.0 * nearestpoweroftwo) {
            nearestpoweroftwo *= 2.0;
          }
          while (segmentlength < 1.5 * nearestpoweroftwo) {
            nearestpoweroftwo *= 0.5;
          }
                   split = nearestpoweroftwo / segmentlength;
          if (acutedest) {
            split = 1.0 - split;
          }
        } else {
                            split = 0.5;
        }

               newvertex = (vertex) poolalloc(&m->vertices);
               for (i = 0; i < 2 + m->nextras; i++) {
          newvertex[i] = eorg[i] + split * (edest[i] - eorg[i]);
        }

        if (!b->noexact) {
                                     multiplier = counterclockwise(m, b, eorg, edest, newvertex);
          divisor = ((eorg[0] - edest[0]) * (eorg[0] - edest[0]) +
                     (eorg[1] - edest[1]) * (eorg[1] - edest[1]));
          if ((multiplier != 0.0) && (divisor != 0.0)) {
            multiplier = multiplier / divisor;
                       if (multiplier == multiplier) {
              newvertex[0] += multiplier * (edest[1] - eorg[1]);
              newvertex[1] += multiplier * (eorg[0] - edest[0]);
            }
          }
        }

        setvertexmark(newvertex, mark(currentenc));
        setvertextype(newvertex, SEGMENTVERTEX);
        if (b->verbose > 1) {
          printf(
  "  Splitting subsegment (%.12g, %.12g) (%.12g, %.12g) at (%.12g, %.12g).\n",
                 eorg[0], eorg[1], edest[0], edest[1],
                 newvertex[0], newvertex[1]);
        }
               if (((newvertex[0] == eorg[0]) && (newvertex[1] == eorg[1])) ||
            ((newvertex[0] == edest[0]) && (newvertex[1] == edest[1]))) {
          printf("Error:  Ran out of precision at (%.12g, %.12g).\n",
                 newvertex[0], newvertex[1]);
          printf("I attempted to split a segment to a smaller size than\n");
          printf("  can be accommodated by the finite precision of\n");
          printf("  floating point arithmetic.\n");
          precisionerror();
          triexit(1);
        }
               success = insertvertex(m, b, newvertex, &enctri, &currentenc,
                               1, triflaws);
        if ((success != SUCCESSFULVERTEX) && (success != ENCROACHINGVERTEX)) {
          printf("Internal error in splitencsegs():\n");
          printf("  Failure to split a segment.\n");
          internalerror();
        }
        if (m->steinerleft > 0) {
          m->steinerleft--;
        }
               dummy = checkseg4encroach(m, b, &currentenc);
        snextself(currentenc);
        dummy = checkseg4encroach(m, b, &currentenc);
      }

      badsubsegdealloc(m, encloop);
      encloop = badsubsegtraverse(m);
    }
  }
}

#endif


#ifndef CDT_ONLY

#ifdef ANSI_DECLARATORS
void tallyfaces(struct mesh *m, struct behavior *b)
#else
void tallyfaces(m, b)
struct mesh *m;
struct behavior *b;
#endif

{
  struct otri triangleloop;

  if (b->verbose) {
    printf("  Making a list of bad triangles.\n");
  }
  traversalinit(&m->triangles);
  triangleloop.orient = 0;
  triangleloop.tri = triangletraverse(m);
  while (triangleloop.tri != (triangle *) NULL) {
       testtriangle(m, b, &triangleloop);
    triangleloop.tri = triangletraverse(m);
  }
}

#endif


#ifndef CDT_ONLY

#ifdef ANSI_DECLARATORS
void splittriangle(struct mesh *m, struct behavior *b,
                   struct badtriang *badtri)
#else
void splittriangle(m, b, badtri)
struct mesh *m;
struct behavior *b;
struct badtriang *badtri;
#endif

{
  struct otri badotri;
  vertex borg, bdest, bapex;
  vertex newvertex;
  REAL xi, eta;
  enum insertvertexresult success;
  int errorflag;
  int i;

  decode(badtri->poortri, badotri);
  org(badotri, borg);
  dest(badotri, bdest);
  apex(badotri, bapex);
     if (!deadtri(badotri.tri) && (borg == badtri->triangorg) &&
      (bdest == badtri->triangdest) && (bapex == badtri->triangapex)) {
    if (b->verbose > 1) {
      printf("  Splitting this triangle at its circumcenter:\n");
      printf("    (%.12g, %.12g) (%.12g, %.12g) (%.12g, %.12g)\n", borg[0],
             borg[1], bdest[0], bdest[1], bapex[0], bapex[1]);
    }

    errorflag = 0;
       newvertex = (vertex) poolalloc(&m->vertices);
    findcircumcenter(m, b, borg, bdest, bapex, newvertex, &xi, &eta, 1);

       if (((newvertex[0] == borg[0]) && (newvertex[1] == borg[1])) ||
        ((newvertex[0] == bdest[0]) && (newvertex[1] == bdest[1])) ||
        ((newvertex[0] == bapex[0]) && (newvertex[1] == bapex[1]))) {
      if (!b->quiet) {
        printf(
             "Warning:  New vertex (%.12g, %.12g) falls on existing vertex.\n",
               newvertex[0], newvertex[1]);
        errorflag = 1;
      }
      vertexdealloc(m, newvertex);
    } else {
      for (i = 2; i < 2 + m->nextras; i++) {
               newvertex[i] = borg[i] + xi * (bdest[i] - borg[i])
                              + eta * (bapex[i] - borg[i]);
      }
                setvertexmark(newvertex, 0);
      setvertextype(newvertex, FREEVERTEX);

                                         if (eta < xi) {
        lprevself(badotri);
      }

                success = insertvertex(m, b, newvertex, &badotri, (struct osub *) NULL,
                             1, 1);
      if (success == SUCCESSFULVERTEX) {
        if (m->steinerleft > 0) {
          m->steinerleft--;
        }
      } else if (success == ENCROACHINGVERTEX) {
                      undovertex(m, b);
        if (b->verbose > 1) {
          printf("  Rejecting (%.12g, %.12g).\n", newvertex[0], newvertex[1]);
        }
        vertexdealloc(m, newvertex);
      } else if (success == VIOLATINGVERTEX) {
                      vertexdealloc(m, newvertex);
      } else {
               if (!b->quiet) {
          printf(
            "Warning:  New vertex (%.12g, %.12g) falls on existing vertex.\n",
                 newvertex[0], newvertex[1]);
          errorflag = 1;
        }
        vertexdealloc(m, newvertex);
      }
    }
    if (errorflag) {
      if (b->verbose) {
        printf("  The new vertex is at the circumcenter of triangle\n");
        printf("    (%.12g, %.12g) (%.12g, %.12g) (%.12g, %.12g)\n",
               borg[0], borg[1], bdest[0], bdest[1], bapex[0], bapex[1]);
      }
      printf("This probably means that I am trying to refine triangles\n");
      printf("  to a smaller size than can be accommodated by the finite\n");
      printf("  precision of floating point arithmetic.  (You can be\n");
      printf("  sure of this if I fail to terminate.)\n");
      precisionerror();
    }
  }
}

#endif


#ifndef CDT_ONLY

#ifdef ANSI_DECLARATORS
void enforcequality(struct mesh *m, struct behavior *b)
#else
void enforcequality(m, b)
struct mesh *m;
struct behavior *b;
#endif

{
  struct badtriang *badtri;
  int i;

  if (!b->quiet) {
    printf("Adding Steiner points to enforce quality.\n");
  }
   poolinit(&m->badsubsegs, sizeof(struct badsubseg), BADSUBSEGPERBLOCK,
           BADSUBSEGPERBLOCK, 0);
  if (b->verbose) {
    printf("  Looking for encroached subsegments.\n");
  }
   tallyencs(m, b);
  if (b->verbose && (m->badsubsegs.items > 0)) {
    printf("  Splitting encroached subsegments.\n");
  }
   splitencsegs(m, b, 0);

   if ((b->minangle > 0.0) || b->vararea || b->fixedarea || b->usertest) {
       poolinit(&m->badtriangles, sizeof(struct badtriang), BADTRIPERBLOCK,
             BADTRIPERBLOCK, 0);
       for (i = 0; i < 4096; i++) {
      m->queuefront[i] = (struct badtriang *) NULL;
    }
    m->firstnonemptyq = -1;
       tallyfaces(m, b);
       poolinit(&m->flipstackers, sizeof(struct flipstacker), FLIPSTACKERPERBLOCK,
             FLIPSTACKERPERBLOCK, 0);
    m->checkquality = 1;
    if (b->verbose) {
      printf("  Splitting bad triangles.\n");
    }
    while ((m->badtriangles.items > 0) && (m->steinerleft != 0)) {
           badtri = dequeuebadtriang(m);
      splittriangle(m, b, badtri);
      if (m->badsubsegs.items > 0) {
               enqueuebadtriang(m, b, badtri);
                      splitencsegs(m, b, 1);
      } else {
               pooldealloc(&m->badtriangles, (VOID *) badtri);
      }
    }
  }

   if (!b->quiet && b->conformdel && (m->badsubsegs.items > 0) &&
      (m->steinerleft == 0)) {
    printf("\nWarning:  I ran out of Steiner points, but the mesh has\n");
    if (m->badsubsegs.items == 1) {
      printf("  one encroached subsegment, and therefore might not be truly\n"
             );
    } else {
      printf("  %ld encroached subsegments, and therefore might not be truly\n"
             , m->badsubsegs.items);
    }
    printf("  Delaunay.  If the Delaunay property is important to you,\n");
    printf("  try increasing the number of Steiner points (controlled by\n");
    printf("  the -S switch) slightly and try again.\n\n");
  }
}

#endif



#ifdef ANSI_DECLARATORS
void highorder(struct mesh *m, struct behavior *b)
#else
void highorder(m, b)
struct mesh *m;
struct behavior *b;
#endif

{
  struct otri triangleloop, trisym;
  struct osub checkmark;
  vertex newvertex;
  vertex torg, tdest;
  int i;
  triangle ptr;
  subseg sptr;

  if (!b->quiet) {
    printf("Adding vertices for second-order triangles.\n");
  }
       m->vertices.deaditemstack = (VOID *) NULL;

  traversalinit(&m->triangles);
  triangleloop.tri = triangletraverse(m);
        while (triangleloop.tri != (triangle *) NULL) {
    for (triangleloop.orient = 0; triangleloop.orient < 3;
         triangleloop.orient++) {
      sym(triangleloop, trisym);
      if ((triangleloop.tri < trisym.tri) || (trisym.tri == m->dummytri)) {
        org(triangleloop, torg);
        dest(triangleloop, tdest);
                      newvertex = (vertex) poolalloc(&m->vertices);
        for (i = 0; i < 2 + m->nextras; i++) {
          newvertex[i] = 0.5 * (torg[i] + tdest[i]);
        }
                      setvertexmark(newvertex, trisym.tri == m->dummytri);
        setvertextype(newvertex,
                      trisym.tri == m->dummytri ? FREEVERTEX : SEGMENTVERTEX);
        if (b->usesegments) {
          tspivot(triangleloop, checkmark);
                   if (checkmark.ss != m->dummysub) {
            setvertexmark(newvertex, mark(checkmark));
            setvertextype(newvertex, SEGMENTVERTEX);
          }
        }
        if (b->verbose > 1) {
          printf("  Creating (%.12g, %.12g).\n", newvertex[0], newvertex[1]);
        }
               triangleloop.tri[m->highorderindex + triangleloop.orient] =
                (triangle) newvertex;
        if (trisym.tri != m->dummytri) {
          trisym.tri[m->highorderindex + trisym.orient] = (triangle) newvertex;
        }
      }
    }
    triangleloop.tri = triangletraverse(m);
  }
}



#ifndef TRILIBRARY

#ifdef ANSI_DECLARATORS
char *readline(char *string, FILE *infile, char *infilename)
#else
char *readline(string, infile, infilename)
char *string;
FILE *infile;
char *infilename;
#endif

{
  char *result;

   do {
    result = fgets(string, INPUTLINESIZE, infile);
    if (result == (char *) NULL) {
      printf("  Error:  Unexpected end of file in %s.\n", infilename);
      triexit(1);
    }
          while ((*result != '\0') && (*result != '#')
           && (*result != '.') && (*result != '+') && (*result != '-')
           && ((*result < '0') || (*result > '9'))) {
      result++;
    }
   } while ((*result == '#') || (*result == '\0'));
  return result;
}

#endif


#ifndef TRILIBRARY

#ifdef ANSI_DECLARATORS
char *findfield(char *string)
#else
char *findfield(string)
char *string;
#endif

{
  char *result;

  result = string;
   while ((*result != '\0') && (*result != '#')
         && (*result != ' ') && (*result != '\t')) {
    result++;
  }
    while ((*result != '\0') && (*result != '#')
         && (*result != '.') && (*result != '+') && (*result != '-')
         && ((*result < '0') || (*result > '9'))) {
    result++;
  }
   if (*result == '#') {
    *result = '\0';
  }
  return result;
}

#endif


#ifndef TRILIBRARY

#ifdef ANSI_DECLARATORS
void readnodes(struct mesh *m, struct behavior *b, char *nodefilename,
               char *polyfilename, FILE **polyfile)
#else
void readnodes(m, b, nodefilename, polyfilename, polyfile)
struct mesh *m;
struct behavior *b;
char *nodefilename;
char *polyfilename;
FILE **polyfile;
#endif

{
  FILE *infile;
  vertex vertexloop;
  char inputline[INPUTLINESIZE];
  char *stringptr;
  char *infilename;
  REAL x, y;
  int firstnode;
  int nodemarkers;
  int currentmarker;
  int i, j;

  if (b->poly) {
       if (!b->quiet) {
      printf("Opening %s.\n", polyfilename);
    }
    *polyfile = fopen(polyfilename, "r");
    if (*polyfile == (FILE *) NULL) {
      printf("  Error:  Cannot access file %s.\n", polyfilename);
      triexit(1);
    }
          stringptr = readline(inputline, *polyfile, polyfilename);
    m->invertices = (int) strtol(stringptr, &stringptr, 0);
    stringptr = findfield(stringptr);
    if (*stringptr == '\0') {
      m->mesh_dim = 2;
    } else {
      m->mesh_dim = (int) strtol(stringptr, &stringptr, 0);
    }
    stringptr = findfield(stringptr);
    if (*stringptr == '\0') {
      m->nextras = 0;
    } else {
      m->nextras = (int) strtol(stringptr, &stringptr, 0);
    }
    stringptr = findfield(stringptr);
    if (*stringptr == '\0') {
      nodemarkers = 0;
    } else {
      nodemarkers = (int) strtol(stringptr, &stringptr, 0);
    }
    if (m->invertices > 0) {
      infile = *polyfile;
      infilename = polyfilename;
      m->readnodefile = 0;
    } else {
                m->readnodefile = 1;
      infilename = nodefilename;
    }
  } else {
    m->readnodefile = 1;
    infilename = nodefilename;
    *polyfile = (FILE *) NULL;
  }

  if (m->readnodefile) {
       if (!b->quiet) {
      printf("Opening %s.\n", nodefilename);
    }
    infile = fopen(nodefilename, "r");
    if (infile == (FILE *) NULL) {
      printf("  Error:  Cannot access file %s.\n", nodefilename);
      triexit(1);
    }
          stringptr = readline(inputline, infile, nodefilename);
    m->invertices = (int) strtol(stringptr, &stringptr, 0);
    stringptr = findfield(stringptr);
    if (*stringptr == '\0') {
      m->mesh_dim = 2;
    } else {
      m->mesh_dim = (int) strtol(stringptr, &stringptr, 0);
    }
    stringptr = findfield(stringptr);
    if (*stringptr == '\0') {
      m->nextras = 0;
    } else {
      m->nextras = (int) strtol(stringptr, &stringptr, 0);
    }
    stringptr = findfield(stringptr);
    if (*stringptr == '\0') {
      nodemarkers = 0;
    } else {
      nodemarkers = (int) strtol(stringptr, &stringptr, 0);
    }
  }

  if (m->invertices < 3) {
    printf("Error:  Input must have at least three input vertices.\n");
    triexit(1);
  }
  if (m->mesh_dim != 2) {
    printf("Error:  Triangle only works with two-dimensional meshes.\n");
    triexit(1);
  }
  if (m->nextras == 0) {
    b->weighted = 0;
  }

  initializevertexpool(m, b);

   for (i = 0; i < m->invertices; i++) {
    vertexloop = (vertex) poolalloc(&m->vertices);
    stringptr = readline(inputline, infile, infilename);
    if (i == 0) {
      firstnode = (int) strtol(stringptr, &stringptr, 0);
      if ((firstnode == 0) || (firstnode == 1)) {
        b->firstnumber = firstnode;
      }
    }
    stringptr = findfield(stringptr);
    if (*stringptr == '\0') {
      printf("Error:  Vertex %d has no x coordinate.\n", b->firstnumber + i);
      triexit(1);
    }
    x = (REAL) strtod(stringptr, &stringptr);
    stringptr = findfield(stringptr);
    if (*stringptr == '\0') {
      printf("Error:  Vertex %d has no y coordinate.\n", b->firstnumber + i);
      triexit(1);
    }
    y = (REAL) strtod(stringptr, &stringptr);
    vertexloop[0] = x;
    vertexloop[1] = y;
       for (j = 2; j < 2 + m->nextras; j++) {
      stringptr = findfield(stringptr);
      if (*stringptr == '\0') {
        vertexloop[j] = 0.0;
      } else {
        vertexloop[j] = (REAL) strtod(stringptr, &stringptr);
      }
    }
    if (nodemarkers) {
           stringptr = findfield(stringptr);
      if (*stringptr == '\0') {
        setvertexmark(vertexloop, 0);
      } else {
        currentmarker = (int) strtol(stringptr, &stringptr, 0);
        setvertexmark(vertexloop, currentmarker);
      }
    } else {
           setvertexmark(vertexloop, 0);
    }
    setvertextype(vertexloop, INPUTVERTEX);
    if (b->poly) {
           setvertex2tri(vertexloop, (triangle) NULL);
    }
       if (i == 0) {
      m->xmin = m->xmax = x;
      m->ymin = m->ymax = y;
    } else {
      m->xmin = (x < m->xmin) ? x : m->xmin;
      m->xmax = (x > m->xmax) ? x : m->xmax;
      m->ymin = (y < m->ymin) ? y : m->ymin;
      m->ymax = (y > m->ymax) ? y : m->ymax;
    }
  }
  if (m->readnodefile) {
    fclose(infile);
  }

    m->xminextreme = 10 * m->xmin - 9 * m->xmax;
}

#endif


#ifdef TRILIBRARY

#ifdef ANSI_DECLARATORS
void transfernodes(struct mesh *m, struct behavior *b, REAL *pointlist,
                   REAL *pointattriblist, int *pointmarkerlist,
                   int numberofpoints, int numberofpointattribs)
#else
void transfernodes(m, b, pointlist, pointattriblist, pointmarkerlist,
                   numberofpoints, numberofpointattribs)
struct mesh *m;
struct behavior *b;
REAL *pointlist;
REAL *pointattriblist;
int *pointmarkerlist;
int numberofpoints;
int numberofpointattribs;
#endif

{
  vertex vertexloop;
  REAL x, y;
  int i, j;
  int coordindex;
  int attribindex;

  m->invertices = numberofpoints;
  m->mesh_dim = 2;
  m->nextras = numberofpointattribs;
  m->readnodefile = 0;
  if (m->invertices < 3) {
    printf("Error:  Input must have at least three input vertices.\n");
    triexit(1);
  }
  if (m->nextras == 0) {
    b->weighted = 0;
  }

  initializevertexpool(m, b);

   coordindex = 0;
  attribindex = 0;
  for (i = 0; i < m->invertices; i++) {
    vertexloop = (vertex) poolalloc(&m->vertices);
       x = vertexloop[0] = pointlist[coordindex++];
    y = vertexloop[1] = pointlist[coordindex++];
       for (j = 0; j < numberofpointattribs; j++) {
      vertexloop[2 + j] = pointattriblist[attribindex++];
    }
    if (pointmarkerlist != (int *) NULL) {
           setvertexmark(vertexloop, pointmarkerlist[i]);
    } else {
           setvertexmark(vertexloop, 0);
    }
    setvertextype(vertexloop, INPUTVERTEX);
    if (b->poly) {
           setvertex2tri(vertexloop, (triangle) NULL);
    }
       if (i == 0) {
      m->xmin = m->xmax = x;
      m->ymin = m->ymax = y;
    } else {
      m->xmin = (x < m->xmin) ? x : m->xmin;
      m->xmax = (x > m->xmax) ? x : m->xmax;
      m->ymin = (y < m->ymin) ? y : m->ymin;
      m->ymax = (y > m->ymax) ? y : m->ymax;
    }
  }

    m->xminextreme = 10 * m->xmin - 9 * m->xmax;
}

#endif


#ifndef TRILIBRARY

#ifdef ANSI_DECLARATORS
void readholes(struct mesh *m, struct behavior *b,
               FILE *polyfile, char *polyfilename, REAL **hlist, int *holes,
               REAL **rlist, int *regions)
#else
void readholes(m, b, polyfile, polyfilename, hlist, holes, rlist, regions)
struct mesh *m;
struct behavior *b;
FILE *polyfile;
char *polyfilename;
REAL **hlist;
int *holes;
REAL **rlist;
int *regions;
#endif

{
  REAL *holelist;
  REAL *regionlist;
  char inputline[INPUTLINESIZE];
  char *stringptr;
  int index;
  int i;

   stringptr = readline(inputline, polyfile, polyfilename);
  *holes = (int) strtol(stringptr, &stringptr, 0);
  if (*holes > 0) {
    holelist = (REAL *) trimalloc(2 * *holes * (int) sizeof(REAL));
    *hlist = holelist;
    for (i = 0; i < 2 * *holes; i += 2) {
      stringptr = readline(inputline, polyfile, polyfilename);
      stringptr = findfield(stringptr);
      if (*stringptr == '\0') {
        printf("Error:  Hole %d has no x coordinate.\n",
               b->firstnumber + (i >> 1));
        triexit(1);
      } else {
        holelist[i] = (REAL) strtod(stringptr, &stringptr);
      }
      stringptr = findfield(stringptr);
      if (*stringptr == '\0') {
        printf("Error:  Hole %d has no y coordinate.\n",
               b->firstnumber + (i >> 1));
        triexit(1);
      } else {
        holelist[i + 1] = (REAL) strtod(stringptr, &stringptr);
      }
    }
  } else {
    *hlist = (REAL *) NULL;
  }

#ifndef CDT_ONLY
  if ((b->regionattrib || b->vararea) && !b->refine) {
       stringptr = readline(inputline, polyfile, polyfilename);
    *regions = (int) strtol(stringptr, &stringptr, 0);
    if (*regions > 0) {
      regionlist = (REAL *) trimalloc(4 * *regions * (int) sizeof(REAL));
      *rlist = regionlist;
      index = 0;
      for (i = 0; i < *regions; i++) {
        stringptr = readline(inputline, polyfile, polyfilename);
        stringptr = findfield(stringptr);
        if (*stringptr == '\0') {
          printf("Error:  Region %d has no x coordinate.\n",
                 b->firstnumber + i);
          triexit(1);
        } else {
          regionlist[index++] = (REAL) strtod(stringptr, &stringptr);
        }
        stringptr = findfield(stringptr);
        if (*stringptr == '\0') {
          printf("Error:  Region %d has no y coordinate.\n",
                 b->firstnumber + i);
          triexit(1);
        } else {
          regionlist[index++] = (REAL) strtod(stringptr, &stringptr);
        }
        stringptr = findfield(stringptr);
        if (*stringptr == '\0') {
          printf(
            "Error:  Region %d has no region attribute or area constraint.\n",
                 b->firstnumber + i);
          triexit(1);
        } else {
          regionlist[index++] = (REAL) strtod(stringptr, &stringptr);
        }
        stringptr = findfield(stringptr);
        if (*stringptr == '\0') {
          regionlist[index] = regionlist[index - 1];
        } else {
          regionlist[index] = (REAL) strtod(stringptr, &stringptr);
        }
        index++;
      }
    }
  } else {
       *regions = 0;
    *rlist = (REAL *) NULL;
  }
#endif

  fclose(polyfile);
}

#endif


#ifndef TRILIBRARY

#ifdef ANSI_DECLARATORS
void finishfile(FILE *outfile, int argc, char **argv)
#else
void finishfile(outfile, argc, argv)
FILE *outfile;
int argc;
char **argv;
#endif

{
  int i;

  fprintf(outfile, "# Generated by");
  for (i = 0; i < argc; i++) {
    fprintf(outfile, " ");
    fputs(argv[i], outfile);
  }
  fprintf(outfile, "\n");
  fclose(outfile);
}

#endif


#ifdef TRILIBRARY

#ifdef ANSI_DECLARATORS
void writenodes(struct mesh *m, struct behavior *b, REAL **pointlist,
                REAL **pointattriblist, int **pointmarkerlist)
#else
void writenodes(m, b, pointlist, pointattriblist, pointmarkerlist)
struct mesh *m;
struct behavior *b;
REAL **pointlist;
REAL **pointattriblist;
int **pointmarkerlist;
#endif

#else

#ifdef ANSI_DECLARATORS
void writenodes(struct mesh *m, struct behavior *b, char *nodefilename,
                int argc, char **argv)
#else
void writenodes(m, b, nodefilename, argc, argv)
struct mesh *m;
struct behavior *b;
char *nodefilename;
int argc;
char **argv;
#endif

#endif

{
#ifdef TRILIBRARY
  REAL *plist;
  REAL *palist;
  int *pmlist;
  int coordindex;
  int attribindex;
#else
  FILE *outfile;
#endif
  vertex vertexloop;
  long outvertices;
  int vertexnumber;
  int i;

  if (b->jettison) {
    outvertices = m->vertices.items - m->undeads;
  } else {
    outvertices = m->vertices.items;
  }

#ifdef TRILIBRARY
  if (!b->quiet) {
    printf("Writing vertices.\n");
  }
   if (*pointlist == (REAL *) NULL) {
    *pointlist = (REAL *) trimalloc((int) (outvertices * 2 * sizeof(REAL)));
  }
   if ((m->nextras > 0) && (*pointattriblist == (REAL *) NULL)) {
    *pointattriblist = (REAL *) trimalloc((int) (outvertices * m->nextras *
                                                 sizeof(REAL)));
  }
   if (!b->nobound && (*pointmarkerlist == (int *) NULL)) {
    *pointmarkerlist = (int *) trimalloc((int) (outvertices * sizeof(int)));
  }
  plist = *pointlist;
  palist = *pointattriblist;
  pmlist = *pointmarkerlist;
  coordindex = 0;
  attribindex = 0;
#else
  if (!b->quiet) {
    printf("Writing %s.\n", nodefilename);
  }
  outfile = fopen(nodefilename, "w");
  if (outfile == (FILE *) NULL) {
    printf("  Error:  Cannot create file %s.\n", nodefilename);
    triexit(1);
  }
    fprintf(outfile, "%ld  %d  %d  %d\n", outvertices, m->mesh_dim,
          m->nextras, 1 - b->nobound);
#endif

  traversalinit(&m->vertices);
  vertexnumber = b->firstnumber;
  vertexloop = vertextraverse(m);
  while (vertexloop != (vertex) NULL) {
    if (!b->jettison || (vertextype(vertexloop) != UNDEADVERTEX)) {
#ifdef TRILIBRARY
           plist[coordindex++] = vertexloop[0];
      plist[coordindex++] = vertexloop[1];
           for (i = 0; i < m->nextras; i++) {
        palist[attribindex++] = vertexloop[2 + i];
      }
      if (!b->nobound) {
               pmlist[vertexnumber - b->firstnumber] = vertexmark(vertexloop);
      }
#else
           fprintf(outfile, "%4d    %.17g  %.17g", vertexnumber, vertexloop[0],
              vertexloop[1]);
      for (i = 0; i < m->nextras; i++) {
               fprintf(outfile, "  %.17g", vertexloop[i + 2]);
      }
      if (b->nobound) {
        fprintf(outfile, "\n");
      } else {
               fprintf(outfile, "    %d\n", vertexmark(vertexloop));
      }
#endif

      setvertexmark(vertexloop, vertexnumber);
      vertexnumber++;
    }
    vertexloop = vertextraverse(m);
  }

#ifndef TRILIBRARY
  finishfile(outfile, argc, argv);
#endif
}


#ifdef ANSI_DECLARATORS
void numbernodes(struct mesh *m, struct behavior *b)
#else
void numbernodes(m, b)
struct mesh *m;
struct behavior *b;
#endif

{
  vertex vertexloop;
  int vertexnumber;

  traversalinit(&m->vertices);
  vertexnumber = b->firstnumber;
  vertexloop = vertextraverse(m);
  while (vertexloop != (vertex) NULL) {
    setvertexmark(vertexloop, vertexnumber);
    if (!b->jettison || (vertextype(vertexloop) != UNDEADVERTEX)) {
      vertexnumber++;
    }
    vertexloop = vertextraverse(m);
  }
}


#ifdef TRILIBRARY

#ifdef ANSI_DECLARATORS
void writeelements(struct mesh *m, struct behavior *b,
                   int **trianglelist, REAL **triangleattriblist)
#else
void writeelements(m, b, trianglelist, triangleattriblist)
struct mesh *m;
struct behavior *b;
int **trianglelist;
REAL **triangleattriblist;
#endif

#else

#ifdef ANSI_DECLARATORS
void writeelements(struct mesh *m, struct behavior *b, char *elefilename,
                   int argc, char **argv)
#else
void writeelements(m, b, elefilename, argc, argv)
struct mesh *m;
struct behavior *b;
char *elefilename;
int argc;
char **argv;
#endif

#endif

{
#ifdef TRILIBRARY
  int *tlist;
  REAL *talist;
  int vertexindex;
  int attribindex;
#else
  FILE *outfile;
#endif
  struct otri triangleloop;
  vertex p1, p2, p3;
  vertex mid1, mid2, mid3;
  long elementnumber;
  int i;

#ifdef TRILIBRARY
  if (!b->quiet) {
    printf("Writing triangles.\n");
  }
   if (*trianglelist == (int *) NULL) {
    *trianglelist = (int *) trimalloc((int) (m->triangles.items *
                                             ((b->order + 1) * (b->order + 2) /
                                              2) * sizeof(int)));
  }
   if ((m->eextras > 0) && (*triangleattriblist == (REAL *) NULL)) {
    *triangleattriblist = (REAL *) trimalloc((int) (m->triangles.items *
                                                    m->eextras *
                                                    sizeof(REAL)));
  }
  tlist = *trianglelist;
  talist = *triangleattriblist;
  vertexindex = 0;
  attribindex = 0;
#else
  if (!b->quiet) {
    printf("Writing %s.\n", elefilename);
  }
  outfile = fopen(elefilename, "w");
  if (outfile == (FILE *) NULL) {
    printf("  Error:  Cannot create file %s.\n", elefilename);
    triexit(1);
  }
   fprintf(outfile, "%ld  %d  %d\n", m->triangles.items,
          (b->order + 1) * (b->order + 2) / 2, m->eextras);
#endif

  traversalinit(&m->triangles);
  triangleloop.tri = triangletraverse(m);
  triangleloop.orient = 0;
  elementnumber = b->firstnumber;
  while (triangleloop.tri != (triangle *) NULL) {
    org(triangleloop, p1);
    dest(triangleloop, p2);
    apex(triangleloop, p3);
    if (b->order == 1) {
#ifdef TRILIBRARY
      tlist[vertexindex++] = vertexmark(p1);
      tlist[vertexindex++] = vertexmark(p2);
      tlist[vertexindex++] = vertexmark(p3);
#else
           fprintf(outfile, "%4ld    %4d  %4d  %4d", elementnumber,
              vertexmark(p1), vertexmark(p2), vertexmark(p3));
#endif
    } else {
      mid1 = (vertex) triangleloop.tri[m->highorderindex + 1];
      mid2 = (vertex) triangleloop.tri[m->highorderindex + 2];
      mid3 = (vertex) triangleloop.tri[m->highorderindex];
#ifdef TRILIBRARY
      tlist[vertexindex++] = vertexmark(p1);
      tlist[vertexindex++] = vertexmark(p2);
      tlist[vertexindex++] = vertexmark(p3);
      tlist[vertexindex++] = vertexmark(mid1);
      tlist[vertexindex++] = vertexmark(mid2);
      tlist[vertexindex++] = vertexmark(mid3);
#else
           fprintf(outfile, "%4ld    %4d  %4d  %4d  %4d  %4d  %4d", elementnumber,
              vertexmark(p1), vertexmark(p2), vertexmark(p3), vertexmark(mid1),
              vertexmark(mid2), vertexmark(mid3));
#endif
    }

#ifdef TRILIBRARY
    for (i = 0; i < m->eextras; i++) {
      talist[attribindex++] = elemattribute(triangleloop, i);
    }
#else
    for (i = 0; i < m->eextras; i++) {
      fprintf(outfile, "  %.17g", elemattribute(triangleloop, i));
    }
    fprintf(outfile, "\n");
#endif

    triangleloop.tri = triangletraverse(m);
    elementnumber++;
  }

#ifndef TRILIBRARY
  finishfile(outfile, argc, argv);
#endif
}


#ifdef TRILIBRARY

#ifdef ANSI_DECLARATORS
void writepoly(struct mesh *m, struct behavior *b,
               int **segmentlist, int **segmentmarkerlist)
#else
void writepoly(m, b, segmentlist, segmentmarkerlist)
struct mesh *m;
struct behavior *b;
int **segmentlist;
int **segmentmarkerlist;
#endif

#else

#ifdef ANSI_DECLARATORS
void writepoly(struct mesh *m, struct behavior *b, char *polyfilename,
               REAL *holelist, int holes, REAL *regionlist, int regions,
               int argc, char **argv)
#else
void writepoly(m, b, polyfilename, holelist, holes, regionlist, regions,
               argc, argv)
struct mesh *m;
struct behavior *b;
char *polyfilename;
REAL *holelist;
int holes;
REAL *regionlist;
int regions;
int argc;
char **argv;
#endif

#endif

{
#ifdef TRILIBRARY
  int *slist;
  int *smlist;
  int index;
#else
  FILE *outfile;
  long holenumber, regionnumber;
#endif
  struct osub subsegloop;
  vertex endpoint1, endpoint2;
  long subsegnumber;

#ifdef TRILIBRARY
  if (!b->quiet) {
    printf("Writing segments.\n");
  }
   if (*segmentlist == (int *) NULL) {
    *segmentlist = (int *) trimalloc((int) (m->subsegs.items * 2 *
                                            sizeof(int)));
  }
   if (!b->nobound && (*segmentmarkerlist == (int *) NULL)) {
    *segmentmarkerlist = (int *) trimalloc((int) (m->subsegs.items *
                                                  sizeof(int)));
  }
  slist = *segmentlist;
  smlist = *segmentmarkerlist;
  index = 0;
#else
  if (!b->quiet) {
    printf("Writing %s.\n", polyfilename);
  }
  outfile = fopen(polyfilename, "w");
  if (outfile == (FILE *) NULL) {
    printf("  Error:  Cannot create file %s.\n", polyfilename);
    triexit(1);
  }
     fprintf(outfile, "%d  %d  %d  %d\n", 0, m->mesh_dim, m->nextras,
          1 - b->nobound);
   fprintf(outfile, "%ld  %d\n", m->subsegs.items, 1 - b->nobound);
#endif

  traversalinit(&m->subsegs);
  subsegloop.ss = subsegtraverse(m);
  subsegloop.ssorient = 0;
  subsegnumber = b->firstnumber;
  while (subsegloop.ss != (subseg *) NULL) {
    sorg(subsegloop, endpoint1);
    sdest(subsegloop, endpoint2);
#ifdef TRILIBRARY
       slist[index++] = vertexmark(endpoint1);
    slist[index++] = vertexmark(endpoint2);
    if (!b->nobound) {
           smlist[subsegnumber - b->firstnumber] = mark(subsegloop);
    }
#else
       if (b->nobound) {
      fprintf(outfile, "%4ld    %4d  %4d\n", subsegnumber,
              vertexmark(endpoint1), vertexmark(endpoint2));
    } else {
      fprintf(outfile, "%4ld    %4d  %4d    %4d\n", subsegnumber,
              vertexmark(endpoint1), vertexmark(endpoint2), mark(subsegloop));
    }
#endif

    subsegloop.ss = subsegtraverse(m);
    subsegnumber++;
  }

#ifndef TRILIBRARY
#ifndef CDT_ONLY
  fprintf(outfile, "%d\n", holes);
  if (holes > 0) {
    for (holenumber = 0; holenumber < holes; holenumber++) {
           fprintf(outfile, "%4ld   %.17g  %.17g\n", b->firstnumber + holenumber,
              holelist[2 * holenumber], holelist[2 * holenumber + 1]);
    }
  }
  if (regions > 0) {
    fprintf(outfile, "%d\n", regions);
    for (regionnumber = 0; regionnumber < regions; regionnumber++) {
           fprintf(outfile, "%4ld   %.17g  %.17g  %.17g  %.17g\n",
              b->firstnumber + regionnumber,
              regionlist[4 * regionnumber], regionlist[4 * regionnumber + 1],
              regionlist[4 * regionnumber + 2],
              regionlist[4 * regionnumber + 3]);
    }
  }
#endif

  finishfile(outfile, argc, argv);
#endif
}


#ifdef TRILIBRARY

#ifdef ANSI_DECLARATORS
void writeedges(struct mesh *m, struct behavior *b,
                int **edgelist, int **edgemarkerlist)
#else
void writeedges(m, b, edgelist, edgemarkerlist)
struct mesh *m;
struct behavior *b;
int **edgelist;
int **edgemarkerlist;
#endif

#else

#ifdef ANSI_DECLARATORS
void writeedges(struct mesh *m, struct behavior *b, char *edgefilename,
                int argc, char **argv)
#else
void writeedges(m, b, edgefilename, argc, argv)
struct mesh *m;
struct behavior *b;
char *edgefilename;
int argc;
char **argv;
#endif

#endif

{
#ifdef TRILIBRARY
  int *elist;
  int *emlist;
  int index;
#else
  FILE *outfile;
#endif
  struct otri triangleloop, trisym;
  struct osub checkmark;
  vertex p1, p2;
  long edgenumber;
  triangle ptr;
  subseg sptr;

#ifdef TRILIBRARY
  if (!b->quiet) {
    printf("Writing edges.\n");
  }
   if (*edgelist == (int *) NULL) {
    *edgelist = (int *) trimalloc((int) (m->edges * 2 * sizeof(int)));
  }
   if (!b->nobound && (*edgemarkerlist == (int *) NULL)) {
    *edgemarkerlist = (int *) trimalloc((int) (m->edges * sizeof(int)));
  }
  elist = *edgelist;
  emlist = *edgemarkerlist;
  index = 0;
#else
  if (!b->quiet) {
    printf("Writing %s.\n", edgefilename);
  }
  outfile = fopen(edgefilename, "w");
  if (outfile == (FILE *) NULL) {
    printf("  Error:  Cannot create file %s.\n", edgefilename);
    triexit(1);
  }
   fprintf(outfile, "%ld  %d\n", m->edges, 1 - b->nobound);
#endif

  traversalinit(&m->triangles);
  triangleloop.tri = triangletraverse(m);
  edgenumber = b->firstnumber;
        while (triangleloop.tri != (triangle *) NULL) {
    for (triangleloop.orient = 0; triangleloop.orient < 3;
         triangleloop.orient++) {
      sym(triangleloop, trisym);
      if ((triangleloop.tri < trisym.tri) || (trisym.tri == m->dummytri)) {
        org(triangleloop, p1);
        dest(triangleloop, p2);
#ifdef TRILIBRARY
        elist[index++] = vertexmark(p1);
        elist[index++] = vertexmark(p2);
#endif
        if (b->nobound) {
#ifndef TRILIBRARY
                   fprintf(outfile, "%4ld   %d  %d\n", edgenumber,
                  vertexmark(p1), vertexmark(p2));
#endif
        } else {
                            if (b->usesegments) {
            tspivot(triangleloop, checkmark);
            if (checkmark.ss == m->dummysub) {
#ifdef TRILIBRARY
              emlist[edgenumber - b->firstnumber] = 0;
#else
              fprintf(outfile, "%4ld   %d  %d  %d\n", edgenumber,
                      vertexmark(p1), vertexmark(p2), 0);
#endif
            } else {
#ifdef TRILIBRARY
              emlist[edgenumber - b->firstnumber] = mark(checkmark);
#else
              fprintf(outfile, "%4ld   %d  %d  %d\n", edgenumber,
                      vertexmark(p1), vertexmark(p2), mark(checkmark));
#endif
            }
          } else {
#ifdef TRILIBRARY
            emlist[edgenumber - b->firstnumber] = trisym.tri == m->dummytri;
#else
            fprintf(outfile, "%4ld   %d  %d  %d\n", edgenumber,
                    vertexmark(p1), vertexmark(p2), trisym.tri == m->dummytri);
#endif
          }
        }
        edgenumber++;
      }
    }
    triangleloop.tri = triangletraverse(m);
  }

#ifndef TRILIBRARY
  finishfile(outfile, argc, argv);
#endif
}


#ifdef TRILIBRARY

#ifdef ANSI_DECLARATORS
void writevoronoi(struct mesh *m, struct behavior *b, REAL **vpointlist,
                  REAL **vpointattriblist, int **vpointmarkerlist,
                  int **vedgelist, int **vedgemarkerlist, REAL **vnormlist)
#else
void writevoronoi(m, b, vpointlist, vpointattriblist, vpointmarkerlist,
                  vedgelist, vedgemarkerlist, vnormlist)
struct mesh *m;
struct behavior *b;
REAL **vpointlist;
REAL **vpointattriblist;
int **vpointmarkerlist;
int **vedgelist;
int **vedgemarkerlist;
REAL **vnormlist;
#endif

#else

#ifdef ANSI_DECLARATORS
void writevoronoi(struct mesh *m, struct behavior *b, char *vnodefilename,
                  char *vedgefilename, int argc, char **argv)
#else
void writevoronoi(m, b, vnodefilename, vedgefilename, argc, argv)
struct mesh *m;
struct behavior *b;
char *vnodefilename;
char *vedgefilename;
int argc;
char **argv;
#endif

#endif

{
#ifdef TRILIBRARY
  REAL *plist;
  REAL *palist;
  int *elist;
  REAL *normlist;
  int coordindex;
  int attribindex;
#else
  FILE *outfile;
#endif
  struct otri triangleloop, trisym;
  vertex torg, tdest, tapex;
  REAL circumcenter[2];
  REAL xi, eta;
  long vnodenumber, vedgenumber;
  int p1, p2;
  int i;
  triangle ptr;

#ifdef TRILIBRARY
  if (!b->quiet) {
    printf("Writing Voronoi vertices.\n");
  }
   if (*vpointlist == (REAL *) NULL) {
    *vpointlist = (REAL *) trimalloc((int) (m->triangles.items * 2 *
                                            sizeof(REAL)));
  }
   if (*vpointattriblist == (REAL *) NULL) {
    *vpointattriblist = (REAL *) trimalloc((int) (m->triangles.items *
                                                  m->nextras * sizeof(REAL)));
  }
  *vpointmarkerlist = (int *) NULL;
  plist = *vpointlist;
  palist = *vpointattriblist;
  coordindex = 0;
  attribindex = 0;
#else
  if (!b->quiet) {
    printf("Writing %s.\n", vnodefilename);
  }
  outfile = fopen(vnodefilename, "w");
  if (outfile == (FILE *) NULL) {
    printf("  Error:  Cannot create file %s.\n", vnodefilename);
    triexit(1);
  }
    fprintf(outfile, "%ld  %d  %d  %d\n", m->triangles.items, 2, m->nextras, 0);
#endif

  traversalinit(&m->triangles);
  triangleloop.tri = triangletraverse(m);
  triangleloop.orient = 0;
  vnodenumber = b->firstnumber;
  while (triangleloop.tri != (triangle *) NULL) {
    org(triangleloop, torg);
    dest(triangleloop, tdest);
    apex(triangleloop, tapex);
    findcircumcenter(m, b, torg, tdest, tapex, circumcenter, &xi, &eta, 0);
#ifdef TRILIBRARY
       plist[coordindex++] = circumcenter[0];
    plist[coordindex++] = circumcenter[1];
    for (i = 2; i < 2 + m->nextras; i++) {
           palist[attribindex++] = torg[i] + xi * (tdest[i] - torg[i])
                                     + eta * (tapex[i] - torg[i]);
    }
#else
       fprintf(outfile, "%4ld    %.17g  %.17g", vnodenumber, circumcenter[0],
            circumcenter[1]);
    for (i = 2; i < 2 + m->nextras; i++) {
           fprintf(outfile, "  %.17g", torg[i] + xi * (tdest[i] - torg[i])
                                         + eta * (tapex[i] - torg[i]));
    }
    fprintf(outfile, "\n");
#endif

    * (int *) (triangleloop.tri + 6) = (int) vnodenumber;
    triangleloop.tri = triangletraverse(m);
    vnodenumber++;
  }

#ifndef TRILIBRARY
  finishfile(outfile, argc, argv);
#endif

#ifdef TRILIBRARY
  if (!b->quiet) {
    printf("Writing Voronoi edges.\n");
  }
   if (*vedgelist == (int *) NULL) {
    *vedgelist = (int *) trimalloc((int) (m->edges * 2 * sizeof(int)));
  }
  *vedgemarkerlist = (int *) NULL;
   if (*vnormlist == (REAL *) NULL) {
    *vnormlist = (REAL *) trimalloc((int) (m->edges * 2 * sizeof(REAL)));
  }
  elist = *vedgelist;
  normlist = *vnormlist;
  coordindex = 0;
#else
  if (!b->quiet) {
    printf("Writing %s.\n", vedgefilename);
  }
  outfile = fopen(vedgefilename, "w");
  if (outfile == (FILE *) NULL) {
    printf("  Error:  Cannot create file %s.\n", vedgefilename);
    triexit(1);
  }
   fprintf(outfile, "%ld  %d\n", m->edges, 0);
#endif

  traversalinit(&m->triangles);
  triangleloop.tri = triangletraverse(m);
  vedgenumber = b->firstnumber;
        while (triangleloop.tri != (triangle *) NULL) {
    for (triangleloop.orient = 0; triangleloop.orient < 3;
         triangleloop.orient++) {
      sym(triangleloop, trisym);
      if ((triangleloop.tri < trisym.tri) || (trisym.tri == m->dummytri)) {
               p1 = * (int *) (triangleloop.tri + 6);
        if (trisym.tri == m->dummytri) {
          org(triangleloop, torg);
          dest(triangleloop, tdest);
#ifdef TRILIBRARY
                   elist[coordindex] = p1;
          normlist[coordindex++] = tdest[1] - torg[1];
          elist[coordindex] = -1;
          normlist[coordindex++] = torg[0] - tdest[0];
#else
                                     fprintf(outfile, "%4ld   %d  %d   %.17g  %.17g\n", vedgenumber,
                  p1, -1, tdest[1] - torg[1], torg[0] - tdest[0]);
#endif
        } else {
                   p2 = * (int *) (trisym.tri + 6);
         #ifdef TRILIBRARY
          elist[coordindex] = p1;
          normlist[coordindex++] = 0.0;
          elist[coordindex] = p2;
          normlist[coordindex++] = 0.0;
#else
          fprintf(outfile, "%4ld   %d  %d\n", vedgenumber, p1, p2);
#endif
        }
        vedgenumber++;
      }
    }
    triangleloop.tri = triangletraverse(m);
  }

#ifndef TRILIBRARY
  finishfile(outfile, argc, argv);
#endif
}

#ifdef TRILIBRARY

#ifdef ANSI_DECLARATORS
void writeneighbors(struct mesh *m, struct behavior *b, int **neighborlist)
#else
void writeneighbors(m, b, neighborlist)
struct mesh *m;
struct behavior *b;
int **neighborlist;
#endif

#else

#ifdef ANSI_DECLARATORS
void writeneighbors(struct mesh *m, struct behavior *b, char *neighborfilename,
                    int argc, char **argv)
#else
void writeneighbors(m, b, neighborfilename, argc, argv)
struct mesh *m;
struct behavior *b;
char *neighborfilename;
int argc;
char **argv;
#endif

#endif

{
#ifdef TRILIBRARY
  int *nlist;
  int index;
#else
  FILE *outfile;
#endif
  struct otri triangleloop, trisym;
  long elementnumber;
  int neighbor1, neighbor2, neighbor3;
  triangle ptr;

#ifdef TRILIBRARY
  if (!b->quiet) {
    printf("Writing neighbors.\n");
  }
   if (*neighborlist == (int *) NULL) {
    *neighborlist = (int *) trimalloc((int) (m->triangles.items * 3 *
                                             sizeof(int)));
  }
  nlist = *neighborlist;
  index = 0;
#else
  if (!b->quiet) {
    printf("Writing %s.\n", neighborfilename);
  }
  outfile = fopen(neighborfilename, "w");
  if (outfile == (FILE *) NULL) {
    printf("  Error:  Cannot create file %s.\n", neighborfilename);
    triexit(1);
  }
   fprintf(outfile, "%ld  %d\n", m->triangles.items, 3);
#endif

  traversalinit(&m->triangles);
  triangleloop.tri = triangletraverse(m);
  triangleloop.orient = 0;
  elementnumber = b->firstnumber;
  while (triangleloop.tri != (triangle *) NULL) {
    * (int *) (triangleloop.tri + 6) = (int) elementnumber;
    triangleloop.tri = triangletraverse(m);
    elementnumber++;
  }
  * (int *) (m->dummytri + 6) = -1;

  traversalinit(&m->triangles);
  triangleloop.tri = triangletraverse(m);
  elementnumber = b->firstnumber;
  while (triangleloop.tri != (triangle *) NULL) {
    triangleloop.orient = 1;
    sym(triangleloop, trisym);
    neighbor1 = * (int *) (trisym.tri + 6);
    triangleloop.orient = 2;
    sym(triangleloop, trisym);
    neighbor2 = * (int *) (trisym.tri + 6);
    triangleloop.orient = 0;
    sym(triangleloop, trisym);
    neighbor3 = * (int *) (trisym.tri + 6);
#ifdef TRILIBRARY
    nlist[index++] = neighbor1;
    nlist[index++] = neighbor2;
    nlist[index++] = neighbor3;
#else
       fprintf(outfile, "%4ld    %d  %d  %d\n", elementnumber,
            neighbor1, neighbor2, neighbor3);
#endif

    triangleloop.tri = triangletraverse(m);
    elementnumber++;
  }

#ifndef TRILIBRARY
  finishfile(outfile, argc, argv);
#endif
}


#ifndef TRILIBRARY

#ifdef ANSI_DECLARATORS
void writeoff(struct mesh *m, struct behavior *b, char *offfilename,
              int argc, char **argv)
#else
void writeoff(m, b, offfilename, argc, argv)
struct mesh *m;
struct behavior *b;
char *offfilename;
int argc;
char **argv;
#endif

{
  FILE *outfile;
  struct otri triangleloop;
  vertex vertexloop;
  vertex p1, p2, p3;
  long outvertices;

  if (!b->quiet) {
    printf("Writing %s.\n", offfilename);
  }

  if (b->jettison) {
    outvertices = m->vertices.items - m->undeads;
  } else {
    outvertices = m->vertices.items;
  }

  outfile = fopen(offfilename, "w");
  if (outfile == (FILE *) NULL) {
    printf("  Error:  Cannot create file %s.\n", offfilename);
    triexit(1);
  }
   fprintf(outfile, "OFF\n%ld  %ld  %ld\n", outvertices, m->triangles.items,
          m->edges);

   traversalinit(&m->vertices);
  vertexloop = vertextraverse(m);
  while (vertexloop != (vertex) NULL) {
    if (!b->jettison || (vertextype(vertexloop) != UNDEADVERTEX)) {
           fprintf(outfile, " %.17g  %.17g  %.17g\n", vertexloop[0], vertexloop[1],
              0.0);
    }
    vertexloop = vertextraverse(m);
  }

   traversalinit(&m->triangles);
  triangleloop.tri = triangletraverse(m);
  triangleloop.orient = 0;
  while (triangleloop.tri != (triangle *) NULL) {
    org(triangleloop, p1);
    dest(triangleloop, p2);
    apex(triangleloop, p3);
       fprintf(outfile, " 3   %4d  %4d  %4d\n", vertexmark(p1) - b->firstnumber,
            vertexmark(p2) - b->firstnumber, vertexmark(p3) - b->firstnumber);
    triangleloop.tri = triangletraverse(m);
  }
  finishfile(outfile, argc, argv);
}

#endif



#ifdef ANSI_DECLARATORS
void quality_statistics(struct mesh *m, struct behavior *b)
#else
void quality_statistics(m, b)
struct mesh *m;
struct behavior *b;
#endif

{
  struct otri triangleloop;
  vertex p[3];
  REAL cossquaretable[8];
  REAL ratiotable[16];
  REAL dx[3], dy[3];
  REAL edgelength[3];
  REAL dotproduct;
  REAL cossquare;
  REAL triarea;
  REAL shortest, longest;
  REAL trilongest2;
  REAL smallestarea, biggestarea;
  REAL triminaltitude2;
  REAL minaltitude;
  REAL triaspect2;
  REAL worstaspect;
  REAL smallestangle, biggestangle;
  REAL radconst, degconst;
  int angletable[18];
  int aspecttable[16];
  int aspectindex;
  int tendegree;
  int acutebiggest;
  int i, ii, j, k;

  printf("Mesh quality statistics:\n\n");
  radconst = PI / 18.0;
  degconst = 180.0 / PI;
  for (i = 0; i < 8; i++) {
    cossquaretable[i] = cos(radconst * (REAL) (i + 1));
    cossquaretable[i] = cossquaretable[i] * cossquaretable[i];
  }
  for (i = 0; i < 18; i++) {
    angletable[i] = 0;
  }

  ratiotable[0]  =      1.5;      ratiotable[1]  =     2.0;
  ratiotable[2]  =      2.5;      ratiotable[3]  =     3.0;
  ratiotable[4]  =      4.0;      ratiotable[5]  =     6.0;
  ratiotable[6]  =     10.0;      ratiotable[7]  =    15.0;
  ratiotable[8]  =     25.0;      ratiotable[9]  =    50.0;
  ratiotable[10] =    100.0;      ratiotable[11] =   300.0;
  ratiotable[12] =   1000.0;      ratiotable[13] = 10000.0;
  ratiotable[14] = 100000.0;      ratiotable[15] =     0.0;
  for (i = 0; i < 16; i++) {
    aspecttable[i] = 0;
  }

  worstaspect = 0.0;
  minaltitude = m->xmax - m->xmin + m->ymax - m->ymin;
  minaltitude = minaltitude * minaltitude;
  shortest = minaltitude;
  longest = 0.0;
  smallestarea = minaltitude;
  biggestarea = 0.0;
  worstaspect = 0.0;
  smallestangle = 0.0;
  biggestangle = 2.0;
  acutebiggest = 1;

  traversalinit(&m->triangles);
  triangleloop.tri = triangletraverse(m);
  triangleloop.orient = 0;
  while (triangleloop.tri != (triangle *) NULL) {
    org(triangleloop, p[0]);
    dest(triangleloop, p[1]);
    apex(triangleloop, p[2]);
    trilongest2 = 0.0;

    for (i = 0; i < 3; i++) {
      j = plus1mod3[i];
      k = minus1mod3[i];
      dx[i] = p[j][0] - p[k][0];
      dy[i] = p[j][1] - p[k][1];
      edgelength[i] = dx[i] * dx[i] + dy[i] * dy[i];
      if (edgelength[i] > trilongest2) {
        trilongest2 = edgelength[i];
      }
      if (edgelength[i] > longest) {
        longest = edgelength[i];
      }
      if (edgelength[i] < shortest) {
        shortest = edgelength[i];
      }
    }

    triarea = counterclockwise(m, b, p[0], p[1], p[2]);
    if (triarea < smallestarea) {
      smallestarea = triarea;
    }
    if (triarea > biggestarea) {
      biggestarea = triarea;
    }
    triminaltitude2 = triarea * triarea / trilongest2;
    if (triminaltitude2 < minaltitude) {
      minaltitude = triminaltitude2;
    }
    triaspect2 = trilongest2 / triminaltitude2;
    if (triaspect2 > worstaspect) {
      worstaspect = triaspect2;
    }
    aspectindex = 0;
    while ((triaspect2 > ratiotable[aspectindex] * ratiotable[aspectindex])
           && (aspectindex < 15)) {
      aspectindex++;
    }
    aspecttable[aspectindex]++;

    for (i = 0; i < 3; i++) {
      j = plus1mod3[i];
      k = minus1mod3[i];
      dotproduct = dx[j] * dx[k] + dy[j] * dy[k];
      cossquare = dotproduct * dotproduct / (edgelength[j] * edgelength[k]);
      tendegree = 8;
      for (ii = 7; ii >= 0; ii--) {
        if (cossquare > cossquaretable[ii]) {
          tendegree = ii;
        }
      }
      if (dotproduct <= 0.0) {
        angletable[tendegree]++;
        if (cossquare > smallestangle) {
          smallestangle = cossquare;
        }
        if (acutebiggest && (cossquare < biggestangle)) {
          biggestangle = cossquare;
        }
      } else {
        angletable[17 - tendegree]++;
        if (acutebiggest || (cossquare > biggestangle)) {
          biggestangle = cossquare;
          acutebiggest = 0;
        }
      }
    }
    triangleloop.tri = triangletraverse(m);
  }

  shortest = sqrt(shortest);
  longest = sqrt(longest);
  minaltitude = sqrt(minaltitude);
  worstaspect = sqrt(worstaspect);
  smallestarea *= 0.5;
  biggestarea *= 0.5;
  if (smallestangle >= 1.0) {
    smallestangle = 0.0;
  } else {
    smallestangle = degconst * acos(sqrt(smallestangle));
  }
  if (biggestangle >= 1.0) {
    biggestangle = 180.0;
  } else {
    if (acutebiggest) {
      biggestangle = degconst * acos(sqrt(biggestangle));
    } else {
      biggestangle = 180.0 - degconst * acos(sqrt(biggestangle));
    }
  }

  printf("  Smallest area: %16.5g   |  Largest area: %16.5g\n",
         smallestarea, biggestarea);
  printf("  Shortest edge: %16.5g   |  Longest edge: %16.5g\n",
         shortest, longest);
  printf("  Shortest altitude: %12.5g   |  Largest aspect ratio: %8.5g\n\n",
         minaltitude, worstaspect);

  printf("  Triangle aspect ratio histogram:\n");
  printf("  1.1547 - %-6.6g    :  %8d    | %6.6g - %-6.6g     :  %8d\n",
         ratiotable[0], aspecttable[0], ratiotable[7], ratiotable[8],
         aspecttable[8]);
  for (i = 1; i < 7; i++) {
    printf("  %6.6g - %-6.6g    :  %8d    | %6.6g - %-6.6g     :  %8d\n",
           ratiotable[i - 1], ratiotable[i], aspecttable[i],
           ratiotable[i + 7], ratiotable[i + 8], aspecttable[i + 8]);
  }
  printf("  %6.6g - %-6.6g    :  %8d    | %6.6g -            :  %8d\n",
         ratiotable[6], ratiotable[7], aspecttable[7], ratiotable[14],
         aspecttable[15]);
  printf("  (Aspect ratio is longest edge divided by shortest altitude)\n\n");

  printf("  Smallest angle: %15.5g   |  Largest angle: %15.5g\n\n",
         smallestangle, biggestangle);

  printf("  Angle histogram:\n");
  for (i = 0; i < 9; i++) {
    printf("    %3d - %3d degrees:  %8d    |    %3d - %3d degrees:  %8d\n",
           i * 10, i * 10 + 10, angletable[i],
           i * 10 + 90, i * 10 + 100, angletable[i + 9]);
  }
  printf("\n");
}


#ifdef ANSI_DECLARATORS
void statistics(struct mesh *m, struct behavior *b)
#else
void statistics(m, b)
struct mesh *m;
struct behavior *b;
#endif

{
  printf("\nStatistics:\n\n");
  printf("  Input vertices: %d\n", m->invertices);
  if (b->refine) {
    printf("  Input triangles: %d\n", m->inelements);
  }
  if (b->poly) {
    printf("  Input segments: %d\n", m->insegments);
    if (!b->refine) {
      printf("  Input holes: %d\n", m->holes);
    }
  }

  printf("\n  Mesh vertices: %ld\n", m->vertices.items - m->undeads);
  printf("  Mesh triangles: %ld\n", m->triangles.items);
  printf("  Mesh edges: %ld\n", m->edges);
  printf("  Mesh exterior boundary edges: %ld\n", m->hullsize);
  if (b->poly || b->refine) {
    printf("  Mesh interior boundary edges: %ld\n",
           m->subsegs.items - m->hullsize);
    printf("  Mesh subsegments (constrained edges): %ld\n",
           m->subsegs.items);
  }
  printf("\n");

  if (b->verbose) {
    quality_statistics(m, b);
    printf("Memory allocation statistics:\n\n");
    printf("  Maximum number of vertices: %ld\n", m->vertices.maxitems);
    printf("  Maximum number of triangles: %ld\n", m->triangles.maxitems);
    if (m->subsegs.maxitems > 0) {
      printf("  Maximum number of subsegments: %ld\n", m->subsegs.maxitems);
    }
    if (m->viri.maxitems > 0) {
      printf("  Maximum number of viri: %ld\n", m->viri.maxitems);
    }
    if (m->badsubsegs.maxitems > 0) {
      printf("  Maximum number of encroached subsegments: %ld\n",
             m->badsubsegs.maxitems);
    }
    if (m->badtriangles.maxitems > 0) {
      printf("  Maximum number of bad triangles: %ld\n",
             m->badtriangles.maxitems);
    }
    if (m->flipstackers.maxitems > 0) {
      printf("  Maximum number of stacked triangle flips: %ld\n",
             m->flipstackers.maxitems);
    }
    if (m->splaynodes.maxitems > 0) {
      printf("  Maximum number of splay tree nodes: %ld\n",
             m->splaynodes.maxitems);
    }
    printf("  Approximate heap memory use (bytes): %ld\n\n",
           m->vertices.maxitems * m->vertices.itembytes +
           m->triangles.maxitems * m->triangles.itembytes +
           m->subsegs.maxitems * m->subsegs.itembytes +
           m->viri.maxitems * m->viri.itembytes +
           m->badsubsegs.maxitems * m->badsubsegs.itembytes +
           m->badtriangles.maxitems * m->badtriangles.itembytes +
           m->flipstackers.maxitems * m->flipstackers.itembytes +
           m->splaynodes.maxitems * m->splaynodes.itembytes);

    printf("Algorithmic statistics:\n\n");
    if (!b->weighted) {
      printf("  Number of incircle tests: %ld\n", m->incirclecount);
    } else {
      printf("  Number of 3D orientation tests: %ld\n", m->orient3dcount);
    }
    printf("  Number of 2D orientation tests: %ld\n", m->counterclockcount);
    if (m->hyperbolacount > 0) {
      printf("  Number of right-of-hyperbola tests: %ld\n",
             m->hyperbolacount);
    }
    if (m->circletopcount > 0) {
      printf("  Number of circle top computations: %ld\n",
             m->circletopcount);
    }
    if (m->circumcentercount > 0) {
      printf("  Number of triangle circumcenter computations: %ld\n",
             m->circumcentercount);
    }
    printf("\n");
  }
}


#ifdef TRILIBRARY

#ifdef ANSI_DECLARATORS
void triangulate(char *triswitches, struct triangulateio *in,
                 struct triangulateio *out, struct triangulateio *vorout)
#else
void triangulate(triswitches, in, out, vorout)
char *triswitches;
struct triangulateio *in;
struct triangulateio *out;
struct triangulateio *vorout;
#endif

#else

#ifdef ANSI_DECLARATORS
int main(int argc, char **argv)
#else
int main(argc, argv)
int argc;
char **argv;
#endif

#endif

{
  struct mesh m;
  struct behavior b;
  REAL *holearray;
  REAL *regionarray;
#ifndef TRILIBRARY
  FILE *polyfile;
#endif
#ifndef NO_TIMER
    struct timeval tv0, tv1, tv2, tv3, tv4, tv5, tv6;
  struct timezone tz;
#endif

#ifndef NO_TIMER
  gettimeofday(&tv0, &tz);
#endif

  triangleinit(&m);
#ifdef TRILIBRARY
  parsecommandline(1, &triswitches, &b);
#else
  parsecommandline(argc, argv, &b);
#endif
  m.steinerleft = b.steiner;

#ifdef TRILIBRARY
  transfernodes(&m, &b, in->pointlist, in->pointattributelist,
                in->pointmarkerlist, in->numberofpoints,
                in->numberofpointattributes);
#else
  readnodes(&m, &b, b.innodefilename, b.inpolyfilename, &polyfile);
#endif

#ifndef NO_TIMER
  if (!b.quiet) {
    gettimeofday(&tv1, &tz);
  }
#endif

#ifdef CDT_ONLY
  m.hullsize = delaunay(&m, &b);
#else
  if (b.refine) {
   #ifdef TRILIBRARY
    m.hullsize = reconstruct(&m, &b, in->trianglelist,
                             in->triangleattributelist, in->trianglearealist,
                             in->numberoftriangles, in->numberofcorners,
                             in->numberoftriangleattributes,
                             in->segmentlist, in->segmentmarkerlist,
                             in->numberofsegments);
#else
    m.hullsize = reconstruct(&m, &b, b.inelefilename, b.areafilename,
                             b.inpolyfilename, polyfile);
#endif
  } else {
    m.hullsize = delaunay(&m, &b);
  }
#endif

#ifndef NO_TIMER
  if (!b.quiet) {
    gettimeofday(&tv2, &tz);
    if (b.refine) {
      printf("Mesh reconstruction");
    } else {
      printf("Delaunay");
    }
    printf(" milliseconds:  %ld\n", 1000l * (tv2.tv_sec - tv1.tv_sec) +
           (tv2.tv_usec - tv1.tv_usec) / 1000l);
  }
#endif

    m.infvertex1 = (vertex) NULL;
  m.infvertex2 = (vertex) NULL;
  m.infvertex3 = (vertex) NULL;

  if (b.usesegments) {
    m.checksegments = 1;
    if (!b.refine) {
     #ifdef TRILIBRARY
      formskeleton(&m, &b, in->segmentlist,
                   in->segmentmarkerlist, in->numberofsegments);
#else
      formskeleton(&m, &b, polyfile, b.inpolyfilename);
#endif
    }
  }

#ifndef NO_TIMER
  if (!b.quiet) {
    gettimeofday(&tv3, &tz);
    if (b.usesegments && !b.refine) {
      printf("Segment milliseconds:  %ld\n",
             1000l * (tv3.tv_sec - tv2.tv_sec) +
             (tv3.tv_usec - tv2.tv_usec) / 1000l);
    }
  }
#endif

  if (b.poly && (m.triangles.items > 0)) {
#ifdef TRILIBRARY
    holearray = in->holelist;
    m.holes = in->numberofholes;
    regionarray = in->regionlist;
    m.regions = in->numberofregions;
#else
    readholes(&m, &b, polyfile, b.inpolyfilename, &holearray, &m.holes,
              &regionarray, &m.regions);
#endif
    if (!b.refine) {
           carveholes(&m, &b, holearray, m.holes, regionarray, m.regions);
    }
  } else {
             m.holes = 0;
    m.regions = 0;
  }

#ifndef NO_TIMER
  if (!b.quiet) {
    gettimeofday(&tv4, &tz);
    if (b.poly && !b.refine) {
      printf("Hole milliseconds:  %ld\n", 1000l * (tv4.tv_sec - tv3.tv_sec) +
             (tv4.tv_usec - tv3.tv_usec) / 1000l);
    }
  }
#endif

#ifndef CDT_ONLY
  if (b.quality && (m.triangles.items > 0)) {
    enforcequality(&m, &b);
  }
#endif

#ifndef NO_TIMER
  if (!b.quiet) {
    gettimeofday(&tv5, &tz);
#ifndef CDT_ONLY
    if (b.quality) {
      printf("Quality milliseconds:  %ld\n",
             1000l * (tv5.tv_sec - tv4.tv_sec) +
             (tv5.tv_usec - tv4.tv_usec) / 1000l);
    }
#endif
  }
#endif

   m.edges = (3l * m.triangles.items + m.hullsize) / 2l;

  if (b.order > 1) {
    highorder(&m, &b);
  }
  if (!b.quiet) {
    printf("\n");
  }

#ifdef TRILIBRARY
  if (b.jettison) {
    out->numberofpoints = m.vertices.items - m.undeads;
  } else {
    out->numberofpoints = m.vertices.items;
  }
  out->numberofpointattributes = m.nextras;
  out->numberoftriangles = m.triangles.items;
  out->numberofcorners = (b.order + 1) * (b.order + 2) / 2;
  out->numberoftriangleattributes = m.eextras;
  out->numberofedges = m.edges;
  if (b.usesegments) {
    out->numberofsegments = m.subsegs.items;
  } else {
    out->numberofsegments = m.hullsize;
  }
  if (vorout != (struct triangulateio *) NULL) {
    vorout->numberofpoints = m.triangles.items;
    vorout->numberofpointattributes = m.nextras;
    vorout->numberofedges = m.edges;
  }
#endif
    if (b.nonodewritten || (b.noiterationnum && m.readnodefile)) {
    if (!b.quiet) {
#ifdef TRILIBRARY
      printf("NOT writing vertices.\n");
#else
      printf("NOT writing a .node file.\n");
#endif
    }
    numbernodes(&m, &b);
  } else {
   #ifdef TRILIBRARY
    writenodes(&m, &b, &out->pointlist, &out->pointattributelist,
               &out->pointmarkerlist);
#else
    writenodes(&m, &b, b.outnodefilename, argc, argv);
#endif
  }
  if (b.noelewritten) {
    if (!b.quiet) {
#ifdef TRILIBRARY
      printf("NOT writing triangles.\n");
#else
      printf("NOT writing an .ele file.\n");
#endif
    }
  } else {
#ifdef TRILIBRARY
    writeelements(&m, &b, &out->trianglelist, &out->triangleattributelist);
#else
    writeelements(&m, &b, b.outelefilename, argc, argv);
#endif
  }
    if (b.poly || b.convex) {
       if (b.nopolywritten || b.noiterationnum) {
      if (!b.quiet) {
#ifdef TRILIBRARY
        printf("NOT writing segments.\n");
#else
        printf("NOT writing a .poly file.\n");
#endif
      }
    } else {
#ifdef TRILIBRARY
      writepoly(&m, &b, &out->segmentlist, &out->segmentmarkerlist);
      out->numberofholes = m.holes;
      out->numberofregions = m.regions;
      if (b.poly) {
        out->holelist = in->holelist;
        out->regionlist = in->regionlist;
      } else {
        out->holelist = (REAL *) NULL;
        out->regionlist = (REAL *) NULL;
      }
#else
      writepoly(&m, &b, b.outpolyfilename, holearray, m.holes, regionarray,
                m.regions, argc, argv);
#endif
    }
  }
#ifndef TRILIBRARY
#ifndef CDT_ONLY
  if (m.regions > 0) {
    trifree((VOID *) regionarray);
  }
#endif
  if (m.holes > 0) {
    trifree((VOID *) holearray);
  }
  if (b.geomview) {
    writeoff(&m, &b, b.offfilename, argc, argv);
  }
#endif
  if (b.edgesout) {
#ifdef TRILIBRARY
    writeedges(&m, &b, &out->edgelist, &out->edgemarkerlist);
#else
    writeedges(&m, &b, b.edgefilename, argc, argv);
#endif
  }
  if (b.voronoi) {
#ifdef TRILIBRARY
    writevoronoi(&m, &b, &vorout->pointlist, &vorout->pointattributelist,
                 &vorout->pointmarkerlist, &vorout->edgelist,
                 &vorout->edgemarkerlist, &vorout->normlist);
#else
    writevoronoi(&m, &b, b.vnodefilename, b.vedgefilename, argc, argv);
#endif
  }
  if (b.neighbors) {
#ifdef TRILIBRARY
    writeneighbors(&m, &b, &out->neighborlist);
#else
    writeneighbors(&m, &b, b.neighborfilename, argc, argv);
#endif
  }

  if (!b.quiet) {
#ifndef NO_TIMER
    gettimeofday(&tv6, &tz);
    printf("\nOutput milliseconds:  %ld\n",
           1000l * (tv6.tv_sec - tv5.tv_sec) +
           (tv6.tv_usec - tv5.tv_usec) / 1000l);
    printf("Total running milliseconds:  %ld\n",
           1000l * (tv6.tv_sec - tv0.tv_sec) +
           (tv6.tv_usec - tv0.tv_usec) / 1000l);
#endif

    statistics(&m, &b);
  }

#ifndef REDUCED
  if (b.docheck) {
    checkmesh(&m, &b);
    checkdelaunay(&m, &b);
  }
#endif

  triangledeinit(&m, &b);
#ifndef TRILIBRARY
  return 0;
#endif
}
