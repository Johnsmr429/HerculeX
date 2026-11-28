/*  Copyright 1996, 2005                                                     */



struct triangulateio {
  REAL *pointlist;
  REAL *pointattributelist;
  int *pointmarkerlist;
  int numberofpoints;
  int numberofpointattributes;

  int *trianglelist;
  REAL *triangleattributelist;
  REAL *trianglearealist;
  int *neighborlist;
  int numberoftriangles;
  int numberofcorners;
  int numberoftriangleattributes;

  int *segmentlist;
  int *segmentmarkerlist;
  int numberofsegments;

  REAL *holelist;
  int numberofholes;

  REAL *regionlist;
  int numberofregions;

  int *edgelist;
  int *edgemarkerlist;
  REAL *normlist;
  int numberofedges;
};

#ifdef ANSI_DECLARATORS
void triangulate(char *, struct triangulateio *, struct triangulateio *,
                 struct triangulateio *);
void trifree(VOID *memptr);
#else
void triangulate();
void trifree();
#endif
