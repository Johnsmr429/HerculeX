#ifndef CLOSEUP_CTTRACK_INCLUDED
#define CLOSEUP_CTTRACK_INCLUDED

int CloseupCtTrack_audit(const char *surface_stem, const char *raw,
                         const char *out_stem, const double box[6], double seed_z);
int CloseupCtTrack_selftest(void);

#endif
