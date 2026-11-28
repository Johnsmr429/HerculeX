#ifndef PITCH_TABLE_INCLUDED
#define PITCH_TABLE_INCLUDED

typedef struct PitchTable *PitchTable;

int PitchTable_load(const char *path, PitchTable *out);
void PitchTable_free(PitchTable table);

double PitchTable_pitch_at(PitchTable table, double r);

double PitchTable_dturns(PitchTable table, double r_a, double r_b,
                         double scalar_fallback);

PitchTable PitchTable_from_env(void);

int PitchTable_knots(PitchTable table);

int PitchTable_selftest(void);

#endif
