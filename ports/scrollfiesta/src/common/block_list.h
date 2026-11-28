#ifndef BLOCK_LIST_INCLUDED
#define BLOCK_LIST_INCLUDED

int BlockList_read(const char *path, int *n_out, char ***blocks_out);

const char *BlockList_arg(const char *arg);

int BlockList_selftest(void);

#endif
