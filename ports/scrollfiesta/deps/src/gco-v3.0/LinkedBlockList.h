
#ifndef __LINKEDBLOCKLIST_H__
#define __LINKEDBLOCKLIST_H__

#define GCLL_BLOCK_SIZE 4
typedef char BlockType;

typedef void * ListType;

class LinkedBlockList{

public:
	void addFront(ListType item);
	inline bool isEmpty(){if (m_head == 0) return(true); else return(false);};
	inline LinkedBlockList(){m_head = 0; m_head_block_size = GCLL_BLOCK_SIZE;};
	~LinkedBlockList();

	inline void setCursorFront(){m_cursor = m_head; m_cursor_ind = 0;};
	ListType next();
	bool hasNext();

private:
	typedef struct LLBlockStruct{
		ListType m_item[GCLL_BLOCK_SIZE];
		struct LLBlockStruct *m_next;
	} LLBlock;

	LLBlock *m_head;
	BlockType m_head_block_size;
	BlockType m_cursor_ind;
	LLBlock *m_cursor;
};

#endif
