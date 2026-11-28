



#ifndef __GRAPH_H__
#define __GRAPH_H__

#include <string.h>
#include "block.h"

#include <assert.h>



template <typename captype, typename tcaptype, typename flowtype> class Graph
{
public:
	typedef enum
	{
		SOURCE	= 0,
		SINK	= 1
	} termtype;
	typedef int node_id;


	Graph(int node_num_max, int edge_num_max, void (*err_function)(const char *) = NULL);

	~Graph();

	node_id add_node(int num = 1);

	void add_edge(node_id i, node_id j, captype cap, captype rev_cap);

	void add_tweights(node_id i, tcaptype cap_source, tcaptype cap_sink);


	flowtype maxflow(bool reuse_trees = false, Block<node_id>* changed_list = NULL);

	termtype what_segment(node_id i, termtype default_segm = SOURCE);




private:
	struct node;
	struct arc;

public:


	void reset();


	typedef arc* arc_id;
	arc_id get_first_arc();
	arc_id get_next_arc(arc_id a);

	int get_node_num() { return node_num; }
	int get_arc_num() { return (int)(arc_last - arcs); }
	void get_arc_ends(arc_id a, node_id& i, node_id& j);


	tcaptype get_trcap(node_id i);
	captype get_rcap(arc* a);


	void set_trcap(node_id i, tcaptype trcap);
	void set_rcap(arc* a, captype rcap);


	void mark_node(node_id i);

	void remove_from_changed_list(node_id i)
	{
		assert(i>=0 && i<node_num && nodes[i].is_in_changed_list);
		nodes[i].is_in_changed_list = 0;
	}

	void Copy(Graph<captype, tcaptype, flowtype>* g0);





private:

	struct node
	{
		arc			*first;

		arc			*parent;
		node		*next;
									int			TS;
		int			DIST;
		int			is_sink : 1;
		int			is_marked : 1;
		int			is_in_changed_list : 1;

		tcaptype	tr_cap;

	};

	struct arc
	{
		node		*head;
		arc			*next;
		arc			*sister;

		captype		r_cap;
	};

	struct nodeptr
	{
		node    	*ptr;
		nodeptr		*next;
	};
	static const int NODEPTR_BLOCK_SIZE = 128;

	node				*nodes, *node_last, *node_max;
	arc					*arcs, *arc_last, *arc_max;

	int					node_num;

	DBlock<nodeptr>		*nodeptr_block;

	void	(*error_function)(const char *);

	flowtype			flow;

	int					maxflow_iteration;
	Block<node_id>		*changed_list;


	node				*queue_first[2], *queue_last[2];
	nodeptr				*orphan_first, *orphan_last;
	int					TIME;


	void reallocate_nodes(int num);
	void reallocate_arcs();

	void set_active(node *i);
	node *next_active();

	void set_orphan_front(node* i);
	void set_orphan_rear(node* i);

	void add_to_changed_list(node* i);

	void maxflow_init();
	void maxflow_reuse_trees_init();
	void augment(arc *middle_arc);
	void process_source_orphan(node *i);
	void process_sink_orphan(node *i);

	void test_consistency(node* current_node=NULL);
};














template <typename captype, typename tcaptype, typename flowtype>
	inline typename Graph<captype,tcaptype,flowtype>::node_id Graph<captype,tcaptype,flowtype>::add_node(int num)
{
	assert(num > 0);

	if (node_last + num > node_max) reallocate_nodes(num);

	if (num == 1)
	{
		node_last -> first = NULL;
		node_last -> tr_cap = 0;
		node_last -> is_marked = 0;
		node_last -> is_in_changed_list = 0;

		node_last ++;
		return node_num ++;
	}
	else
	{
		memset(node_last, 0, num*sizeof(node));

		node_id i = node_num;
		node_num += num;
		node_last += num;
		return i;
	}
}

template <typename captype, typename tcaptype, typename flowtype>
	inline void Graph<captype,tcaptype,flowtype>::add_tweights(node_id i, tcaptype cap_source, tcaptype cap_sink)
{
	assert(i >= 0 && i < node_num);

	tcaptype delta = nodes[i].tr_cap;
	if (delta > 0) cap_source += delta;
	else           cap_sink   -= delta;
	flow += (cap_source < cap_sink) ? cap_source : cap_sink;
	nodes[i].tr_cap = cap_source - cap_sink;
}

template <typename captype, typename tcaptype, typename flowtype>
	inline void Graph<captype,tcaptype,flowtype>::add_edge(node_id _i, node_id _j, captype cap, captype rev_cap)
{
	assert(_i >= 0 && _i < node_num);
	assert(_j >= 0 && _j < node_num);
	assert(_i != _j);
	assert(cap >= 0);
	assert(rev_cap >= 0);

	if (arc_last == arc_max) reallocate_arcs();

	arc *a = arc_last ++;
	arc *a_rev = arc_last ++;

	node* i = nodes + _i;
	node* j = nodes + _j;

	a -> sister = a_rev;
	a_rev -> sister = a;
	a -> next = i -> first;
	i -> first = a;
	a_rev -> next = j -> first;
	j -> first = a_rev;
	a -> head = j;
	a_rev -> head = i;
	a -> r_cap = cap;
	a_rev -> r_cap = rev_cap;
}

template <typename captype, typename tcaptype, typename flowtype>
	inline typename Graph<captype,tcaptype,flowtype>::arc* Graph<captype,tcaptype,flowtype>::get_first_arc()
{
	return arcs;
}

template <typename captype, typename tcaptype, typename flowtype>
	inline typename Graph<captype,tcaptype,flowtype>::arc* Graph<captype,tcaptype,flowtype>::get_next_arc(arc* a)
{
	return a + 1;
}

template <typename captype, typename tcaptype, typename flowtype>
	inline void Graph<captype,tcaptype,flowtype>::get_arc_ends(arc* a, node_id& i, node_id& j)
{
	assert(a >= arcs && a < arc_last);
	i = (node_id) (a->sister->head - nodes);
	j = (node_id) (a->head - nodes);
}

template <typename captype, typename tcaptype, typename flowtype>
	inline tcaptype Graph<captype,tcaptype,flowtype>::get_trcap(node_id i)
{
	assert(i>=0 && i<node_num);
	return nodes[i].tr_cap;
}

template <typename captype, typename tcaptype, typename flowtype>
	inline captype Graph<captype,tcaptype,flowtype>::get_rcap(arc* a)
{
	assert(a >= arcs && a < arc_last);
	return a->r_cap;
}

template <typename captype, typename tcaptype, typename flowtype>
	inline void Graph<captype,tcaptype,flowtype>::set_trcap(node_id i, tcaptype trcap)
{
	assert(i>=0 && i<node_num);
	nodes[i].tr_cap = trcap;
}

template <typename captype, typename tcaptype, typename flowtype>
	inline void Graph<captype,tcaptype,flowtype>::set_rcap(arc* a, captype rcap)
{
	assert(a >= arcs && a < arc_last);
	a->r_cap = rcap;
}


template <typename captype, typename tcaptype, typename flowtype>
	inline typename Graph<captype,tcaptype,flowtype>::termtype Graph<captype,tcaptype,flowtype>::what_segment(node_id i, termtype default_segm)
{
	if (nodes[i].parent)
	{
		return (nodes[i].is_sink) ? SINK : SOURCE;
	}
	else
	{
		return default_segm;
	}
}

template <typename captype, typename tcaptype, typename flowtype>
	inline void Graph<captype,tcaptype,flowtype>::mark_node(node_id _i)
{
	node* i = nodes + _i;
	if (!i->next)
	{
			if (queue_last[1]) queue_last[1] -> next = i;
		else               queue_first[1]        = i;
		queue_last[1] = i;
		i -> next = i;
	}
	i->is_marked = 1;
}


#endif
