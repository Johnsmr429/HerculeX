

#ifndef __ENERGY_H__
#define __ENERGY_H__

#include <assert.h>
#include "graph.h"

template <typename captype, typename tcaptype, typename flowtype> class Energy: public Graph<captype,tcaptype,flowtype>
{
	typedef Graph<captype,tcaptype,flowtype> GraphT;
public:
	typedef typename GraphT::node_id Var;

	typedef captype Value;
	typedef flowtype TotalValue;


	Energy(int var_num_max, int edge_num_max, void (*err_function)(const char *) = NULL);

	~Energy();

	Var add_variable(int num=1);

	void add_constant(Value E);

	void add_term1(Var x,
	               Value E0, Value E1);

	void add_term2(Var x, Var y,
	               Value E00, Value E01,
	               Value E10, Value E11);

	void add_term3(Var x, Var y, Var z,
	               Value E000, Value E001,
	               Value E010, Value E011,
	               Value E100, Value E101,
	               Value E110, Value E111);

	TotalValue minimize();

	int get_var(Var x);


private:

	TotalValue	Econst;
	void		(*error_function)(const char *);


};
















template <typename captype, typename tcaptype, typename flowtype>
inline Energy<captype,tcaptype,flowtype>::Energy(int var_num_max, int edge_num_max, void (*err_function)(const char *)) : Graph<captype,tcaptype,flowtype>(var_num_max, edge_num_max, err_function)
{
	Econst = 0;
	error_function = err_function;
}

template <typename captype, typename tcaptype, typename flowtype>
inline Energy<captype,tcaptype,flowtype>::~Energy() {}

template <typename captype, typename tcaptype, typename flowtype>
inline typename Energy<captype,tcaptype,flowtype>::Var Energy<captype,tcaptype,flowtype>::add_variable(int num)
{	return GraphT::add_node(num); }

template <typename captype, typename tcaptype, typename flowtype>
inline void Energy<captype,tcaptype,flowtype>::add_constant(Value A) { Econst += A; }

template <typename captype, typename tcaptype, typename flowtype>
inline void Energy<captype,tcaptype,flowtype>::add_term1(Var x,
                              Value A, Value B)
{
	this->add_tweights(x, B, A);
}

template <typename captype, typename tcaptype, typename flowtype>
inline void Energy<captype,tcaptype,flowtype>::add_term2(Var x, Var y,
                              Value A, Value B,
                              Value C, Value D)
{
	this->add_tweights(x, D, A);
	B -= A; C -= D;


	assert(B + C >= 0);
	if (B < 0)
	{
			this->add_tweights(x, 0, B);
		this->add_tweights(y, 0, -B);
		this->add_edge(x, y, 0, B+C);
	}
	else if (C < 0)
	{
			this->add_tweights(x, 0, -C);
		this->add_tweights(y, 0, C);
		this->add_edge(x, y, B+C, 0);
	}
	else
	{
		this->add_edge(x, y, B, C);
	}
}

template <typename captype, typename tcaptype, typename flowtype>
inline void Energy<captype,tcaptype,flowtype>::add_term3(Var x, Var y, Var z,
                              Value E000, Value E001,
                              Value E010, Value E011,
                              Value E100, Value E101,
                              Value E110, Value E111)
{
Value pi = (E000 + E011 + E101 + E110) - (E100 + E010 + E001 + E111);
Value delta;
Var u;

	if (pi >= 0)
	{
		Econst += E111 - (E011 + E101 + E110);

		add_tweights(x, E101, E001);
		add_tweights(y, E110, E100);
		add_tweights(z, E011, E010);

		delta = (E010 + E001) - (E000 + E011);
		assert(delta >= 0);
		add_edge(y, z, delta, 0);

		delta = (E100 + E001) - (E000 + E101);
		assert(delta >= 0);
		add_edge(z, x, delta, 0);

		delta = (E100 + E010) - (E000 + E110);
		assert(delta >= 0);
		add_edge(x, y, delta, 0);

		if (pi > 0)
		{
			u = add_variable();
			add_edge(x, u, pi, 0);
			add_edge(y, u, pi, 0);
			add_edge(z, u, pi, 0);
			add_tweights(u, 0, pi);
		}
	}
	else
	{
		Econst += E000 - (E100 + E010 + E001);

		add_tweights(x, E110, E010);
		add_tweights(y, E011, E001);
		add_tweights(z, E101, E100);

		delta = (E110 + E101) - (E100 + E111);
		assert(delta >= 0);
		add_edge(z, y, delta, 0);

		delta = (E110 + E011) - (E010 + E111);
		assert(delta >= 0);
		add_edge(x, z, delta, 0);

		delta = (E101 + E011) - (E001 + E111);
		assert(delta >= 0);
		add_edge(y, x, delta, 0);

		u = add_variable();
		add_edge(u, x, -pi, 0);
		add_edge(u, y, -pi, 0);
		add_edge(u, z, -pi, 0);
		this->add_tweights(u, -pi, 0);
	}
}

template <typename captype, typename tcaptype, typename flowtype>
inline typename Energy<captype,tcaptype,flowtype>::TotalValue Energy<captype,tcaptype,flowtype>::minimize() {
return Econst + GraphT::maxflow(); }

template <typename captype, typename tcaptype, typename flowtype>
inline int Energy<captype,tcaptype,flowtype>::get_var(Var x) { return (int) this->what_segment(x); }

#endif
