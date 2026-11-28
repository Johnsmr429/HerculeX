
#ifndef __ATTENE_PREDICATES_H_
#define __ATTENE_PREDICATES_H_

#include <stdint.h>
#include <float.h>
#include <math.h>
#include <fenv.h>
#include <string.h>
#include "attene_vec.h"

#pragma intrinsic(fabs)

#if defined(_M_X64) || defined(__x86_64__)
#  ifdef __SSE2__
#    define ATTENE_USE_SIMD
#  endif
#  ifdef __AVX2__
#    define ATTENE_USE_SIMD
#    define ATTENE_USE_AVX2
#  endif
#  ifdef _MSC_VER
#    define ATTENE_USE_SIMD
#    include <emmintrin.h>
#  endif
#endif


inline void attene_setFPUModeToRoundUP() { fesetround(FE_UPWARD); }
inline void attene_setFPUModeToRoundNEAR() { fesetround(FE_TONEAREST); }


#pragma float_control(push)
#pragma float_control(precise, on)
#pragma fenv_access(on)

struct attene_interval {
	double min_low;
	double high;

	attene_interval() : min_low(0.0), high(0.0) {}

	attene_interval(double a) : min_low(-a), high(a) {}

	attene_interval(double _low, double _high) : min_low(-_low), high(_high) {}

	double inf() const { return -min_low; }
	double sup() const { return high; }

	double estimate() const { return (inf() + sup()) * 0.5; }

	bool isNegative() const { return high < 0; }
	bool isPositive() const { return min_low < 0; }
	bool containsZero() const { return min_low >= 0 && high >= 0; }

	bool signIsReliable() const { return !containsZero(); }

	int sign() const {
		if (isPositive()) return 1;
		if (isNegative()) return -1;
		return 0;
	}

	bool isNAN() const { return high != high; }

	attene_interval operator-() const {
		attene_interval r;
		r.min_low = high;
		r.high = min_low;
		return r;
	}

	attene_interval operator+(const attene_interval& b) const {
		attene_interval r;
		r.min_low = min_low + b.min_low;
		r.high = high + b.high;
		return r;
	}

	attene_interval operator-(const attene_interval& b) const {
		attene_interval r;
		r.min_low = min_low + b.high;
		r.high = high + b.min_low;
		return r;
	}

	attene_interval operator*(const attene_interval& b) const {
										double a1 = -min_low;
		double b1 = high;
		double a2 = -b.min_low;
		double b2 = b.high;

			double h = a1 * a2;
		double t = a1 * b2; if (t > h) h = t;
		t = b1 * a2; if (t > h) h = t;
		t = b1 * b2; if (t > h) h = t;

				double na1 = min_low;
		double nb1 = -high;
		double m = na1 * a2;
		t = na1 * b2; if (t > m) m = t;
		t = nb1 * a2; if (t > m) m = t;
		t = nb1 * b2; if (t > m) m = t;

		attene_interval r;
		r.min_low = m;
		r.high = h;
		return r;
	}

	attene_interval operator*(double b) const {
		attene_interval r;
		if (b >= 0) {
			r.min_low = min_low * b;
			r.high = high * b;
		} else {
			r.min_low = high * (-b);
			r.high = min_low * (-b);
		}
		return r;
	}

	attene_interval sqr() const {
		attene_interval r;
		if (min_low <= 0) {
			r.min_low = 0;
			r.high = high * high;
		} else if (high <= 0) {
			r.min_low = 0;
			r.high = min_low * min_low;
		} else {
			r.min_low = 0;
			double h2 = high * high;
			double l2 = min_low * min_low;
			r.high = (h2 > l2) ? h2 : l2;
		}
		return r;
	}
};

inline int attene_sgn(const attene_interval& p) {
	return (p.isPositive()) ? 1 : ((p.isNegative()) ? -1 : 0);
}

#pragma fenv_access(off)
#pragma float_control(pop)


#define ATTENE_EXPANSION_MAX_SIZE 256

struct attene_expansion {
	int length;
	double components[ATTENE_EXPANSION_MAX_SIZE];

	attene_expansion() : length(0) {}

	attene_expansion(double a) : length(1) { components[0] = a; }

	double estimate() const {
		double sum = 0;
		for (int i = length - 1; i >= 0; i--) {
			sum += components[i];
		}
		return sum;
	}

	int sign() const {
		for (int i = length - 1; i >= 0; i--) {
			if (components[i] > 0) return 1;
			if (components[i] < 0) return -1;
		}
		return 0;
	}
};

#pragma float_control(push)
#pragma float_control(precise, on)
#pragma fenv_access(on)

inline void attene_TwoSum(double a, double b, double& x, double& y) {
	x = a + b;
	double bv = x - a;
	double av = x - bv;
	double br = b - bv;
	double ar = a - av;
	y = ar + br;
}

inline void attene_TwoDiff(double a, double b, double& x, double& y) {
	x = a - b;
	double bv = a - x;
	double av = x + bv;
	double br = bv - b;
	double ar = a - av;
	y = ar + br;
}

inline void attene_Split(double a, double& ahi, double& alo) {
	const double splitter = 134217729.0;
	double c = splitter * a;
	double abig = c - a;
	ahi = c - abig;
	alo = a - ahi;
}

inline void attene_TwoProd(double a, double b, double& x, double& y) {
	x = a * b;
	double ahi, alo, bhi, blo;
	attene_Split(a, ahi, alo);
	attene_Split(b, bhi, blo);
	double err1 = x - (ahi * bhi);
	double err2 = err1 - (alo * bhi);
	double err3 = err2 - (ahi * blo);
	y = (alo * blo) - err3;
}

#pragma fenv_access(off)
#pragma float_control(pop)

int attene_growExpansion(int elen, const double* e, double b, double* h);
int attene_growExpansionZeroElim(int elen, const double* e, double b, double* h);
int attene_fastExpansionSumZeroElim(int elen, const double* e, int flen, const double* f, double* h);
int attene_scaleExpansionZeroElim(int elen, const double* e, double b, double* h);

attene_expansion operator+(const attene_expansion& a, const attene_expansion& b);
attene_expansion operator-(const attene_expansion& a, const attene_expansion& b);
attene_expansion operator*(const attene_expansion& a, double b);
attene_expansion operator*(double a, const attene_expansion& b);
attene_expansion operator*(const attene_expansion& a, const attene_expansion& b);

inline attene_expansion attene_sqr(const attene_expansion& a) {
	return a * a;
}

inline int attene_sgn(const attene_expansion& e) {
	return e.sign();
}


int attene_orient3d_interval(
	double ax, double ay, double az,
	double bx, double by, double bz,
	double cx, double cy, double cz,
	double dx, double dy, double dz);

int attene_orient3d_expansion(
	double ax, double ay, double az,
	double bx, double by, double bz,
	double cx, double cy, double cz,
	double dx, double dy, double dz);

int attene_orient3d(
	double ax, double ay, double az,
	double bx, double by, double bz,
	double cx, double cy, double cz,
	double dx, double dy, double dz);

int attene_orient3d(const Vec3d& a, const Vec3d& b, const Vec3d& c, const Vec3d& d);


int attene_orient2d_interval(
	double ax, double ay,
	double bx, double by,
	double cx, double cy);

int attene_orient2d_expansion(
	double ax, double ay,
	double bx, double by,
	double cx, double cy);

int attene_orient2d(
	double ax, double ay,
	double bx, double by,
	double cx, double cy);

int attene_orient2d(const Vec2d& a, const Vec2d& b, const Vec2d& c);


int attene_compare_ratios_same_denom_interval(
	const attene_interval& a, const attene_interval& b, const attene_interval& d);

int attene_compare_ratios_interval(
	const attene_interval& a, const attene_interval& da,
	const attene_interval& b, const attene_interval& db);

#endif
