/*
 * This file is part of QDLDL, a library for performing the LDL^T factorization
 * of a symmetric indefinite matrix.
 *
 * QDLDL is part of the OSQP project, and is available at https://github.com/osqp/qdldl.
 *
 * Copyright 2018, Paul Goulart, Bartolomeo Stellato, Goran Banjac, Ian McInerney, The OSQP developers
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *  http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-ExternalRef: PACKAGE_MANAGER purl pkg:github/osqp/qdldl
 */
#ifndef QDLDL_H
#define QDLDL_H

#include "qdldl_types.h"
#include "qdldl_version.h"

#if defined(_WIN32)
#if defined(BUILDING_QDLDL)
#define QDLDL_API_EXPORT __declspec(dllexport)
#else
#define QDLDL_API_EXPORT __declspec(dllimport)
#endif
#else
#if defined(BUILDING_QDLDL)
#define QDLDL_API_EXPORT __attribute__((visibility("default")))
#else
#define QDLDL_API_EXPORT
#endif
#endif

#if defined(QDLDL_SHARED_LIB)
#define QDLDL_API QDLDL_API_EXPORT
#else
#define QDLDL_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

QDLDL_API QDLDL_int QDLDL_etree(const QDLDL_int n, const QDLDL_int* Ap, const QDLDL_int* Ai,
                                QDLDL_int* work, QDLDL_int* Lnz, QDLDL_int* etree);


QDLDL_API QDLDL_int QDLDL_factor(const QDLDL_int n, const QDLDL_int* Ap, const QDLDL_int* Ai,
                                 const QDLDL_float* Ax, QDLDL_int* Lp, QDLDL_int* Li,
                                 QDLDL_float* Lx, QDLDL_float* D, QDLDL_float* Dinv,
                                 const QDLDL_int* Lnz, const QDLDL_int* etree, QDLDL_bool* bwork,
                                 QDLDL_int* iwork, QDLDL_float* fwork);


QDLDL_API void QDLDL_solve(const QDLDL_int n, const QDLDL_int* Lp, const QDLDL_int* Li,
                           const QDLDL_float* Lx, const QDLDL_float* Dinv, QDLDL_float* x);


QDLDL_API void QDLDL_Lsolve(const QDLDL_int n, const QDLDL_int* Lp, const QDLDL_int* Li,
                            const QDLDL_float* Lx, QDLDL_float* x);


QDLDL_API void QDLDL_Ltsolve(const QDLDL_int n, const QDLDL_int* Lp, const QDLDL_int* Li,
                             const QDLDL_float* Lx, QDLDL_float* x);

#ifdef __cplusplus
}
#endif

#endif
