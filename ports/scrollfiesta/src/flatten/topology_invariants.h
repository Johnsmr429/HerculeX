#ifndef TOPOLOGY_INVARIANTS_INCLUDED
#define TOPOLOGY_INVARIANTS_INCLUDED

#include <stddef.h>
#include <stdint.h>

enum {
    TOPOLOGY_DEFECT_NONMANIFOLD_EDGE       = 1u << 0,
    TOPOLOGY_DEFECT_NONMANIFOLD_VERTEX     = 1u << 1,
    TOPOLOGY_DEFECT_IRREGULAR_BOUNDARY     = 1u << 2,
    TOPOLOGY_DEFECT_NONORIENTABLE          = 1u << 3,
    TOPOLOGY_DEFECT_BOUNDARY_COUNT         = 1u << 4,
    TOPOLOGY_DEFECT_NONTRIVIAL_H1          = 1u << 5,
    TOPOLOGY_DEFECT_NONTRIVIAL_H2          = 1u << 6,
    TOPOLOGY_DEFECT_INCONSISTENT_WINDING   = 1u << 7,
    TOPOLOGY_DEFECT_GENERATOR_FAILURE      = 1u << 8,
    TOPOLOGY_DEFECT_ALGEBRA_MISMATCH       = 1u << 9
};

typedef struct {
    int emit_generators;

    size_t max_generators_per_component;
} TopologyAuditOptions;

typedef struct {
    size_t rank;
    int32_t root_vertex;

    size_t vertices, edges, faces;
    int has_geometry;
    double bbox_min[3], bbox_max[3];
    double centroid[3];
    size_t boundary_edges;
    size_t boundary_loops;
    size_t nonmanifold_edges, nonmanifold_vertices;
    size_t same_direction_edges;
    size_t boundary_irregular_vertices;

    int64_t euler_characteristic;
    int orientable;
    int input_winding_consistent;
    int surface_valid;

    int64_t beta_0, beta_1, beta_2;

    int64_t orientable_genus;
    int64_t crosscap_number;

    int homeomorphic_to_disk;
    uint32_t defect_mask;

    size_t minimal_generator_rank;
    size_t emitted_generators;
    int generator_basis_complete;
} TopologyComponentInvariant;

typedef struct {
    size_t component;
    size_t ordinal;
    int32_t closing_edge_a;
    int32_t closing_edge_b;

    double length;
    double rooted_length;

    size_t nvertices;
    int32_t *vertices;
} TopologyGenerator;

typedef struct {
    size_t rank;
    size_t component;
    size_t ordinal;
    int32_t root_vertex;

    size_t vertices, edges;
    size_t irregular_vertices;
    int simple_cycle;
    int component_perimeter;

    int has_geometry;
    double bbox_min[3], bbox_max[3];
    double length;
    double max_edge_length;
    double diameter;
    int diameter_exact;
} TopologyBoundaryInvariant;

typedef struct {
    size_t vertices, edges, faces;
    size_t face_components;
    size_t isolated_vertices;
    size_t topological_components;

    size_t disk_components;
    size_t nondisk_components;
    size_t invalid_surface_components;
    size_t nonorientable_components;
    size_t inconsistent_winding_components;

    int64_t euler_characteristic;
    int64_t beta_0, beta_1, beta_2;
    int betti_complete;
    int all_components_are_disks;

    size_t minimal_generator_rank;
    size_t emitted_generators;
    int generators_truncated;
    int generator_basis_complete;

    TopologyComponentInvariant *component;
    TopologyGenerator *generator;

    size_t boundary_components;
    TopologyBoundaryInvariant *boundary;

    int32_t *vertex_component;

    char error[256];
} TopologyAuditReport;

void TopologyAudit_options_default(TopologyAuditOptions *options);

int TopologyAudit_analyze(const float *verts, size_t nv,
                          const int32_t *faces, size_t nf,
                          const TopologyAuditOptions *options,
                          TopologyAuditReport *report);

void TopologyAudit_dispose(TopologyAuditReport *report);

int TopologyAudit_selftest(void);

#endif
