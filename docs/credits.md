# Credits and licence

`tails` is MIT-licensed (© 2026 Charles-Auguste Marois and rare-labs contributors).

**`tails.idf` is a port of [IDFCurves.jl](https://github.com/JuliaExtremes/IDFCurves.jl)** (MIT,
commit `66a34d1`). The models, initialisation strategy, interval formulas and scaling test follow
that implementation, and `tails` reproduces its estimates to ~1e-9 (see [Validation](validation.md)).
Attribution and third-party licence text are in the repository `NOTICE` file.

Other methods: Koutsoyiannis et al. (1998) d-GEV; Martins & Stedinger (2000) GMLE; Hosking (1990)
L-moments; Coles (2001); Bader, Yan & Zhang (2018) ForwardStop; Ferro & Segers (2003); Smith &
Weissman (1994); Glick (1978) records.

**Data**: Environment and Climate Change Canada, Engineering Climate Datasets – IDF v3.40
(2025-12-05), Open Government Licence – Canada. The three CSVs bundled with IDFCurves.jl are reused
under its MIT licence. See the repository `NOTICE` file.
