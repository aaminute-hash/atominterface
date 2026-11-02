---
title: 'AtomInterface: reliable interface modeling for atomistic materials simulations'
tags:
  - Python
  - materials science
  - interfaces
  - machine learning
  - atomistic simulation
  - density functional theory
authors:
  - name: 'Feng Qi'
    orcid: '0000-0000-0000-0000'
    affiliation: '1'
affiliations:
  - name: 'Guangdong Zixiang Technology Co., Ltd.'
    index: 1
date: 3 November 2026
bibliography: paper.bib
---

# Summary

AtomInterface is a Python middleware for building and evaluating atomistic
models of interfaces between two crystalline materials — the boundaries where a
battery electrode meets its electrolyte, a coating adheres to a substrate, or
two semiconductors form a heterojunction. Whether such an interface is stable,
and how strongly the two crystals adhere, is governed by atomic-scale geometry
and chemistry, and is typically probed with density-functional theory (DFT) or
with universal machine-learning interatomic potentials (U-MLIPs).

Obtaining a *trustworthy* answer is harder than it looks. A coherent interface
must be constructed so that the two lattices actually match; a U-MLIP must only
be asked about structures that fall inside its training distribution; and an
interface energy is assembled from several separately converged calculations,
any one of which can fail silently. AtomInterface wraps the existing building
blocks — the Atomic Simulation Environment (ASE), pymatgen, MACE and CHGNet —
and adds precisely this missing reliability layer. It enumerates
physically plausible interface candidates, runs every relaxation behind a guard
that watches for out-of-distribution and non-physical states, and only then
emits a converged interface energy together with a reproducible trail. It also
bridges the output of thirteen simulation codes (seven DFT, six molecular
dynamics) into one analysis pipeline, and exports cohesive-zone fracture inputs
for finite-element codes. AtomInterface is not a new force field and not a DFT
code: it is the trust layer that tells a researcher whether a candidate is
physical, whether the model is still reliable, and where to spend the next
expensive DFT calculation.

# Statement of need

Heterointerfaces govern the performance of a large fraction of functional
materials — solid-state battery electrodes, epitaxial films, thermal-barrier
and anti-corrosion coatings, and semiconductor heterostructures. Predicting
their stability and adhesion requires (i) constructing an atomistic interface
whose two lattices are coherently matched, (ii) relaxing it with a model whose
reliability is known, and (iii) combining the results into a well-defined
interfacial free energy. Each of these three steps has a failure mode that is
easy to overlook and costly to miss. A naive lattice match can produce a
structure that is geometrically strained but chemically nonsensical. A U-MLIP
can be extrapolated far outside its training data, returning smooth but wrong
energies. And an interface energy is a small difference of large numbers, so a
single unconverged reference calculation silently corrupts the final number.

Existing tools cover individual pieces of this pipeline. pymatgen provides a
coherent-interface builder [@Ong2013], ASE provides calculator and relaxation
abstractions [@Larsen2017], and MACE and CHGNet provide pre-trained U-MLIPs
[@Batatia2022; @Deng2023]. What they do not provide is the connective,
reliability-oriented layer in between: out-of-distribution monitoring during
relaxation, fail-closed assembly of the energy ledger, and a uniform unit and
sign convention across the DFT and molecular-dynamics inputs that feed the same
analysis. AtomInterface fills that gap for researchers who build or evaluate
interfaces and need the result to be defensible, not just computable.

# State of the field

Several packages overlap with parts of AtomInterface. pymatgen's
`CoherentInterfaceBuilder` implements the Zur–McGill lattice-match approach
[@Zur1984] and is the de facto standard for generating interface candidates;
AtomInterface deliberately builds on it rather than reimplementing it. ASE is
the universal substrate for calculators, relaxations and trajectory I/O.
AiiDA and similar workflow engines solve *provenance and orchestration*;
AtomInterface does not aim to replace them. What distinguishes AtomInterface is
the reliability guard and the fail-closed energy ledger, which none of these
packages provide as a default-on layer.

The "build vs. contribute" judgment is therefore: the lattice-matching and
force-field capabilities belong to pymatgen, ASE, MACE and CHGNet, and the
right contribution is to *consume* them behind a guard, not to reimplement
them. AtomInterface's unique scholarly contribution is the guard itself — a
relaxation driver that quarantines out-of-distribution local environments and
writes a DFT single-point request instead of silently trusting the model — plus
a thermo layer that refuses to emit an interface energy unless every reference
energy is finite and converged, and a code-agnostic bridge that normalizes the
outputs of thirteen codes to one physical convention.

# Software design

The package is organized as a pipeline of four layers. The **builders** layer
enumerates coherent interface candidates using pymatgen's
`CoherentInterfaceBuilder`, then filters them by in-plane strain and lateral
registry. The **relax-and-guard** layer runs every relaxation behind a guard
that is *always on*, not opt-in: it checks finite energies and forces, watches
the local atomic environments for out-of-distribution states, and on
triggering pauses the run, excises the offending cluster, and writes a DFT
single-point input. The **thermo** layer assembles interface energies, work of
separation and cohesive-zone parameters from the relaxed structures and their
pristine-bulk references. The **physics** and **export** layers normalize
trajectory data and emit downstream inputs (VASP/QE, MOOSE/UMAT/VUMAT).

Two design decisions matter for correctness. First, *fail-closed assembly*:
the energy ledger is populated only when every contributing calculation is
finite and converged; otherwise the cell is left empty and the result is marked
invalid, rather than substituting a stand-in value. Second, *one physical
convention everywhere*: energies are electronvolts, stresses are symmetric
3×3 tension-positive tensors in eV/Å³, and forces are in eV/Å, so that DFT and
molecular-dynamics inputs produced by different codes enter the same analysis
without unit or sign errors. The package is built on NumPy [@Harris2020] and
the Atomic Simulation Environment [@Larsen2017]. Physical invariance — energy
under rigid rotation, forces as vectors, stress as a rank-2 tensor — is tested
directly.

# Research impact statement

AtomInterface has been validated end to end on a real system: the Cu/Ni(001)
interface, whose interfacial energy is reproduced as 0.059 ± 0.003 J/m² against
a VASP plane-wave cross-check, with the full candidate table and the (111)/(110)
orientation archives published alongside the validation notes for
reproducibility. The molecular-dynamics path is validated on real LAMMPS and
CP2K trajectories (stress and strain fields bit-exact against the codes' own
outputs), and the bridge layer is verified bit-exact against real output from
VASP, Quantum ESPRESSO, SIESTA and ABACUS. The project ships an 821-test suite
covering physical invariance and API surface, with continuous integration on
multiple Python versions. These are concrete, reproducible signals of
correctness rather than aspirational claims; the expected near-term impact is a
reusable trust layer for interface screening in battery and heterostructure
design, delivered through upstream integration with the ASE and pymatgen
ecosystems.

# AI usage disclosure

Generative AI (an LLM-driven coding assistant) was used during development to
draft portions of the source code, documentation and this manuscript. All
AI-generated content was treated as an untrusted draft: it was reviewed by a
human, checked by static analysis (ruff and mypy), exercised by the 821-test
suite, and — where a claim concerns physical output — verified bit-exact
against the originating code's own reported values. No AI-generated claim was
accepted without an independent computational check.

# Acknowledgements

We acknowledge the developers of ASE, pymatgen, MACE and CHGNet, on which
AtomInterface builds.

# References
