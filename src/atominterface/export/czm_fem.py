"""Export CZM parameters to FEM inputs (MOOSE / Abaqus UMAT / VUMAT text)."""

from __future__ import annotations

from pathlib import Path


def export_czm_moose(
    czm_params: dict,
    *,
    interface_id: str = "interface0",
    out_path: str | Path = "interface_czm.i",
) -> Path:
    """Emit a MOOSE input snippet with CohesiveZoneMasterMaterial params."""
    p = Path(out_path)
    sigma = float(czm_params.get("max_traction", 0.0))
    delta = float(czm_params.get("critical_separation", 0.0))
    Gc = float(czm_params.get("fracture_energy", 0.0))
    txt = f"""# AtomInterface-generated CZM parameters for {interface_id}
# sigma_max = {sigma:.6e}
# delta_c   = {delta:.6e}
# G_c       = {Gc:.6e}

[Materials]
  [czm]
    type = CohesiveZoneMasterMaterial
    interface_ids = {interface_id}
    max_traction = {sigma:.6e}
    critical_separation = {delta:.6e}
    fracture_energy = {Gc:.6e}
  []
[]
"""
    p.write_text(txt, encoding="utf-8")
    return p


def export_czm_umat(
    czm_params: dict,
    *,
    interface_id: str = "interface0",
    out_path: str | Path = "interface_czm_umat.f",
) -> Path:
    """Emit a bilinear cohesive-zone UMAT for Abaqus/Standard (implicit).

    Physics convention (bilinear cohesive law, traction-separation):

    * Loading: ``T = max_traction * (delta / critical_separation)`` for
      ``0 <= delta <= critical_separation``; after the critical separation
      the interface is fully failed (``T = 0``).
    * Fracture energy: ``Gc = 0.5 * max_traction * critical_separation``.
      The three parameters are related, so all three values are embedded in
      the generated source and their relation is written as a comment.
    * Unloading/reloading follows the origin secant. The damage variable is
      ``D = delta_max / critical_separation`` with the history maximum
      separation ``delta_max`` stored in ``STATEV(1)``; ``D`` is monotonic
      non-decreasing and capped at 1.

    The generated Fortran uses the standard Abaqus/Standard ``UMAT``
    signature. ``PROPS`` carries, in order, ``max_traction``,
    ``critical_separation``, ``fracture_energy``; ``STATEV(1)`` stores
    ``delta_max`` (and ``STATEV(2)`` stores ``D`` for postprocessing).
    """
    p = Path(out_path)
    sigma = float(czm_params.get("max_traction", 0.0))
    delta = float(czm_params.get("critical_separation", 0.0))
    gc = float(czm_params.get("fracture_energy", 0.0))
    s_sigma = f"{sigma:.6e}"
    s_delta = f"{delta:.6e}"
    s_gc = f"{gc:.6e}"
    txt = f"""      SUBROUTINE UMAT(STRESS,STATEV,DDSDDE,SSE,SPD,SCD,
     1 RPL,DDSDDT,DRPLDE,DRPLDT,
     2 STRAN,DSTRAN,TIME,DTIME,TEMP,DTEMP,PREDEF,DPRED,CMNAME,
     3 NDI,NSHR,NTENS,NSTATV,PROPS,NPROPS,COORDS,DROT,PNEWDT,
     4 CELENT,DFGRD0,DFGRD1,NOEL,NPT,LAYER,KSPT,JSTEP,KINC)
C
C     AtomInterface bilinear cohesive-zone UMAT (Abaqus/Standard)
C     Interface id: {interface_id}
C
C     PROPS carries, in order:
C       PROPS(1) = max_traction        = {s_sigma}
C       PROPS(2) = critical_separation = {s_delta}
C       PROPS(3) = fracture_energy     = {s_gc}
C
C     Bilinear cohesive law (traction-separation):
C       loading : T = max_traction * (delta / critical_separation)
C                 for 0 <= delta <= critical_separation
C       failed  : T = 0 for delta > critical_separation
C
C     Fracture energy relation (area under the T-delta curve):
C       Gc = 0.5 * max_traction * critical_separation
C
C     Damage and unloading: D = delta_max / critical_separation (secant
C     to the origin). Unloading/reloading retraces the loading line
C     T = K0 * delta, K0 = max_traction / critical_separation.
C     STATEV(1) stores the history maximum separation delta_max
C     (D monotonic non-decreasing, capped at 1); STATEV(2) stores D.
C
C     NOTE: for cohesive elements STRAN/DSTRAN are the total/increment
C     separation and STRESS is the traction.
C
      IMPLICIT NONE
C
      CHARACTER*80 CMNAME
      INTEGER NDI,NSHR,NTENS,NSTATV,NPROPS,NOEL,NPT,LAYER,KSPT,
     1 JSTEP,KINC
      REAL*8 STRESS(NTENS),STATEV(NSTATV),DDSDDE(NTENS,NTENS),
     1 SSE,SPD,SCD,RPL,DDSDDT(NTENS),DRPLDE(NTENS),DRPLDT,
     2 STRAN(NTENS),DSTRAN(NTENS),TIME(2),DTIME,TEMP,DTEMP,
     3 PREDEF(1),DPRED(1),PROPS(NPROPS),COORDS(3),DROT(3,3),
     4 PNEWDT,CELENT,DFGRD0(3,3),DFGRD1(3,3)
C
      REAL*8 TMAX,DC,K0,DEFF,DMAX,D
      INTEGER I,J
C
C     Traction-separation parameters. PROPS(3) = fracture_energy is
C     carried for the interface contract; the bilinear loading branch
C     depends only on max_traction and critical_separation.
      TMAX = PROPS(1)
      DC   = PROPS(2)
      K0   = TMAX/DC
C
C     Effective separation (Euclidean norm of the separation vector).
      DEFF = 0.0D0
      DO I = 1,NTENS
        DEFF = DEFF + STRAN(I)*STRAN(I)
      END DO
      DEFF = SQRT(DEFF)
C
C     History maximum separation and secant damage.
      DMAX = STATEV(1)
      IF (DEFF.GT.DMAX) DMAX = DEFF
      D = DMAX/DC
      IF (D.GT.1.0D0) D = 1.0D0
      STATEV(1) = DMAX
      STATEV(2) = D
C
C     Traction and tangent stiffness.
      IF (DMAX.GE.DC) THEN
        DO I = 1,NTENS
          STRESS(I) = 0.0D0
          DO J = 1,NTENS
            DDSDDE(I,J) = 0.0D0
          END DO
        END DO
      ELSE
        DO I = 1,NTENS
          STRESS(I) = K0*STRAN(I)
        END DO
        DO I = 1,NTENS
          DO J = 1,NTENS
            DDSDDE(I,J) = 0.0D0
          END DO
          DDSDDE(I,I) = K0
        END DO
      END IF
C
      RETURN
      END
"""
    with open(p, "w", encoding="utf-8", newline="") as fh:
        fh.write(txt)
    return p


def export_czm_vumat(
    czm_params: dict,
    *,
    interface_id: str = "interface0",
    out_path: str | Path = "interface_czm_vumat.f",
) -> Path:
    """Emit a bilinear cohesive-zone VUMAT for Abaqus/Explicit (explicit).

    Physics convention (bilinear cohesive law, traction-separation):

    * Loading: ``T = max_traction * (delta / critical_separation)`` for
      ``0 <= delta <= critical_separation``; after the critical separation
      the interface is fully failed (``T = 0``).
    * Fracture energy: ``Gc = 0.5 * max_traction * critical_separation``.
      The three parameters are related, so all three values are embedded in
      the generated source and their relation is written as a comment.
    * Unloading/reloading follows the origin secant. The damage variable is
      ``D = delta_max / critical_separation`` with the history maximum
      separation ``delta_max`` stored in ``STATEV(1)``; ``D`` is monotonic
      non-decreasing and capped at 1.

    The generated Fortran uses the standard Abaqus/Explicit ``VUMAT``
    signature. ``PROPS`` carries, in order, ``max_traction``,
    ``critical_separation``, ``fracture_energy``; ``STATEV(1)`` stores
    ``delta_max``. Because the cohesive ``VUMAT`` receives only separation
    increments, the accumulated total separation components are stored in
    ``STATEV(2..1+NDIR+NSHR)`` (requires ``NSTATEV >= NDIR+NSHR+1``).
    """
    p = Path(out_path)
    sigma = float(czm_params.get("max_traction", 0.0))
    delta = float(czm_params.get("critical_separation", 0.0))
    gc = float(czm_params.get("fracture_energy", 0.0))
    s_sigma = f"{sigma:.6e}"
    s_delta = f"{delta:.6e}"
    s_gc = f"{gc:.6e}"
    txt = f"""      SUBROUTINE VUMAT(
C     Read only (unmodifiable)variables -
     1  NBLOCK, NDIR, NSHR, NSTATEV, NFIELDV, NPROPS, LANNEAL,
     2  STEPTIME, TOTALTIME, DT, CMNAME, PNEWDT, COORDMP, CHARLENGTH,
     3  PROPS, DENSITY, STRAININC, RELSPININC,
     4  TEMPOLD, STRETCHOLD, DEFGRADOLD, FIELDOLD,
     5  STRESSOLD, STATEOLD, ENERINTERNOLD, ENERINELASOLD,
     6  TEMPNEW, STRETCHNEW, DEFGRADNEW, FIELDNEW,
C     Write only (modifiable)variables -
     7  STRESSNEW, STATENEW, ENERINTERNNEW, ENERINELASNEW)
C
      INCLUDE 'vaba_param.inc'
C
C     AtomInterface bilinear cohesive-zone VUMAT (Abaqus/Explicit)
C     Interface id: {interface_id}
C
C     PROPS carries, in order:
C       PROPS(1) = max_traction        = {s_sigma}
C       PROPS(2) = critical_separation = {s_delta}
C       PROPS(3) = fracture_energy     = {s_gc}
C
C     Bilinear cohesive law (traction-separation):
C       loading : T = max_traction * (delta / critical_separation)
C                 for 0 <= delta <= critical_separation
C       failed  : T = 0 for delta > critical_separation
C
C     Fracture energy relation (area under the T-delta curve):
C       Gc = 0.5 * max_traction * critical_separation
C
C     Damage and unloading: D = delta_max / critical_separation (secant
C     to the origin). Unloading/reloading retraces the loading line
C     T = K0 * delta, K0 = max_traction / critical_separation.
C     STATEV(1) stores delta_max (monotonic, capped through D <= 1);
C     STATEV(2..1+NDIR+NSHR) store accumulated separation components.
C     Requires NSTATEV >= NDIR+NSHR+1.
C
      DIMENSION COORDMP(NBLOCK,3), CHARLENGTH(NBLOCK), PROPS(NPROPS),
     1 DENSITY(NBLOCK), STRAININC(NBLOCK,NDIR+NSHR),
     2 RELSPININC(NBLOCK,NSHR), TEMPOLD(NBLOCK),
     3 STRETCHOLD(NBLOCK,NDIR+NSHR),
     4 DEFGRADOLD(NBLOCK,NDIR+NSHR+NSHR),
     5 FIELDOLD(NBLOCK,NFIELDV), STRESSOLD(NBLOCK,NDIR+NSHR),
     6 STATEOLD(NBLOCK,NSTATEV), ENERINTERNOLD(NBLOCK),
     7 ENERINELASOLD(NBLOCK), TEMPNEW(NBLOCK),
     8 STRETCHNEW(NBLOCK,NDIR+NSHR),
     9 DEFGRADNEW(NBLOCK,NDIR+NSHR+NSHR),
     1 FIELDNEW(NBLOCK,NFIELDV), STRESSNEW(NBLOCK,NDIR+NSHR),
     2 STATENEW(NBLOCK,NSTATEV), ENERINTERNNEW(NBLOCK),
     3 ENERINELASNEW(NBLOCK)
C
      CHARACTER*80 CMNAME
C
      REAL*8 TMAX,DC,K0,DEFF,DMAX,D
      INTEGER I,KM
C
C     Traction-separation parameters. PROPS(3) = fracture_energy is
C     carried for the interface contract; the bilinear loading branch
C     depends only on max_traction and critical_separation.
      TMAX = PROPS(1)
      DC   = PROPS(2)
      K0   = TMAX/DC
C
      DO KM = 1,NBLOCK
C
C       Accumulate total separation components from the increments.
        DEFF = 0.0D0
        DO I = 1,NDIR+NSHR
          STATENEW(KM,I+1) = STATEOLD(KM,I+1) + STRAININC(KM,I)
          DEFF = DEFF + STATENEW(KM,I+1)*STATENEW(KM,I+1)
        END DO
        DEFF = SQRT(DEFF)
C
C       History maximum separation and secant damage.
        DMAX = STATEOLD(KM,1)
        IF (DEFF.GT.DMAX) DMAX = DEFF
        D = DMAX/DC
        IF (D.GT.1.0D0) D = 1.0D0
        STATENEW(KM,1) = DMAX
C
C       Traction (STRESSNEW is the cohesive traction).
        IF (DMAX.GE.DC) THEN
          DO I = 1,NDIR+NSHR
            STRESSNEW(KM,I) = 0.0D0
          END DO
        ELSE
          DO I = 1,NDIR+NSHR
            STRESSNEW(KM,I) = K0*STATENEW(KM,I+1)
          END DO
        END IF
C
C       Rate-independent bilinear law: no inelastic energy accumulation.
        ENERINTERNNEW(KM) = 0.0D0
        ENERINELASNEW(KM) = 0.0D0
      END DO
C
      RETURN
      END
"""
    with open(p, "w", encoding="utf-8", newline="") as fh:
        fh.write(txt)
    return p
