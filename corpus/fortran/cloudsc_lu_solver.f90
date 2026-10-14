SUBROUTINE lu_solver_microphysics( KIDIA, KFDIA, KLON, NCLV, ZQLHS, ZQXN) BIND(C, NAME="lu_solver_microphysics")
  USE, INTRINSIC :: ISO_C_BINDING, ONLY: c_int, c_double
  IMPLICIT NONE

  ! Dimensions (scalars passed by value)
  INTEGER(c_int), VALUE :: KIDIA, KFDIA, KLON, NCLV
  
  ! InOut arrays
  REAL(c_double), INTENT(INOUT) :: ZQLHS(KLON, NCLV, NCLV)
  REAL(c_double), INTENT(INOUT) :: ZQXN(KLON, NCLV)
  
  ! Local variables
  INTEGER(c_int) :: JL, JN, JM, IK

  ! LU factorization (per column JL)
  DO JN = 1, NCLV - 1  ! 0
    DO JM = JN + 1, NCLV ! 1

      DO JL = 1, KLON
        ZQLHS(JL, JM, JN) = ZQLHS(JL, JM, JN) / ZQLHS(JL, JN, JN)
      END DO

      ! Adds error
      DO IK = JN + 1, NCLV ! 2
        DO JL = 1, KLON ! 3
          ! 3, 1, 2 = 3, 1, 2 - 3, 1, 0 * 3, 0, 2
          ZQLHS(JL, JM, IK) = ZQLHS(JL, JM, IK) - (ZQLHS(JL, JM, JN) * ZQLHS(JL, JN, IK))
        END DO
      END DO
    END DO
  END DO        

  ! Forward substitution
  ! Adds error
  DO JN = 2, NCLV
    DO JM = 1, JN - 1
      DO JL = 1, KLON
        ZQXN(JL, JN) = ZQXN(JL, JN) - (ZQLHS(JL, JN, JM) * ZQXN(JL, JM))
      END DO
    END DO
  END DO

  ! Backward substitution: last variable
  DO JL = 1, KLON
    ZQXN(JL, NCLV) = ZQXN(JL, NCLV) / ZQLHS(JL, NCLV, NCLV)
  END DO
  
  ! Backward substitution: remaining variables
  DO JN = NCLV - 1, 1, -1
    DO JM = JN + 1, NCLV
      ! adds error
      DO JL = 1, KLON
        ZQXN(JL, JN) = ZQXN(JL, JN) - (ZQLHS(JL, JN, JM) * ZQXN(JL, JM))
      END DO
    END DO
    DO JL = 1, KLON
      ZQXN(JL, JN) = ZQXN(JL, JN) / ZQLHS(JL, JN, JN)
    END DO
  END DO


END SUBROUTINE lu_solver_microphysics