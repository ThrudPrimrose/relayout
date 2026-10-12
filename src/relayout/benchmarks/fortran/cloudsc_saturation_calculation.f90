SUBROUTINE compute_saturation_values(KIDIA, KFDIA, KLON, KLEV, ZTP1, PAP, &
    & ZFOEALFA, ZFOEEWMT, ZQSMIX, ZFOEEW, ZQSICE, ZFOEELIQT, ZQSLIQ, &
    & RTT, RETV, R2ES, R3LES, R3IES, R4LES, R4IES, RTICE, RTWAT, RTWAT_RTICE_R) BIND(c)

  use iso_c_binding, only: c_int, c_double
  implicit none


  ! ===============================
  ! Integer arguments (by value)
  ! ===============================
  integer(c_int), value :: KIDIA    ! start column (1)
  integer(c_int), value :: KFDIA    ! end column   (= KLON)
  integer(c_int), value :: KLON     ! horizontal size
  integer(c_int), value :: KLEV     ! vertical size

  ! ===============================
  ! Array arguments
  ! ===============================
  real(c_double), intent(in)  :: ZTP1     (KLON, KLEV)
  real(c_double), intent(in)  :: PAP      (KLON, KLEV)
  real(c_double), intent(out) :: ZFOEALFA (KLON, KLEV)
  real(c_double), intent(out) :: ZFOEEWMT (KLON, KLEV)
  real(c_double), intent(out) :: ZQSMIX   (KLON, KLEV)
  real(c_double), intent(out) :: ZFOEEW   (KLON, KLEV)
  real(c_double), intent(out) :: ZQSICE   (KLON, KLEV)
  real(c_double), intent(out) :: ZFOEELIQT(KLON, KLEV)
  real(c_double), intent(out) :: ZQSLIQ   (KLON, KLEV)

  ! ===============================
  ! Scalar arguments (by value)
  ! ===============================
  real(c_double), value :: RTT
  real(c_double), value :: RETV
  real(c_double), value :: R2ES
  real(c_double), value :: R3LES
  real(c_double), value :: R3IES
  real(c_double), value :: R4LES
  real(c_double), value :: R4IES
  real(c_double), value :: RTICE
  real(c_double), value :: RTWAT
  real(c_double), value :: RTWAT_RTICE_R

  ! ===============================
  ! Local variables
  ! ===============================
  integer(c_int)  :: JK, JL
  real(c_double)  :: PTARE
  real(c_double)  :: ZALFA, ZDELTA
  real(c_double)  :: ZFOEALFA_loc
  real(c_double)  :: ZFOEEWM_loc
  real(c_double)  :: ZFOEELIQ_loc
  real(c_double)  :: ZFOEEICE_loc

  ! Main computation loop
  DO JK = 1, KLEV
    DO JL = 1, KLON
      PTARE = ZTP1(JL, JK)

      ! --- FOEALFA(PTARE) ---
      ZFOEALFA_loc = ((MAX(RTICE, MIN(RTWAT, PTARE)) - RTICE) * RTWAT_RTICE_R)**2
      ZFOEALFA_loc = MIN(1.0D0, ZFOEALFA_loc)
      ZFOEALFA(JL, JK) = ZFOEALFA_loc

      ! --- FOEEWM(PTARE) ---
      ZFOEELIQ_loc = R2ES * EXP(R3LES * (PTARE - RTT) / (PTARE - R4LES))
      ZFOEEICE_loc = R2ES * EXP(R3IES * (PTARE - RTT) / (PTARE - R4IES))

      ZFOEEWM_loc = R2ES * ( ZFOEALFA_loc * EXP(R3LES * (PTARE - RTT) / (PTARE - R4LES)) &
                       + (1.0D0 - ZFOEALFA_loc) * EXP(R3IES * (PTARE - RTT) / (PTARE - R4IES)) )

      ! ZFOEEWMT / ZQSMIX
      ZFOEEWMT(JL, JK) = MIN(ZFOEEWM_loc / PAP(JL, JK), 0.5D0)
      ZQSMIX(JL, JK)   = ZFOEEWMT(JL, JK)
      ZQSMIX(JL, JK)   = ZQSMIX(JL, JK) / (1.0D0 - RETV * ZQSMIX(JL, JK))

      ! --- FOEDELTA(PTARE) ---
      ! FOEDELTA = max(0, sign(1, PTARE-RTT)) -> 1 if PTARE>RTT else 0
      ZDELTA = MAX(0.0D0, SIGN(1.0D0, PTARE - RTT))

      ! ZFOEEW / ZQSICE
      ZFOEEW(JL, JK) = ( ZDELTA      * ZFOEELIQ_loc &
                  + (1.0D0 - ZDELTA) * ZFOEEICE_loc ) / PAP(JL, JK)
      ZFOEEW(JL, JK) = MIN(0.5D0, ZFOEEW(JL, JK))
      ZQSICE(JL, JK) = ZFOEEW(JL, JK) / (1.0D0 - RETV * ZFOEEW(JL, JK))

      ! ZFOEELIQT / ZQSLIQ
      ZFOEELIQT(JL, JK) = MIN(ZFOEELIQ_loc / PAP(JL, JK), 0.5D0)
      ZQSLIQ(JL, JK)    = ZFOEELIQT(JL, JK)
      ZQSLIQ(JL, JK)    = ZQSLIQ(JL, JK) / (1.0D0 - RETV * ZQSLIQ(JL, JK))

    END DO
  END DO

  CALL SYSTEM_CLOCK(t_end)

END SUBROUTINE compute_saturation_values
